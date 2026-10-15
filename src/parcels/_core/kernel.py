from __future__ import annotations

import types
import warnings
from typing import TYPE_CHECKING

import numpy as np

from parcels._core.basegrid import GridType
from parcels._core.statuscodes import (
    StatusCode,
    _raise_field_interpolation_error,
    _raise_field_out_of_bound_error,
    _raise_field_out_of_bound_surface_error,
    _raise_general_error,
    _raise_grid_searching_error,
    _raise_outside_time_interval_error,
)
from parcels._core.utils.kernel_linting import validate_kernel
from parcels._core.warnings import FieldEvalWarning, KernelWarning
from parcels.kernels import (
    AdvectionAnalytical,
    AdvectionRK45,
)

if TYPE_CHECKING:
    from collections.abc import Callable


ErrorsToThrow = {
    StatusCode.ErrorOutsideTimeInterval: _raise_outside_time_interval_error,
    StatusCode.ErrorOutOfBounds: _raise_field_out_of_bound_error,
    StatusCode.ErrorThroughSurface: _raise_field_out_of_bound_surface_error,
    StatusCode.ErrorInterpolation: _raise_field_interpolation_error,
    StatusCode.ErrorGridSearching: _raise_grid_searching_error,
    StatusCode.Error: _raise_general_error,
}


class Kernel:
    """Kernel object that encapsulates auto-generated code.

    Parameters
    ----------
    kernels :
        list of Kernel functions
    fieldset : parcels.Fieldset
        FieldSet object providing the field information (possibly None)
    pclass :
        pclass object for the kernel particle

    Notes
    -----
    A Kernel is either created from a <function ...> object
    or an ast.FunctionDef object.
    """

    def __init__(
        self,
        kernels: list[types.FunctionType],
        pset,
    ):
        if not isinstance(kernels, list):
            raise ValueError(f"kernels must be a list. Got {kernels=!r}")

        for f in kernels:
            if not isinstance(f, types.FunctionType):
                raise TypeError(f"Argument `kernels` should be a function or list of functions. Got {type(f)}")
            validate_kernel(f)

        if len(kernels) == 0:
            raise ValueError("List of `kernels` should have at least one function.")

        self._fieldset = pset.fieldset
        self._pclass = pset._pclass

        for f in kernels:
            self.check_fieldsets_in_kernels(f)

        self._kernels: list[Callable] = kernels

    @property  #! Ported from v3. To be removed in v4? (/find another way to name kernels in output file)
    def funcname(self):
        ret = ""
        for f in self._kernels:
            ret += f.__name__
        return ret

    @property
    def pclass(self):
        return self._pclass

    @property
    def fieldset(self):
        return self._fieldset

    def remove_deleted(self, pset):
        """Utility to remove all particles that signalled deletion."""
        bool_indices = pset._data["state"] == StatusCode.Delete
        indices = np.where(bool_indices)[0]
        # TODO v4: need to implement ParticleFile writing of deleted particles
        # if len(indices) > 0 and self.fieldset.particlefile is not None:
        #     self.fieldset.particlefile.write(pset, None, indices=indices)
        if len(indices) > 0:
            pset.remove_indices(indices)

    def _position_update(self, particles, fieldset):
        particles.x += particles.dx
        particles.y += particles.dy
        particles.z += particles.dz
        particles.t += particles.dt

        particles.dx = 0
        particles.dy = 0
        particles.dz = 0

        if hasattr(self.fieldset, "RK45_tol"):
            # Update dt in case it's increased in RK45 kernel
            particles.dt = particles.next_dt

    def check_fieldsets_in_kernels(self, kernel):  # TODO v4: this can go into another method? assert_is_compatible()?
        """
        Checks the integrity of the fieldset with the kernels.

        This function is to be called from the derived class when setting up the 'kernel'.
        """
        if self.fieldset is not None:
            if kernel is AdvectionAnalytical:
                if self._fieldset.U.interp_method != "cgrid_velocity":
                    raise NotImplementedError("Analytical Advection only works with C-grids")
                if self._fieldset.U.grid._gtype not in [GridType.CurvilinearZGrid, GridType.RectilinearZGrid]:
                    raise NotImplementedError("Analytical Advection only works with Z-grids in the vertical")
            elif kernel is AdvectionRK45:
                if "next_dt" not in [v.name for v in self.pclass.variables]:
                    raise ValueError('ParticleClass requires a "next_dt" for AdvectionRK45 Kernel.')
                if not hasattr(self.fieldset, "RK45_tol"):
                    warnings.warn(
                        "Setting RK45 tolerance to 10 m. Use fieldset.add_context('RK45_tol', [distance]) to change.",
                        KernelWarning,
                        stacklevel=2,
                    )
                    self.fieldset.add_context("RK45_tol", 10)
                if self.fieldset.U.grid._mesh.is_spherical():
                    self.fieldset.context["RK45_tol"] = self.fieldset.RK45_tol / self.fieldset.U.grid.deg2m
                if not hasattr(self.fieldset, "RK45_min_dt"):
                    warnings.warn(
                        "Setting RK45 minimum timestep to 1 s. Use fieldset.add_context('RK45_min_dt', [timestep]) to change.",
                        KernelWarning,
                        stacklevel=2,
                    )
                    self.fieldset.add_context("RK45_min_dt", 1)
                if not hasattr(self.fieldset, "RK45_max_dt"):
                    warnings.warn(
                        "Setting RK45 maximum timestep to 1 day. Use fieldset.add_context('RK45_max_dt', [timestep]) to change.",
                        KernelWarning,
                        stacklevel=2,
                    )
                    self.fieldset.add_context("RK45_max_dt", 60 * 60 * 24)

    def merge(self, kernel):
        if not isinstance(kernel, type(self)):
            raise TypeError(f"Cannot merge {type(kernel)} with {type(self)}. Both should be of type {type(self)}.")

        assert self.fieldset == kernel.fieldset, "Cannot merge kernels with different fieldsets"
        assert self.pclass == kernel.pclass, "Cannot merge kernels with different particle types"

        return type(self)(
            self._kernels + kernel._kernels,
            self.fieldset,
            self.pclass,
        )

    def execute(self, pset, endtime, dt):
        """Execute this Kernel over a ParticleSet for several timesteps.

        Parameters
        ----------
        pset :
            object of (sub-)type ParticleSet
        endtime :
            endtime of this overall kernel evaluation step
        dt :
            computational integration timestep from pset.execute
        """
        compute_time_direction = 1 if dt > 0 else -1

        pset._data["state"][:] = StatusCode.Success

        while (len(pset) > 0) and np.any(np.isin(pset.state, [StatusCode.Success, StatusCode.Repeat])):
            time_to_endtime = compute_time_direction * (endtime - pset.t)

            evaluate_particles = (pset.state == StatusCode.Success) & (time_to_endtime >= 0)
            if not np.any(evaluate_particles):
                return StatusCode.Success

            # adapt dt to end exactly on endtime
            if compute_time_direction == 1:
                pset.dt = np.maximum(np.minimum(pset.dt, time_to_endtime), 0)
            else:
                pset.dt = np.minimum(np.maximum(pset.dt, -time_to_endtime), 0)

            # run kernels for all particles that need to be evaluated
            for f in self._kernels:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FieldEvalWarning)

                    f(pset[evaluate_particles], self._fieldset)

                    # check for particles that have to be repeated
                    repeat_particles = pset.state == StatusCode.Repeat
                    while np.any(repeat_particles):
                        f(pset[repeat_particles], self._fieldset)
                        repeat_particles = pset.state == StatusCode.Repeat

            # apply position/time update only to particles still in a normal state
            # (particles that signalled Stop*/Delete/errors should not have time/position advanced)
            update_particles = evaluate_particles & (pset.state == StatusCode.Success)
            if np.any(update_particles):
                self._position_update(pset[update_particles], self._fieldset)

            # revert to original dt (unless in RK45 mode)
            if not hasattr(self.fieldset, "RK45_tol"):
                pset._data["dt"][:] = dt

            # Set particle state for particles that reached endtime
            particles_endofloop = (pset.state == StatusCode.Success) & (pset.t == endtime)
            pset[particles_endofloop].state = StatusCode.EndofLoop

            # delete particles that signalled deletion
            self.remove_deleted(pset)

            # check and throw errors
            if np.any(pset.state == StatusCode.StopAllExecution):
                return StatusCode.StopAllExecution

            for error_code, error_func in ErrorsToThrow.items():
                if np.any(pset.state == error_code):
                    inds = pset.state == error_code
                    if error_code == StatusCode.ErrorOutsideTimeInterval:
                        error_func(pset[inds].t)
                    else:
                        error_func(pset[inds].z, pset[inds].y, pset[inds].x)

        return pset
