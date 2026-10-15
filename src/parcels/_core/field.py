from __future__ import annotations

import warnings
from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

from parcels._core.index_search import GRID_SEARCH_ERROR, LEFT_OUT_OF_BOUNDS, RIGHT_OUT_OF_BOUNDS, _search_time_index
from parcels._core.particleset import ParticleSet
from parcels._core.particlesetview import ParticleSetView
from parcels._core.statuscodes import (
    AllParcelsErrorCodes,
    StatusCode,
)
from parcels._core.utils.string import _assert_str_and_python_varname
from parcels._core.uxgrid import UxGrid
from parcels._core.warnings import FieldEvalWarning
from parcels._core.xgrid import XGrid
from parcels._typing import VectorType
from parcels.interpolators._base import ScalarInterpolator, VectorInterpolator

if TYPE_CHECKING:
    from parcels._core.model import ModelData


__all__ = ["Field", "VectorField"]


def _deal_with_errors(error, key, vector_type: VectorType):
    if isinstance(key, ParticleSetView):
        key.state = AllParcelsErrorCodes[type(error)]
    elif isinstance(key[-1], ParticleSetView):
        key[-1].state = AllParcelsErrorCodes[type(error)]
    else:
        raise RuntimeError(f"{error}. Error could not be handled because particles was not part of the Field Sampling.")

    if vector_type == "3D":
        return (0, 0, 0)
    elif vector_type == "2D":
        return (0, 0)
    else:
        return 0


class Field:
    """The Field class that holds scalar field data.
    The `Field` object is a wrapper around a xarray.DataArray or uxarray.UxDataArray object.
    Additionally, it holds a dynamic Callable procedure that is used to interpolate the field data.
    During initialization, the user is required to supply a custom interpolation method that is used
    to interpolate the field data, so long as the interpolation method has the correct signature.

    Notes
    -----
    The xarray.DataArray or uxarray.UxDataArray object contains the field data and metadata.

    * dims: (time, [nz1 | nz], [face_lat | node_lat | edge_lat], [face_lon | node_lon | edge_lon])
    * attrs: (location, mesh, mesh)

    When using a xarray.DataArray object:

    * The xarray.DataArray object must have the "location" and "mesh" attributes set.
    * The "location" attribute must be set to one of the following to define which pairing of points a field is associated with:
        * "node"
        * "face"
        * "x_edge"
        * "y_edge"

    * For an A-Grid, the "location" attribute must be set to / is assumed to be "node" (node_lat,node_lon).
    * For a C-Grid, the "location" setting for a field has the following interpretation:
        * "node" ~> the field is associated with the vorticity points (node_lat, node_lon)
        * "face" ~> the field is associated with the tracer points (face_lat, face_lon)
        * "x_edge" ~> the field is associated with the u-velocity points (face_lat, node_lon)
        * "y_edge" ~> the field is associated with the v-velocity points (node_lat, face_lon)

    When using a uxarray.UxDataArray object:

    * The uxarray.UxDataArray.UxGrid object must have the "Conventions" attribute set to "UGRID-1.0"
      and the uxarray.UxDataArray object must comply with the UGRID conventions.
      See https://ugrid-conventions.github.io/ugrid-conventions/ for more information.

    """

    def __init__(
        self,
        name: str,
        model: ModelData,
    ):
        # TODO PR: Enable isinstance check once ModelData is moved to abc.ModelData
        # if not isinstance(model, "ModelData"):
        #     raise ValueError(
        #         f"Expected `model` to be a Parcels ModelData object. Got {type(model)}."
        #     )

        _assert_str_and_python_varname(name)

        self.name = name
        self.model = model

        self.igrid = -1  # Default the grid index to -1

    @property
    def data(self):
        return self.model.field_data(self.name)

    @property
    def grid(self):  # TODO PR: Remove in favour of referencing model grid directly
        return self.model.grid

    @property
    def time_interval(self):  # TODO PR: Remove in favour of referencing model time_interval directly
        if "time" not in self.model.data[self.name].dims:
            # This field does not have a time-dimension
            return None
        return self.model.time_interval

    def __repr__(self):
        return f"Field(name={self.name!r}, model={self.model})"

    @property
    def interp_method(self):
        try:
            return self.model.field_to_interpolator[self.name]
        except KeyError as e:
            raise AttributeError(
                f"{type(self).__name__} doesn't have an interp_method defined for it. Use `.interp_method = ...`"
            ) from e

    @interp_method.setter
    def interp_method(self, value):
        # Setting the interpolation method dynamically
        if not isinstance(value, ScalarInterpolator):
            raise ValueError(f"interp_method must be a `ScalarInterpolator` object. Got {type(value)=!r}")
        self.model.field_to_interpolator[self.name] = value

    def _check_velocitysampling(self):
        if self.name in ["U", "V", "W"]:
            warnings.warn(
                "Sampling of velocities should normally be done using fieldset.UV or fieldset.UVW object; tread carefully",
                RuntimeWarning,
                stacklevel=2,
            )

    def eval(self, t: datetime, z, y, x, particles=None):
        """Interpolate field values in space and time.

        Parameters
        ----------
        t : float or array-like
            Time(s) at which to sample the field.
        z, y, x : scalar or array-like
            Vertical (z), latitudinal (y) and longitudinal (x) positions to sample.
            Inputs are promoted to 1-D arrays internally.
        particles : ParticleSet, optional
            If provided, used to associate results with particle indices and to
            update particle state and element indices. Defaults to None.

        Returns
        -------
        (value) : float or array-like
            The interpolated value as a numpy.ndarray (or scalar) with the same
            broadcasted shape as the input coordinates.

        Notes
        -----
        - Particle states are updated for out-of-bounds, search errors and NaN
          interpolation values.
        """
        if particles is None:
            _ei = None
        else:
            _ei = particles.ei[:, self.igrid]
        z = np.atleast_1d(z)
        y = np.atleast_1d(y)
        x = np.atleast_1d(x)

        particle_positions, grid_positions = _get_positions(self, t, z, y, x, particles, _ei)

        value = self.interp_method.interp(particle_positions, grid_positions, self)

        _update_particle_states_interp_value(particles, value)
        _mask_outofbounds_values(grid_positions, value)

        return value

    def __getitem__(self, key):
        self._check_velocitysampling()
        try:
            if isinstance(key, (ParticleSetView, ParticleSet)):
                return self.eval(key.t, key.z, key.y, key.x, key)
            else:
                return self.eval(*key)
        except tuple(AllParcelsErrorCodes.keys()) as error:
            return _deal_with_errors(error, key, vector_type=None)


class VectorField:
    """VectorField class that holds vector field data needed to execute particles."""

    def __init__(
        self,
        name: str,
        U: Field,  # noqa: N803
        V: Field,  # noqa: N803
        W: Field | None = None,  # noqa: N803
        interp_method: VectorInterpolator | None = None,
    ):
        if interp_method is None:
            raise ValueError("interp_method must be provided for VectorField initialization.")

        _assert_str_and_python_varname(name)
        self.name = name
        self.U = U
        self.V = V
        self.W = W
        self.grid = U.grid
        self.igrid = U.igrid

        if W is None:
            _assert_same_time_interval((U, V))
        else:
            _assert_same_time_interval((U, V, W))

        self.time_interval = U.time_interval
        self.vector_type: VectorType
        if self.W:
            self.vector_type = "3D"
        else:
            self.vector_type = "2D"

        if not isinstance(interp_method, VectorInterpolator):
            raise ValueError(f"interp_method must be a `VectorInterpolator` object. Got {type(interp_method)=!r}")

        self._interp_method = interp_method

    # def __repr__(self):
    #     return vectorfield_repr(self)

    @property
    def interp_method(self):
        return self._interp_method

    @interp_method.setter
    def interp_method(self, method: VectorInterpolator):
        if not isinstance(method, VectorInterpolator):
            raise ValueError(f"method must be a `VectorInterpolator` object. Got {type(method)=!r}")
        self._interp_method = method

    def eval(self, t: datetime, z, y, x, particles=None):
        """Interpolate vectorfield values in space and time.

        Parameters
        ----------
        t : float or array-like
            Time(s) at which to sample the field.
        z, y, x : scalar or array-like
            Vertical (z), latitudinal (y) and longitudinal (x) positions to sample.
            Inputs are promoted to 1-D arrays internally.
        particles : ParticleSet, optional
            If provided, used to associate results with particle indices and to
            update particle state and element indices. Defaults to None.

        Returns
        -------
        (u, v, (w,)) : tuple or array-like
            The interpolated velocity components: (u, v) for 2D vectors or (u, v, w)
            for 3D vectors. Each element is a numpy.ndarray (or scalar) with the same
            broadcasted shape as the input coordinates.

        Notes
        -----
        - Particle states are updated for out-of-bounds, search errors and NaN
          interpolation values.
        """
        if particles is None:
            _ei = None
        else:
            _ei = particles.ei[:, self.igrid]
        z = np.atleast_1d(z)
        y = np.atleast_1d(y)
        x = np.atleast_1d(x)

        particle_positions, grid_positions = _get_positions(self.U, t, z, y, x, particles, _ei)

        (u, v, w) = self._interp_method.interp(particle_positions, grid_positions, self)

        for vel in (u, v, w):
            _update_particle_states_interp_value(particles, vel)
            _mask_outofbounds_values(grid_positions, vel)

        if "3D" == self.vector_type:
            return (u, v, w)
        else:
            return (u, v)

    def __getitem__(self, key):
        try:
            if isinstance(key, (ParticleSetView, ParticleSet)):
                return self.eval(key.t, key.z, key.y, key.x, key)
            else:
                return self.eval(*key)
        except tuple(AllParcelsErrorCodes.keys()) as error:
            return _deal_with_errors(error, key, vector_type=self.vector_type)


def _update_particles_ei(particles, grid_positions: dict, field: Field):
    """Update the element index (ei) of the particles"""
    if particles is not None:
        if isinstance(field.grid, XGrid):
            particles.ei[:, field.igrid] = field.grid.ravel_index(
                {
                    "X": grid_positions["X"]["index"],
                    "Y": grid_positions["Y"]["index"],
                    "Z": grid_positions["Z"]["index"],
                }
            )
        elif isinstance(field.grid, UxGrid):
            particles.ei[:, field.igrid] = field.grid.ravel_index(
                {
                    "Z": grid_positions["Z"]["index"],
                    "FACE": grid_positions["FACE"]["index"],
                }
            )


def _update_particle_states_position(particles, grid_positions: dict):
    """Update the particle states based on the position dictionary."""
    if particles:
        for dim in ["X", "Y", "FACE"]:
            if dim in grid_positions:
                particles.state = np.maximum(
                    np.where(grid_positions[dim]["index"] == -1, StatusCode.ErrorOutOfBounds, particles.state),
                    particles.state,
                )
                particles.state = np.maximum(
                    np.where(
                        grid_positions[dim]["index"] == GRID_SEARCH_ERROR,
                        StatusCode.ErrorGridSearching,
                        particles.state,
                    ),
                    particles.state,
                )
        if "Z" in grid_positions:
            particles.state = np.maximum(
                np.where(
                    grid_positions["Z"]["index"] == RIGHT_OUT_OF_BOUNDS, StatusCode.ErrorOutOfBounds, particles.state
                ),
                particles.state,
            )
            particles.state = np.maximum(
                np.where(
                    grid_positions["Z"]["index"] == LEFT_OUT_OF_BOUNDS, StatusCode.ErrorThroughSurface, particles.state
                ),
                particles.state,
            )


def _mask_outofbounds_values(grid_positions: dict, value):
    mask = np.zeros(value.shape, dtype=bool)
    for dim in ["X", "Y", "Z", "FACE"]:
        if dim in grid_positions:
            mask[grid_positions[dim]["index"] < 0] = True
    if np.any(mask):
        warnings.warn(
            "Some interpolated values are out-of-bounds. These values are set to 0. Treat carefully.",
            FieldEvalWarning,
            stacklevel=2,
        )
        value[mask] = 0.0


def _update_particle_states_interp_value(particles, value):
    """Update the particle states based on the interpolated value, but only if state is not an Error already."""
    if particles:
        particles.state = np.maximum(
            np.where(np.isnan(value), StatusCode.ErrorInterpolation, particles.state), particles.state
        )


def _assert_same_time_interval(fields: Sequence[Field]) -> None:
    if len(fields) == 0:
        return

    reference_time_interval = fields[0].time_interval

    for field in fields[1:]:
        if field.time_interval != reference_time_interval:
            raise ValueError(
                f"Fields must have the same time domain. {fields[0].name}: {reference_time_interval}, {field.name}: {field.time_interval}"
            )


def _get_positions(field: Field, t, z, y, x, particles, _ei) -> tuple[dict, dict]:
    """Initialize and populate particle_positions and grid_positions dictionaries"""
    if np.any(np.isnan(t)):
        nan_indices = np.where(np.isnan(t))[0]
        raise ValueError(f"Time values for particles with indices {nan_indices} cannot be NaN.")
    particle_positions = {"t": t, "z": z, "y": y, "x": x}
    grid_positions = {}
    time_positions = _search_time_index(field, t)
    grid_positions.update(time_positions)
    grid_positions.update(
        field.grid.search(z, y, x, ei=_ei, ti=time_positions["T"]["index"], tau=time_positions["T"]["bcoord"])
    )
    _update_particles_ei(particles, grid_positions, field)
    _update_particle_states_position(particles, grid_positions)
    return particle_positions, grid_positions
