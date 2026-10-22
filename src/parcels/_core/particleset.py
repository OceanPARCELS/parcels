import datetime  # noqa: ICN001
import sys
import types
import warnings
from collections.abc import Iterable
from contextlib import nullcontext
from typing import Literal

import numpy as np
import xarray as xr
from tqdm import tqdm

from parcels._core.kernel import Kernel
from parcels._core.particle import Particle, create_particle_data
from parcels._core.particlesetview import ParticleSetView
from parcels._core.statuscodes import StatusCode
from parcels._core.utils.time import (
    TimeInterval,
    float_to_datelike,
    timedelta_to_float,
)
from parcels._core.uxgrid import UxGrid
from parcels._core.warnings import ParticleSetWarning
from parcels._core.xgrid import XGrid
from parcels._logger import logger

__all__ = ["ParticleSet"]


class ParticleSet:
    """Class for storing particles and executing kernel over them.

    Please note that this currently only supports fixed size particle sets, meaning that the particle set only
    holds the particles defined on construction. Individual particles can neither be added nor deleted individually,
    and individual particles can only be deleted as a set procedurally (i.e. by changing their state to 'StatusCode.Delete'
    during kernel execution).

    Parameters
    ----------
    fieldset :
        mod:`parcels.fieldset.FieldSet` object from which to sample velocity.
    pclass : parcels.particle.Particle
        Optional object that inherits from :mod:`parcels.particle.Particle` object that defines custom particle
    x :
        List of initial x (longitude) values for particles
    y :
        List of initial y (latitude) values for particles
    z :
        Optional list of initial z values for particles. Default is vertical grid position closest to the surface (z=0)
        that covers all fields in the fieldset. If none of the fields in the fieldset have a vertical grid, z=0 is used.
    t :
        Optional list of initial t (time) values for particles. Default is fieldset.U.grid.time[0]
    repeatdt : datetime.timedelta or float, optional
        Optional interval on which to repeat the release of the ParticleSet. Either timedelta object, or float in seconds.
    particle_ids :
        Optional list of "particle_id" values (integers) for the particle IDs

        Other Variables can be initialised using further arguments (e.g. v=... for a Variable named 'v')
    """

    def __init__(
        self,
        fieldset,
        pclass=Particle,
        *,
        t=None,
        z=None,
        y=None,
        x=None,
        particle_ids=None,
        **kwargs,
    ):
        self._data = None
        self._kernel = None

        self.fieldset = fieldset
        t = np.empty(shape=0) if t is None else np.array(t).flatten()
        y = np.empty(shape=0) if y is None else np.array(y).flatten()
        x = np.empty(shape=0) if x is None else np.array(x).flatten()

        if particle_ids is None:
            particle_ids = np.arange(x.size)

        if z is None:
            minz = None
            for field in self.fieldset.fields.values():
                has_time_varying_z = (isinstance(field.grid, UxGrid) and field.grid.z.ndim == 3) or (
                    isinstance(field.grid, XGrid) and "Z" in field.grid.axes and field.grid._ds["depth"].ndim == 4
                )
                if has_time_varying_z:
                    raise ValueError(
                        f"Field {field.name!r} has a time-varying vertical grid, so there is no default "
                        "particle.z. Pass the particle vertical start locations explicitly with `z`."
                    )
                for depth in field.grid.depth:
                    if minz is None or np.abs(depth) < np.abs(minz):
                        minz = depth
            if minz is not None:
                z = np.ones(x.size) * minz
            else:
                z = np.zeros(x.size)
        else:
            z = np.array(z).flatten()
        assert x.size == y.size and x.size == z.size, "x, y, z don't all have the same lengths"

        if t is None or len(t) == 0:
            # do not set a time yet (because sign_dt not known)
            t = np.array(np.nan)
        elif isinstance(t[0], (np.datetime64, datetime.datetime, datetime.date)) and self.fieldset.time_interval:
            t_dt64 = t.astype("datetime64[ns]") if not isinstance(t[0], np.datetime64) else t
            t = timedelta_to_float(t_dt64 - self.fieldset.time_interval.left)
        elif isinstance(t[0], (np.timedelta64, datetime.timedelta)):
            t = timedelta_to_float(t)
        else:
            raise TypeError("particle t must be a datetime, timedelta, or date object")
        t = np.repeat(t, x.size) if t.size == 1 else t

        assert x.size == t.size, "t and positions (x, y, z) do not have the same lengths."

        if fieldset.time_interval:
            _warn_particle_times_outside_fieldset_time_bounds(t, fieldset.time_interval)

        for kwvar in kwargs:
            kwargs[kwvar] = np.array(kwargs[kwvar]).flatten()
            assert x.size == kwargs[kwvar].size, f"{kwvar} and positions (x, y, z) don't have the same lengths."

        self._data = create_particle_data(
            pclass=pclass,
            nparticles=x.size,
            ngrids=len(fieldset.gridset),
            initial=dict(
                t=t,
                z=z,
                y=y,
                x=x,
                particle_id=particle_ids,
            ),
        )
        self._pclass = pclass

        # update initial values provided on ParticleSet creation # TODO: Wrap this into create_particle_data
        particle_variables = [v.name for v in pclass.variables]
        for kwvar, kwval in kwargs.items():
            if kwvar not in particle_variables:
                raise RuntimeError(f"Particle class does not have Variable {kwvar}")
            self._data[kwvar][:] = kwval

        self._kernel = None

    def __del__(self):
        if self._data is not None and isinstance(self._data, xr.Dataset):
            del self._data
        self._data = None

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index < len(self):
            p = self.__getitem__(self._index)
            self._index += 1
            return p
        raise StopIteration

    def __getattr__(self, name):
        """
        Access a single property of all particles.

        Parameters
        ----------
        name : str
            Name of the property
        """
        return self._data[name]

    def __getitem__(self, index):
        """Get a single particle by index."""
        return ParticleSetView(self._data, index=index, pclass=self._pclass)

    def __setattr__(self, name, value):
        if name in ["_data"]:
            object.__setattr__(self, name, value)
        elif isinstance(self._data, dict) and name in self._data.keys():
            self._data[name][:] = value
        else:
            object.__setattr__(self, name, value)

    @property
    def size(self):
        return len(self)

    # def __repr__(self):
    #     return particleset_repr(self)

    def __len__(self):
        return len(self._data["particle_id"])

    def add(self, particles):
        """Add particles to the ParticleSet. Note that this is an
        incremental add, the particles will be added to the ParticleSet
        on which this function is called.

        Parameters
        ----------
        particles :
            Another ParticleSet containing particles to add to this one.

        Returns
        -------
        type
            The current ParticleSet

        """
        assert particles is not None, (
            f"Trying to add another {type(self)} to this one, but the other one is None - invalid operation."
        )
        assert type(particles) is type(self)

        if len(particles) == 0:
            return

        if len(self) == 0:
            self._data = particles._data
            return

        if isinstance(particles, type(self)):
            if len(self._data["particle_id"]) > 0:
                offset = self._data["particle_id"].max() + 1
            else:
                offset = 0
            particles._data["particle_id"] = particles._data["particle_id"] + offset

        for d in self._data:
            self._data[d] = np.concatenate((self._data[d], particles._data[d]))

        return self

    def __iadd__(self, particles):
        """Add particles to the ParticleSet.

        Note that this is an incremental add, the particles will be added to the ParticleSet
        on which this function is called.

        Parameters
        ----------
        particles :
            Another ParticleSet containing particles to add to this one.

        Returns
        -------
        type
            The current ParticleSet
        """
        self.add(particles)
        return self

    def remove_indices(self, indices):
        """Method to remove particles from the ParticleSet, based on their `indices`."""
        for d in self._data:
            self._data[d] = np.delete(self._data[d], indices, axis=0)

    def populate_indices(self):
        """Pre-populate guesses of particle ei (element id) indices"""
        for i, grid in enumerate(self.fieldset.gridset):
            grid_positions = grid.search(self.z, self.y, self.x)
            self._data["ei"][:, i] = grid.ravel_index(
                {
                    "X": grid_positions["X"]["index"],
                    "Y": grid_positions["Y"]["index"],
                    "Z": grid_positions["Z"]["index"],
                }
            )

    @classmethod
    def from_particlefile(cls, fieldset, pclass, filename, restart=True, restarttime=None, **kwargs):
        """Initialise the ParticleSet from a zarr ParticleFile.
        This creates a new ParticleSet based on locations of all particles written
        in a zarr ParticleFile at a certain time. Particle IDs are preserved if restart=True

        Parameters
        ----------
        fieldset : parcels.fieldset.FieldSet
            mod:`parcels.fieldset.FieldSet` object from which to sample velocity
        pclass :
            Particle class. May be a parcels.particle.Particle class as defined in parcels, or a subclass defining a custom particle.
        filename : str
            Name of the particlefile from which to read initial conditions
        restart : bool
            BSignal if pset is used for a restart (default is True).
            In that case, Particle IDs are preserved.
        restarttime :
            time at which the Particles will be restarted. Default is the last time written.
            Alternatively, restarttime could be a time value (including np.datetime64) or
            a callable function such as np.nanmin. The last is useful when running with dt < 0.
        repeatdt : datetime.timedelta or float, optional
            Optional interval on which to repeat the release of the ParticleSet. Either timedelta object, or float in seconds.
        **kwargs :
            Keyword arguments passed to the particleset constructor.
        """
        raise NotImplementedError(
            "ParticleSet.from_particlefile is not yet implemented in v4."
        )  # TODO implement this when ParticleFile is implemented in v4

    def data_indices(self, variable_name, compare_values, invert=False):
        """Get the indices of all particles where the value of `variable_name` equals (one of) `compare_values`.

        Parameters
        ----------
        variable_name : str
            Name of the variable to check.
        compare_values :
            Value or list of values to compare to.
        invert :
            Whether to invert the selection. I.e., when True,
            return all indices that do not equal (one of)
            `compare_values`. (Default value = False)

        Returns
        -------
        np.ndarray
            Numpy array of indices that satisfy the test.

        """
        compare_values = (
            np.array([compare_values]) if type(compare_values) not in [list, dict, np.ndarray] else compare_values
        )
        return np.where(np.isin(self._data[variable_name], compare_values, invert=invert))[
            0
        ]  # TODO check if this can be faster with xarray indexing?

    @property
    def _error_particles(self):
        """Get indices of all particles that are in an error state.

        Returns
        -------
        indices
            Indices of error particles.
        """
        return np.where(self._data["state"] != StatusCode.Success)[0]

    @property
    def _num_error_particles(self):
        """Get the number of particles that are in an error state.

        Returns
        -------
        int
            Number of error particles.
        """
        return int(np.sum(self._data["state"] != StatusCode.Success))

    def execute(
        self,
        kernels,
        dt: datetime.timedelta | np.timedelta64 | float,
        endtime: np.timedelta64 | np.datetime64 | None = None,
        runtime: datetime.timedelta | np.timedelta64 | float | None = None,
        output_file=None,
        verbose_progress=True,
    ):
        """Execute a given kernel function over the particle set for multiple timesteps.

        Optionally also provide sub-timestepping
        for particle output.

        Parameters
        ----------
        kernels :
            List of Kernel functions to execute. This can be the name of a
            defined Python function or a :class:`parcels.kernel.Kernel` object.
        dt (np.timedelta64 or float):
            Timestep interval (as a np.timedelta64 object of float in seconds) to be passed to the kernel.
            Use a negative value for a backward-in-time simulation.
        endtime (np.datetime64 or np.timedelta64): :
            End time for the timestepping loop. If a np.timedelta64 is provided, it is interpreted as the total simulation time. In this case,
            the absolute end time is the start of the fieldset's time interval plus the np.timedelta64.
            If a datetime is provided, it is interpreted as the absolute end time of the simulation.
        runtime (np.timedelta64 or float):
            The duration of the simulation execution. Must be a float (in seconds) or a np.timedelta64 object and is required to be set when the `fieldset.time_interval` is not defined.
            If the `fieldset.time_interval` is defined and the runtime is provided, the end time will be the start of the fieldset's time interval plus the runtime.
        output_file :
            mod:`parcels.particlefile.ParticleFile` object for particle output (Default value = None)
        verbose_progress : bool
            Boolean for providing a progress bar for the kernel execution loop. (Default value = True)

        Notes
        -----
        ``ParticleSet.execute()`` acts as the main entrypoint for simulations, and provides the simulation time-loop. This method encapsulates the logic controlling the switching between kernel execution, output file writing, reading in fields for new timesteps, adding new particles to the simulation domain, stopping the simulation, and executing custom functions (``postIterationCallbacks`` provided by the user).
        """
        # check if particleset is empty. If so, return immediately
        if len(self) == 0:
            return

        if isinstance(kernels, types.FunctionType):
            kernels = [kernels]

        self._kernel = Kernel(kernels, self)

        if output_file is not None:
            output_file.set_metadata(self.fieldset.gridset[0]._mesh)
            output_file.set_time_interval(self.fieldset.time_interval)
            output_file.metadata["parcels_kernels"] = self._kernel.funcname

        dt, sign_dt = _convert_dt_to_float(dt)
        self._data["dt"][:] = dt

        runtime = _convert_runtime_to_float(runtime)

        start_time, end_time = _get_simulation_start_and_end_times(
            self.fieldset.time_interval, self._data["t"], runtime, endtime, sign_dt
        )

        # Set the time of the particles if it hadn't been set on initialisation
        if np.isnan(self._data["t"]).any():
            self._data["t"][:] = start_time

        outputdt = output_file.outputdt if output_file else None
        _warn_outputdt_release_desync(outputdt, start_time, self._data["t"][:])

        # Set up pbar
        if output_file:
            logger.info(f"Output files are stored in {output_file.path}")

        if verbose_progress:
            pbar = tqdm(
                total=sign_dt * (end_time - start_time),
                file=sys.stdout,
                bar_format="{desc} {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}, {rate_fmt}]",
            )
            pbar.set_description_str(
                "Integration time: " + str(float_to_datelike(start_time, self.fieldset.time_interval))
            )

        next_output = None
        if output_file:
            # Write the initial condition
            output_file.write(self, start_time)
            # Increment the next_output
            next_output = start_time + outputdt * sign_dt

        time = start_time

        with output_file if output_file is not None else nullcontext():
            while sign_dt * (time - end_time) < 0:
                if next_output is not None:
                    f = min if sign_dt > 0 else max
                    next_time = f(next_output, end_time)
                else:
                    next_time = end_time

                self._kernel.execute(self, endtime=next_time, dt=dt)

                if next_output is not None:
                    if np.abs(next_time - next_output) < 0.001:
                        if output_file:
                            output_file.write(self, next_output)
                        if np.isfinite(outputdt):
                            next_output += outputdt * sign_dt

                if verbose_progress:
                    pbar.set_description_str(
                        "Integration time: " + str(float_to_datelike(time, self.fieldset.time_interval))
                    )
                    pbar.update(sign_dt * (next_time - time))

                time = next_time

        if verbose_progress:
            pbar.close()


def _warn_outputdt_release_desync(outputdt: float, starttime: float, release_times: Iterable[float]):
    """Gives the user a warning if the release time isn't a multiple of outputdt."""
    if outputdt and any((np.isfinite(t) and (t - starttime) % outputdt != 0) for t in release_times):
        warnings.warn(
            "Some of the particles have a start time difference that is not a multiple of outputdt. "
            "This could cause the first output of some of the particles that start later "
            "in the simulation to be at a different time than expected.",
            ParticleSetWarning,
            stacklevel=2,
        )


def _warn_particle_times_outside_fieldset_time_bounds(release_times: np.ndarray, time: TimeInterval):
    if np.isnan(release_times).all():
        return

    if np.any(release_times < 0) or np.any(release_times > timedelta_to_float(time.right - time.left)):
        warnings.warn(
            "Some particles are set to be released outside the FieldSet's executable time domain.",
            ParticleSetWarning,
            stacklevel=2,
        )


def _convert_dt_to_float(dt):
    try:
        dt = timedelta_to_float(dt)
        assert dt is not None
        sign_dt = np.sign(dt).astype(int)
        assert sign_dt in [-1, 1]
    except (ValueError, TypeError, AssertionError) as e:
        raise ValueError(f"dt must be a non-zero datetime.timedelta or np.timedelta64 object, got {dt=!r}") from e

    return dt, sign_dt


def _convert_runtime_to_float(runtime):
    if runtime is not None:
        try:
            runtime = timedelta_to_float(runtime)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"The runtime must be a datetime.timedelta, np.timedelta64 or float object. Got {type(runtime)}"
            ) from e
        if runtime < 0:
            raise ValueError(f"The runtime must be a non-negative timedelta or float. Got {runtime=!r}")

    return runtime


def _get_simulation_start_and_end_times(
    time_interval: TimeInterval,
    particle_release_times: np.ndarray,
    runtime: np.timedelta64 | None,
    endtime: np.datetime64 | None,
    sign_dt: Literal[-1, 1],
) -> tuple[np.datetime64, np.datetime64]:
    if runtime is not None and endtime is not None:
        raise ValueError(
            f"runtime and endtime are mutually exclusive - provide one or the other. Got {runtime=!r}, {endtime=!r}"
        )

    if runtime is None and time_interval is None:
        raise ValueError("The runtime must be provided when the time_interval is not defined for a fieldset.")

    if runtime is None and endtime is None:
        raise ValueError("Either runtime or endtime must be provided.")

    if sign_dt == 1:
        first_release_time = particle_release_times.min()
    else:
        first_release_time = particle_release_times.max()

    if (time_interval is not None) and (endtime is not None):
        if type(endtime) != type(time_interval.left):  # noqa: E721
            raise ValueError(
                f"The endtime must be of the same type as the fieldset.time_interval start time. Got {endtime=!r} with {time_interval=!r}"
            )
        if endtime not in time_interval:
            msg = (
                f"Calculated/provided end time of {endtime!r} is not in fieldset time interval {time_interval!r}. Either reduce your runtime, modify your "
                "provided endtime, or change your release timing."
                "Important info:\n"
                f"    First particle release: {first_release_time!r}\n"
                f"    runtime: {runtime!r}\n"
                f"    (calculated) endtime: {endtime!r}"
            )
            raise ValueError(msg)

        endtime = timedelta_to_float(endtime - time_interval.left)

    start_time = _get_start_time(first_release_time, time_interval, sign_dt, runtime)

    if isinstance(runtime, np.timedelta64):
        runtime = timedelta_to_float(runtime)

    if endtime is None:
        endtime = start_time + sign_dt * runtime

    return start_time, endtime


def _get_start_time(first_release_time, time_interval, sign_dt, runtime):
    if time_interval is None:
        time_interval = TimeInterval(np.timedelta64(0, "s"), right=np.timedelta64(int(runtime * 1e9), "ns"))

    if sign_dt == 1:
        fieldset_start = 0.0
    else:
        fieldset_start = timedelta_to_float(time_interval.right - time_interval.left)

    start_time = first_release_time if not np.isnan(first_release_time) else fieldset_start
    return start_time
