from datetime import date
from datetime import datetime
from datetime import timedelta as delta

import sys
import numpy as np
import xarray as xr

from parcels.grid import GridCode
from parcels.kernel import Kernel
from parcels.particle import JITParticle
from parcels.particlefile import ParticleFile
from parcels.tools.statuscodes import StateCode
from .baseparticleset import BaseParticleSet
from .soa import ParticleCollectionSOA
from .soa import ParticleCollectionIteratorSOA
from parcels.tools.converters import _get_cftime_calendars
from parcels.tools.statuscodes import OperationCode
from parcels.tools.loggers import logger
try:
    from mpi4py import MPI
except:
    MPI = None
if MPI:
    try:
        from sklearn.cluster import KMeans  # noqa
    except:
        raise EnvironmentError('sklearn needs to be available if MPI is installed. '
                               'See http://oceanparcels.org/#parallel_install for more information')

__all__ = ['ParticleSet']


def _to_write_particles(pd, time):
    """We don't want to write a particle that is not started yet.
    Particle will be written if particle.time is between time-dt/2 and time+dt (/2)
    # np.greater(time + np.abs(pd['dt']/2), pd['time'], where=np.isfinite(pd['time']))
    # & np.less(pd['time'], time + np.abs(pd['dt'] / 2), where=np.isfinite(pd['time']))
    """
    return (np.less_equal(time - np.abs(pd['dt']/2), pd['time'], where=np.isfinite(pd['time']))
            & np.greater_equal(time + np.abs(pd['dt'] / 2), pd['time'], where=np.isfinite(pd['time']))
            & (np.isfinite(pd['id']))
            & (np.isfinite(pd['time'])))

def _is_particle_started_yet(pd, time):
    """We don't want to write a particle that is not started yet.
    Particle will be written if:
      * particle.time is equal to time argument of pfile.write()
      * particle.time is before time (in case particle was deleted between previous export and current one)
    """
    return np.less_equal(pd['dt']*pd['time'], pd['dt']*time) | np.isclose(pd['time'], time)
    # return (particle.dt*particle.time <= particle.dt*time or np.isclose(particle.time, time))


def convert_to_array(var):
    # Convert lists and single integers/floats to one-dimensional numpy arrays
    if isinstance(var, np.ndarray):
        return var.flatten()
    elif isinstance(var, (int, float, np.float32, np.int32)):
        return np.array([var])
    else:
        return np.array(var)


def convert_to_reltime(time):
    if isinstance(time, np.datetime64) or (hasattr(time, 'calendar') and time.calendar in _get_cftime_calendars()):
        return True
    return False


class ParticleSetSOA(BaseParticleSet):
    """Container class for storing particle and executing kernel over them.

    Please note that this currently only supports fixed size particle sets.

    :param fieldset: :mod:`parcels.fieldset.FieldSet` object from which to sample velocity.
           While fieldset=None is supported, this will throw a warning as it breaks most Parcels functionality
    :param pclass: Optional :mod:`parcels.particle.JITParticle` or
                 :mod:`parcels.particle.ScipyParticle` object that defines custom particle
    :param lon: List of initial longitude values for particles
    :param lat: List of initial latitude values for particles
    :param depth: Optional list of initial depth values for particles. Default is 0m
    :param time: Optional list of initial time values for particles. Default is fieldset.U.grid.time[0]
    :param repeatdt: Optional interval (in seconds) on which to repeat the release of the ParticleSet
    :param lonlatdepth_dtype: Floating precision for lon, lat, depth particle coordinates.
           It is either np.float32 or np.float64. Default is np.float32 if fieldset.U.interp_method is 'linear'
           and np.float64 if the interpolation method is 'cgrid_velocity'
    :param partitions: List of cores on which to distribute the particles for MPI runs. Default: None, in which case particles
           are distributed automatically on the processors

    Other Variables can be initialised using further arguments (e.g. v=... for a Variable named 'v')
    """

    # ==== already user-exposed ==== #
    def __init__(self, fieldset=None, pclass=JITParticle, lon=None, lat=None, depth=None, time=None, repeatdt=None, lonlatdepth_dtype=None, pid_orig=None, **kwargs):
        super(ParticleSetSOA, self).__init__()
        self.fieldset = fieldset
        if self.fieldset is None:
            logger.warning_once("No FieldSet provided in ParticleSet generation. "
                                "This breaks most Parcels functionality")
        else:
            self.fieldset.check_complete()
        partitions = kwargs.pop('partitions', None)

        lon = np.empty(shape=0) if lon is None else convert_to_array(lon)
        lat = np.empty(shape=0) if lat is None else convert_to_array(lat)

        if isinstance(pid_orig, (type(None), type(False))):
            pid_orig = np.arange(lon.size)
        # pid = pid_orig + pclass.lastID

        if depth is None:
            mindepth = self.fieldset.gridset.dimrange('depth')[0] if self.fieldset is not None else 0
            depth = np.ones(lon.size) * mindepth
        else:
            depth = convert_to_array(depth)
        assert lon.size == lat.size and lon.size == depth.size, (
            'lon, lat, depth don''t all have the same lenghts')

        if time is None:
            raise RuntimeError("Particle Set is created with 'time=None' - time-parameter invalid.")

        time = convert_to_array(time)
        time = np.repeat(time, lon.size) if time.size == 1 else time

        if time.size > 0 and type(time[0]) in [datetime, date]:
            time = np.array([np.datetime64(t) for t in time])
        self.time_origin = fieldset.time_origin if self.fieldset is not None else 0
        if time.size > 0 and isinstance(time[0], np.timedelta64) and not self.time_origin:
            raise NotImplementedError('If fieldset.time_origin is not a date, time of a particle must be a double')
        time = np.array([self.time_origin.reltime(t) if convert_to_reltime(t) else t for t in time])
        assert lon.size == time.size, (
            'time and positions (lon, lat, depth) don''t have the same lengths.')

        if lonlatdepth_dtype is None:
            if fieldset is not None:
                lonlatdepth_dtype = self.lonlatdepth_dtype_from_field_interp_method(fieldset.U)
            else:
                lonlatdepth_dtype = np.float32
        assert lonlatdepth_dtype in [np.float32, np.float64], \
            'lon lat depth precision should be set to either np.float32 or np.float64'

        # if partitions is not None and partitions is not False:
        #     partitions = convert_to_array(partitions)

        for kwvar in kwargs:
            kwargs[kwvar] = convert_to_array(kwargs[kwvar])
            assert lon.size == kwargs[kwvar].size, (
                '%s and positions (lon, lat, depth) don''t have the same lengths.' % kwvar)

        self.repeatdt = repeatdt.total_seconds() if isinstance(repeatdt, delta) else repeatdt
        if self.repeatdt:
            if self.repeatdt <= 0:
                raise('Repeatdt should be > 0')
            if time[0] and not np.allclose(time, time[0]):
                raise ('All Particle.time should be the same when repeatdt is not None')
            self.repeat_starttime = time[0]
            self.repeatlon = lon
            self.repeatlat = lat
            self.repeatdepth = depth
            self.repeatpclass = pclass
            self.repeatkwargs = kwargs

        ngrids = fieldset.gridset.size if fieldset is not None else 1
        self._collection = ParticleCollectionSOA(pclass, lon=lon, lat=lat, depth=depth, time=time, lonlatdepth_dtype=lonlatdepth_dtype, partitions=partitions, pid_orig=pid_orig, ngrid=ngrids, **kwargs)


        if self.repeatdt:
            if self._collection.data['time'][0] and not np.allclose(self._collection.data['time'], self._collection.data['time'][0]):
                raise ('All Particle.time should be the same when repeatdt is not None')
            self.repeat_starttime = self._collection.data['time'][0]
            self.repeatlon = self._collection.data['lon']
            self.repeatlat = self._collection.data['lat']
            self.repeatdepth = self._collection.data['depth']
            for kwvar in kwargs:
                self.repeatkwargs[kwvar] = self._collection.data[kwvar]

        # offset = np.max(pid) if len(pid) > 0 else -1
        # if MPI:
        #     mpi_comm = MPI.COMM_WORLD
        #     mpi_rank = mpi_comm.Get_rank()
        #     mpi_size = mpi_comm.Get_size()

        #     if lon.size < mpi_size and mpi_size > 1:
        #         raise RuntimeError('Cannot initialise with fewer particles than MPI processors')

        #     if mpi_size > 1:
        #         if partitions is not False:
        #             if partitions is None:
        #                 if mpi_rank == 0:
        #                     coords = np.vstack((lon, lat)).transpose()
        #                     kmeans = KMeans(n_clusters=mpi_size, random_state=0).fit(coords)
        #                     partitions = kmeans.labels_
        #                 else:
        #                     partitions = None
        #                 partitions = mpi_comm.bcast(partitions, root=0)
        #             elif np.max(partitions) >= mpi_size:
        #                 raise RuntimeError('Particle partitions must vary between 0 and the number of mpi procs')
        #             lon = lon[partitions == mpi_rank]
        #             lat = lat[partitions == mpi_rank]
        #             time = time[partitions == mpi_rank]
        #             depth = depth[partitions == mpi_rank]
        #             pid = pid[partitions == mpi_rank]
        #             for kwvar in kwargs:
        #                 kwargs[kwvar] = kwargs[kwvar][partitions == mpi_rank]
        #         offset = MPI.COMM_WORLD.allreduce(offset, op=MPI.MAX)

        # pclass.setLastID(offset+1)
        # pclass.set_lonlatdepth_dtype(self.lonlatdepth_dtype)
        # self.ptype = pclass.getPType()

        if self.repeatdt:
            if not hasattr(self, 'repeatpid'):
                # self.repeatpid = pid - pclass.lastID  # was computed with pid+pclass.lastID, thus pid=pid_init=pd_orig
                self.repeatpid = pid_orig[self._collection.pu_indicators]
            # self.partitions = self.collection.pu_indicators

        self.kernel = None

        # store particle data as an array per variable (structure of arrays approach)
        # self.particle_data = {}
        # initialised = set()
        # for v in self.ptype.variables:
        #     if v.name in ['xi', 'yi', 'zi', 'ti']:
        #         ngrid = fieldset.gridset.size if fieldset is not None else 1
        #         self.particle_data[v.name] = np.empty((len(lon), ngrid), dtype=v.dtype)
        #     else:
        #         self.particle_data[v.name] = np.empty(len(lon), dtype=v.dtype)

        # if lon is not None and lat is not None:
        #     # Initialise from lists of lon/lat coordinates
        #     assert self.size == len(lon) and self.size == len(lat), (
        #         'Size of ParticleSet does not match length of lon and lat.')

        #     # mimic the variables that get initialised in the constructor
        #     self.particle_data['lat'][:] = lat
        #     self.particle_data['lon'][:] = lon
        #     self.particle_data['depth'][:] = depth
        #     self.particle_data['time'][:] = time
        #     self.particle_data['id'][:] = pid
        #     self.particle_data['fileid'][:] = -1

        #     # special case for exceptions which can only be handled from scipy
        #     self.particle_data['exception'] = np.empty(self.size, dtype=object)

        #     initialised |= {'lat', 'lon', 'depth', 'time', 'id'}

        #     # any fields that were provided on the command line
        #     for kwvar, kwval in kwargs.items():
        #         if not hasattr(pclass, kwvar):
        #             raise RuntimeError('Particle class does not have Variable %s' % kwvar)
        #         self.particle_data[kwvar][:] = kwval
        #         initialised.add(kwvar)

        #     # initialise the rest to their default values
        #     for v in self.ptype.variables:
        #         if v.name in initialised:
        #             continue

        #         if isinstance(v.initial, Field):
        #             for i in range(self.size):
        #                 if np.isnan(time[i]):
        #                     raise RuntimeError('Cannot initialise a Variable with a Field if no time provided. '
        #                                        'Add a "time=" to ParticleSet construction')
        #                 v.initial.fieldset.computeTimeChunk(time[i], 0)
        #                 self.particle_data[v.name][i] = v.initial[
        #                     time[i], depth[i], lat[i], lon[i]
        #                 ]
        #                 logger.warning_once("Particle initialisation from field can be very slow as it is computed in scipy mode.")
        #         elif isinstance(v.initial, attrgetter):
        #             self.particle_data[v.name][:] = v.initial(self)
        #         else:
        #             self.particle_data[v.name][:] = v.initial

        #         initialised.add(v.name)
        # else:
        #     raise ValueError("Latitude and longitude required for generating ParticleSet")

    def _set_particle_vector(self, name, value):
        """Set attributes of all particles to new values.

        :param name: Name of the attribute (str).
        :param value: New value to set the attribute of the particles to.
        """
        self.collection._data[name][:] = value

#         if indices is None:
#         else:
#             self.collection._data[name][indices] = value

#     def _get_particle_vector(self, name, indices=None):
#         """Set attributes of all particles to new values.
# 
#         :param name: Name of the attribute (str).
#         :param indices: (Optional) only set the particles with these indices.
#                         Its length should be equal to the length of 'values'.
#                         If None, all particles are set.
#         :return: The values of the particle attributes.
#         """
#         if indices is None:
#             return self.collection._data[name]
#         else:
#             return self.collection.data[name][indices]

    def _impute_release_times(self, default):
        """Set attribute 'time' to default if encountering NaN values.

        :param default: Default release time.
        :return: Minimum and maximum release times.
        """
        # np.nan_to_num(self._collection._data['time'], nan=default)
        if np.any(np.isnan(self._collection.data['time'])):
            self._collection.data['time'][np.isnan(self._collection.data['time'])] = default
        return np.min(self._collection.data['time']), np.max(self._collection.data['time'])

    def data_indices(self, variable_name, compare_values, not_in=False):
        compare_values = np.array([compare_values, ]) if type(compare_values) not in [list, dict, np.ndarray] else compare_values
        return np.where(np.isin(self._collection.data[variable_name], compare_values, invert=not_in))[0]

    def indices(self, boolean_array_or_statement):
        return np.where(boolean_array_or_statement)[0]

    def indexed_subset(self, indices):
        return ParticleCollectionIteratorSOA(self._collection,
                                             subset=indices)

    @property
    def deleted_particles(self):
        """Get an iterator over all particles that are in an error state.

        :return: Collection iterator over error particles.
        """
        indices = self.data_indices('state', [OperationCode.Delete, ])
        #np.where(np.isin(self._collection.data['state'], [OperationCode.Delete]))[0]
        return ParticleCollectionIteratorSOA(self._collection, subset=indices)

    @property
    def error_particles(self):
        """Get an iterator over all particles that are in an error state.

        :return: Collection iterator over error particles.
        """
        error_indices = self.data_indices('state', [StateCode.Success, StateCode.Evaluate], not_in=True)
        # np.where(np.isin(self._collection.data['state'], [StateCode.Success, StateCode.Evaluate], invert=True))[0]
        return ParticleCollectionIteratorSOA(self._collection, subset=error_indices)

    @property
    def num_error_particles(self):
        return np.sum(np.isin(
            self._collection.data['state'],
            [StateCode.Success, StateCode.Evaluate], invert=True))

    # ==== already user-exposed ==== #
    def __getitem__(self, index):
        # Comment CK: that what we have the iterator or accessor over the collection for -> definitely not a top-level PSet function
        # Comment RB: The collection should provide this function indeed. Until we made a (more) definitive decision on how we want
        #             this to be interfaced, forward this to the collection.
        return self._collection.get_single_by_index(index)

    def cstruct(self):
        """
        'cstruct' returns the ctypes mapping of the combined collections cstruct and the fieldset cstruct.
        This depends on the specific structure in question.
        """
        cstruct = self._collection.cstruct()
        return cstruct

    @property
    def ctypes_struct(self):
        return self.cstruct()

    @classmethod
    def monte_carlo_sample(cls, start_field, size, mode='monte_carlo'):
        """
        Converts a starting field into a monte-carlo sample of lons and lats.

        :param start_field: :mod:`parcels.fieldset.Field` object for initialising particles stochastically (horizontally)  according to the presented density field.

        returns list(lon), list(lat)
        """
        if mode == 'monte_carlo':
            data = start_field.data if isinstance(start_field.data, np.ndarray) else np.array(start_field.data)
            if start_field.interp_method == 'cgrid_tracer':
                p_interior = np.squeeze(data[0, 1:, 1:])
            else:  # if A-grid
                d = data
                p_interior = (d[0, :-1, :-1] + d[0, 1:, :-1] + d[0, :-1, 1:] + d[0, 1:, 1:])/4.
                p_interior = np.where(d[0, :-1, :-1] == 0, 0, p_interior)
                p_interior = np.where(d[0, 1:, :-1] == 0, 0, p_interior)
                p_interior = np.where(d[0, 1:, 1:] == 0, 0, p_interior)
                p_interior = np.where(d[0, :-1, 1:] == 0, 0, p_interior)
            p = np.reshape(p_interior, (1, p_interior.size))
            inds = np.random.choice(p_interior.size, size, replace=True, p=p[0] / np.sum(p))
            xsi = np.random.uniform(size=len(inds))
            eta = np.random.uniform(size=len(inds))
            j, i = np.unravel_index(inds, p_interior.shape)
            grid = start_field.grid
            lon, lat = ([], [])
            if grid.gtype in [GridCode.RectilinearZGrid, GridCode.RectilinearSGrid]:
                lon = grid.lon[i] + xsi * (grid.lon[i + 1] - grid.lon[i])
                lat = grid.lat[j] + eta * (grid.lat[j + 1] - grid.lat[j])
            else:
                lons = np.array([grid.lon[j, i], grid.lon[j, i+1], grid.lon[j+1, i+1], grid.lon[j+1, i]])
                if grid.mesh == 'spherical':
                    lons[1:] = np.where(lons[1:] - lons[0] > 180, lons[1:]-360, lons[1:])
                    lons[1:] = np.where(-lons[1:] + lons[0] > 180, lons[1:]+360, lons[1:])
                lon = (1-xsi)*(1-eta) * lons[0] +\
                    xsi*(1-eta) * lons[1] +\
                    xsi*eta * lons[2] +\
                    (1-xsi)*eta * lons[3]
                lat = (1-xsi)*(1-eta) * grid.lat[j, i] +\
                    xsi*(1-eta) * grid.lat[j, i+1] +\
                    xsi*eta * grid.lat[j+1, i+1] +\
                    (1-xsi)*eta * grid.lat[j+1, i]
            return list(lon), list(lat)
        else:
            raise NotImplementedError('Mode %s not implemented. Please use "monte carlo" algorithm instead.' % mode)

    # ==== already user-exposed ==== #
    @classmethod
    def from_field(cls, fieldset, pclass, start_field, size, mode='monte_carlo', depth=None, time=None, repeatdt=None, lonlatdepth_dtype=None):
        """Initialise the ParticleSet randomly drawn according to distribution from a field

        :param fieldset: :mod:`parcels.fieldset.FieldSet` object from which to sample velocity
        :param pclass: mod:`parcels.particle.JITParticle` or :mod:`parcels.particle.ScipyParticle`
                 object that defines custom particle
        :param start_field: Field for initialising particles stochastically (horizontally)  according to the presented density field.
        :param size: Initial size of particle set
        :param mode: Type of random sampling. Currently only 'monte_carlo' is implemented
        :param depth: Optional list of initial depth values for particles. Default is 0m
        :param time: Optional start time value for particles. Default is fieldset.U.time[0]
        :param repeatdt: Optional interval (in seconds) on which to repeat the release of the ParticleSet
        :param lonlatdepth_dtype: Floating precision for lon, lat, depth particle coordinates.
               It is either np.float32 or np.float64. Default is np.float32 if fieldset.U.interp_method is 'linear'
               and np.float64 if the interpolation method is 'cgrid_velocity'
        """
        lon, lat = cls.monte_carlo_sample(start_field, size, mode)

        return cls(fieldset=fieldset, pclass=pclass, lon=lon, lat=lat, depth=depth, time=time,
                   lonlatdepth_dtype=lonlatdepth_dtype, repeatdt=repeatdt)

    # ==== already user-exposed ==== #
    @classmethod
    def from_particlefile(cls, fieldset, pclass, filename, restart=True, restarttime=None, repeatdt=None, lonlatdepth_dtype=None, **kwargs):
        """Initialise the ParticleSet from a netcdf ParticleFile.
        This creates a new ParticleSet based on locations of all particles written
        in a netcdf ParticleFile at a certain time. Particle IDs are preserved if restart=True

        :param fieldset: :mod:`parcels.fieldset.FieldSet` object from which to sample velocity
        :param pclass: mod:`parcels.particle.JITParticle` or :mod:`parcels.particle.ScipyParticle`
                 object that defines custom particle
        :param filename: Name of the particlefile from which to read initial conditions
        :param restart: Boolean to signal if pset is used for a restart (default is True).
               In that case, Particle IDs are preserved.
        :param restarttime: time at which the Particles will be restarted. Default is the last time written.
               Alternatively, restarttime could be a time value (including np.datetime64) or
               a callable function such as np.nanmin. The last is useful when running with dt < 0.
        :param repeatdt: Optional interval (in seconds) on which to repeat the release of the ParticleSet
        :param lonlatdepth_dtype: Floating precision for lon, lat, depth particle coordinates.
               It is either np.float32 or np.float64. Default is np.float32 if fieldset.U.interp_method is 'linear'
               and np.float64 if the interpolation method is 'cgrid_velocity'
        """

        if repeatdt is not None:
            logger.warning('Note that the `repeatdt` argument is not retained from %s, and that '
                           'setting a new repeatdt will start particles from the _new_ particle '
                           'locations.' % filename)

        pfile = xr.open_dataset(str(filename), decode_cf=True)
        pfile_vars = [v for v in pfile.data_vars]

        vars = {}
        to_write = {}
        for v in pclass.getPType().variables:
            if v.name in pfile_vars:
                vars[v.name] = np.ma.filled(pfile.variables[v.name], np.nan)
            elif v.name not in ['xi', 'yi', 'zi', 'ti', 'dt', '_next_dt', 'depth', 'id', 'fileid', 'state'] \
                    and v.to_write:
                raise RuntimeError('Variable %s is in pclass but not in the particlefile' % v.name)
            to_write[v.name] = v.to_write
        vars['depth'] = np.ma.filled(pfile.variables['z'], np.nan)
        vars['id'] = np.ma.filled(pfile.variables['trajectory'], np.nan)

        if isinstance(vars['time'][0, 0], np.timedelta64):
            vars['time'] = np.array([t/np.timedelta64(1, 's') for t in vars['time']])

        if restarttime is None:
            restarttime = np.nanmax(vars['time'])
        elif callable(restarttime):
            restarttime = restarttime(vars['time'])
        else:
            restarttime = restarttime

        inds = np.where(vars['time'] == restarttime)
        for v in vars:
            if to_write[v] is True:
                vars[v] = vars[v][inds]
            elif to_write[v] == 'once':
                vars[v] = vars[v][inds[0]]
            if v not in ['lon', 'lat', 'depth', 'time', 'id']:
                kwargs[v] = vars[v]

        if restart:
            pclass.setLastID(0)  # reset to zero offset
        else:
            vars['id'] = None

        return cls(fieldset=fieldset, pclass=pclass, lon=vars['lon'], lat=vars['lat'],
                   depth=vars['depth'], time=vars['time'], pid_orig=vars['id'],
                   lonlatdepth_dtype=lonlatdepth_dtype, repeatdt=repeatdt, **kwargs)

    def to_dict(self, pfile, time, deleted_only=False):
        """
        Convert all Particle data from one time step to a python dictionary.
        :param pfile: ParticleFile object requesting the conversion
        :param time: Time at which to write ParticleSet
        :param deleted_only: Flag to write only the deleted Particles
        returns two dictionaries: one for all variables to be written each outputdt,
         and one for all variables to be written once
        """
        data_dict = {}
        data_dict_once = {}

        time = time.total_seconds() if isinstance(time, delta) else time

        pd = self._collection.data

        indices_to_write = []
        if pfile.lasttime_written != time and \
           (pfile.write_ondelete is False or deleted_only is not False):
            if pd['id'].size == 0:
                logger.warning("ParticleSet is empty on writing as array at time %g" % time)
            else:
                if deleted_only is not False:
                    # to_write = deleted_only
                    if type(deleted_only) not in[list, np.ndarray] and deleted_only in [True, 1]:
                        # particles_to_write = self.deleted_particles
                        indices_to_write = np.where(np.isin( self.collection._data['state'],
                                                             [OperationCode.Delete]))[0]
                    elif type(deleted_only) in [list, np.ndarray]:
                        # particles_to_write = self.index_subset(deleted_only)
                        indices_to_write = deleted_only
                else:
                    indices_to_write = _to_write_particles(pd, time)
                if np.any(indices_to_write) > 0:
                    for var in pfile.var_names:
                        data_dict[var] = pd[var][indices_to_write]
                    pfile.maxid_written = np.maximum(pfile.maxid_written, np.max(data_dict['id']))

                pset_errs = ((pd['state'][indices_to_write] != OperationCode.Delete) & np.greater(np.abs(time - pd['time'][indices_to_write]), 1e-3, where=np.isfinite(pd['time'][indices_to_write])))
                #pset_errs = (indices_to_write & (pd['state'] != OperationCode.Delete) & np.greater(np.abs(time - pd['time']), 1e-3, where=np.isfinite(pd['time'])))
                if np.count_nonzero(pset_errs) > 0:
                    logger.warning_once('time argument in pfile.write() is {}, but particles have time {}'.format(time, pd['time'][pset_errs]))

                # ==== this function should probably move back somewhere into the particle-file instead of the to_dict ==== #
                if time not in pfile.time_written:
                    pfile.time_written.append(time)

                if len(pfile.var_names_once) > 0:
                    # first_write = (_to_write_particles(pd, time) & np.isin(pd['id'], pfile.written_once, invert=True))
                    # _to_write_particles(pd, time) &
                    first_write = (_to_write_particles(pd, time) & _is_particle_started_yet(pd, time) & np.isin(pd['id'], pfile.written_once, invert=True))
                    if np.any(first_write):
                        data_dict_once['id'] = np.array(pd['id'][first_write]).astype(dtype=np.int64)
                        for var in pfile.var_names_once:
                            data_dict_once[var] = pd[var][first_write]
                        pfile.written_once.extend(np.array(pd['id'][first_write]).astype(dtype=np.int64).tolist())

            if deleted_only is False:
                pfile.lasttime_written = time

        return data_dict, data_dict_once

    # ==== already user-exposed ==== #
    @property
    def size(self):
        # ==== to change at some point - len and size are different things ==== #
        return len(self._collection)

    # ==== already user-exposed ==== #
    def __repr__(self):
        return "\n".join([str(p) for p in self])

    # ==== already user-exposed ==== #
    def __len__(self):
        return len(self._collection)

    # ==== already user-exposed ==== #
    def __sizeof__(self):
        return sys.getsizeof(self._collection)

    # ==== already user-exposed ==== #
    def __iadd__(self, particles):
        self.add(particles)
        return self

    # ==== already user-exposed ==== #
    def add(self, particles):
        """Method to add particles to the ParticleSet"""
        # Method forward to new implementation
        # Note that this is implemented as an incremental add!
        if isinstance(particles, BaseParticleSet):
            particles = particles.collection
        self._collection += particles
        return self

    # ==== to be removed later ==== #
    def remove_indices(self, indices):
        """Method to remove particles from the ParticleSet, based on their `indices`"""
        # Method forward to new implementation
        if type(indices) in [int, np.int32, np.intp]:
            self._collection.remove_single_by_index(indices)
        else:
            self._collection.remove_multi_by_indices(indices)

    # ==== to be removed later ==== #
    def remove_booleanvector(self, indices):
        """Method to remove particles from the ParticleSet, based on an array of booleans"""
        # Method forward
        self.remove_indices(np.where(indices)[0])

    # ==== already user-exposed ==== #
    def show(self, with_particles=True, show_time=None, field=None, domain=None, projection=None,
             land=True, vmin=None, vmax=None, savefile=None, animation=False, **kwargs):
        """Method to 'show' a Parcels ParticleSet

        :param with_particles: Boolean whether to show particles
        :param show_time: Time at which to show the ParticleSet
        :param field: Field to plot under particles (either None, a Field object, or 'vector')
        :param domain: dictionary (with keys 'N', 'S', 'E', 'W') defining domain to show
        :param projection: type of cartopy projection to use (default PlateCarree)
        :param land: Boolean whether to show land. This is ignored for flat meshes
        :param vmin: minimum colour scale (only in single-plot mode)
        :param vmax: maximum colour scale (only in single-plot mode)
        :param savefile: Name of a file to save the plot to
        :param animation: Boolean whether result is a single plot, or an animation
        """
        from parcels.plotting import plotparticles
        plotparticles(particles=self, with_particles=with_particles, show_time=show_time, field=field, domain=domain,
                      projection=projection, land=land, vmin=vmin, vmax=vmax, savefile=savefile, animation=animation, **kwargs)

    # ==== already user-exposed ==== #
    def density(self, field_name=None, particle_val=None, relative=False, area_scale=False):
        """Method to calculate the density of particles in a ParticleSet from their locations,
        through a 2D histogram.

        :param field: Optional :mod:`parcels.field.Field` object to calculate the histogram
                      on. Default is `fieldset.U`
        :param particle_val: Optional numpy-array of values to weigh each particle with,
                             or string name of particle variable to use weigh particles with.
                             Default is None, resulting in a value of 1 for each particle
        :param relative: Boolean to control whether the density is scaled by the total
                         weight of all particles. Default is False
        :param area_scale: Boolean to control whether the density is scaled by the area
                           (in m^2) of each grid cell. Default is False
        """

        field_name = field_name if field_name else "U"
        field = getattr(self.fieldset, field_name)

        f_str = """
def search_kernel(particle, fieldset, time):
    x = fieldset.{}[time, particle.depth, particle.lat, particle.lon]
        """.format(field_name)

        k = Kernel(
            self.fieldset,
            self._collection.ptype,
            funcname="search_kernel",
            funcvars=["particle", "fieldset", "time", "x"],
            funccode=f_str,
        )
        self.execute(pyfunc=k, runtime=0)

        if isinstance(particle_val, str):
            particle_val = self._collection._data[particle_val]
        else:
            particle_val = particle_val if particle_val else np.ones(self.size)
        density = np.zeros((field.grid.lat.size, field.grid.lon.size), dtype=np.float32)

        for i, p in enumerate(self):
            try:  # breaks if either p.xi, p.yi, p.zi, p.ti do not exist (in scipy) or field not in fieldset
                if p.ti[field.igrid] < 0:  # xi, yi, zi, ti, not initialised
                    raise('error')
                xi = p.xi[field.igrid]
                yi = p.yi[field.igrid]
            except:
                _, _, _, xi, yi, _ = field.search_indices(p.lon, p.lat, p.depth, 0, 0, search2D=True)
            density[yi, xi] += particle_val[i]

        if relative:
            density /= np.sum(particle_val)

        if area_scale:
            density /= field.cell_areas()

        return density

    # ==== already user-exposed ==== #
    def Kernel(self, pyfunc, c_include="", delete_cfiles=True):
        """Wrapper method to convert a `pyfunc` into a :class:`parcels.kernel.Kernel` object
        based on `fieldset` and `ptype` of the ParticleSet

        :param delete_cfiles: Boolean whether to delete the C-files after compilation in JIT mode (default is True)
        """
        return Kernel(self.fieldset, self.collection.ptype, pyfunc=pyfunc, c_include=c_include,
                      delete_cfiles=delete_cfiles)

    # ==== already user-exposed ==== #
    def ParticleFile(self, *args, **kwargs):
        """Wrapper method to initialise a :class:`parcels.particlefile.ParticleFile`
        object from the ParticleSet"""
        return ParticleFile(*args, particleset=self, **kwargs)

    def set_variable_write_status(self, var, write_status):
        """
        Method to set the write status of a Variable
        :param var: Name of the variable (string)
        :param status: Write status of the variable (True, False or 'once')
        """
        # Method forward (for now)
        # Method forward (shall stay) - CK
        self._collection.set_variable_write_status(var, write_status)


ParticleSet = ParticleSetSOA