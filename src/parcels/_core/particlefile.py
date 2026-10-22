"""Module controlling the writing of ParticleSets to Parquet file."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr

import parcels
from parcels._core.mesh import BaseMesh
from parcels._core.particle import ParticleClass
from parcels._core.particlesetview import ParticleSetView
from parcels._core.utils.time import timedelta_to_float
from parcels._repr_utils import particlefile_repr
from parcels._typing import PathLike

if TYPE_CHECKING:
    from parcels._core.particle import Variable
    from parcels._core.particleset import ParticleSet
    from parcels._core.utils.time import TimeInterval

__all__ = ["ParticleFile"]


def get_schema(
    particle: parcels.ParticleClass, file_metadata: dict[Any, Any], fset_time_interval: TimeInterval | None
) -> pa.Schema:

    fields = []
    for v in _get_vars_to_write(particle):
        attrs = v.attrs.copy()
        if v.name == "t":
            if fset_time_interval is not None:
                attrs.update(fset_time_interval.get_cf_attrs())
        fields.append(
            pa.field(
                v.name,
                pa.from_numpy_dtype(v.dtype),
                metadata=attrs,
            )
        )
    return pa.schema(fields, metadata=file_metadata.copy())


class ParticleFile:
    """Initialise trajectory output.

    Parameters
    ----------
    path : PathLike
        Path of the output Parquet file.
    outputdt :
        Interval which dictates the update frequency of file output
        while ParticleFile is given as an argument of ParticleSet.execute()
        It is either a numpy.timedelta64, a datimetime.timedelta object or a positive float (in seconds).
    compression : {"zstd", "gzip", "snappy", "brotli", None}, optional
        Compression algorithm to use for the Parquet file. Default is "zstd".
    mode : {None, "w"}, optional
        Writing behaviour.
        - None (default): Write dataset, and raise an error if it already exists.
        - "w": Write dataset, overwriting it.

    Returns
    -------
    ParticleFile
        ParticleFile object that can be used to write particle data to file
    """

    def __init__(
        self,
        path: PathLike,
        outputdt,
        compression: Literal["zstd", "gzip", "snappy", "brotli", None] = "zstd",
        mode: Literal[None, "w"] = None,
    ):
        if not isinstance(outputdt, (np.timedelta64, timedelta, float)):
            raise ValueError(
                f"Expected outputdt to be a np.timedelta64, datetime.timedelta or float (in seconds), got {type(outputdt)}"
            )
        self._compression = compression

        outputdt = timedelta_to_float(outputdt)
        path = Path(path)

        if path.suffix != ".parquet":
            raise ValueError(
                f"ParticleFile data is stored in Parquet files - file extension must be '.parquet'. Got {path.suffix=!r}."
            )

        if outputdt <= 0:
            raise ValueError(f"outputdt must be positive/non-zero. Got {outputdt=!r}")

        self._outputdt = outputdt

        self._path = path  # TODO v4: Consider https://arrow.apache.org/docs/python/getstarted.html#working-with-large-data - though a significant question becomes how to partition, perhaps using a particle variable "partition"?
        self._writer: pq.ParquetWriter | None = None

        if mode not in {None, "w"}:
            raise ValueError(f"Invalid mode value {mode!r}. Expected one of None or 'w'.")

        if path.exists():
            if mode is None:
                msg = f"Path '{path}' already exists. Use mode='w' or use a new path."
                raise ValueError(msg)
            if mode == "w":
                path.unlink()
        if not path.parent.exists():
            msg = f"Folder location for '{path} does not exist. Create the folder location first."
            raise ValueError(msg)

        self.metadata = {}
        self._time_interval: TimeInterval | None = None

    def __repr__(self) -> str:
        return particlefile_repr(self)

    def set_time_interval(self, time_interval: TimeInterval):
        self._time_interval = time_interval

    def set_metadata(self, parcels_grid_mesh: BaseMesh):
        self.metadata.update(
            {
                "feature_type": "trajectory",
                "Conventions": "CF-1.6/CF-1.7",
                "ncei_template_version": "NCEI_NetCDF_Trajectory_Template_v2.0",
                "parcels_version": parcels.__version__,
                "parcels_grid_mesh": repr(parcels_grid_mesh),
            }
        )

    @property
    def outputdt(self):
        return self._outputdt

    @property
    def path(self):
        return self._path

    def write(self, pset: ParticleSet | ParticleSetView, t, indices=None):
        """Write all data from one time step to the parquet file,
        before the particle locations are updated.

        Parameters
        ----------
        pset :
            ParticleSet object to write
        t :
            Time at which to write ParticleSet (same time object as the fieldset)
        """
        pclass = pset._pclass
        particle_data = pset._data

        if self._writer is None:
            assert not self.path.exists(), "If the file exists, the writer should already be set"
            self._writer = pq.ParquetWriter(
                self.path,
                get_schema(pclass, self.metadata, self._time_interval),
                compression=self._compression,
            )

        if isinstance(t, (np.timedelta64, np.datetime64)):
            if self._time_interval is None:
                raise ValueError(
                    "The time interval is unknown, so datetime/timedelta times cannot be converted. "
                    "Pass the ParticleFile to ParticleSet.execute() or call set_time_interval() first."
                )
            t = timedelta_to_float(t - self._time_interval.left)
        vars_to_write = _get_vars_to_write(pclass)
        if indices is not None:
            indices_to_write = indices
        elif isinstance(pset, ParticleSetView):
            # A view already selects the particles to write, so `t` is not used to filter
            indices_to_write = np.flatnonzero(pset._index)
        else:
            indices_to_write = _to_write_particles(particle_data, t)

        self._writer.write_table(
            pa.table({v.name: pa.array(particle_data[v.name][indices_to_write]) for v in vars_to_write}),
        )

    def close(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _get_vars_to_write(particle: ParticleClass) -> list[Variable]:
    return [v for v in particle.variables if v.to_write is not False]


def _to_write_particles(particle_data, t):
    """Return the Particles that need to be written at time: if particle.t is between time-dt/2 and time+dt (/2)"""
    return np.where(
        (
            np.less_equal(
                t - np.abs(particle_data["dt"] / 2),
                particle_data["t"],
                where=np.isfinite(particle_data["t"]),
                out=None,
            )
            & np.greater_equal(
                t + np.abs(particle_data["dt"] / 2),
                particle_data["t"],
                where=np.isfinite(particle_data["t"]),
                out=None,
            )  # check t - dt/2 <= particle_data["t"] <= t + dt/2
            | (
                (np.isnan(particle_data["dt"]))
                & np.equal(t, particle_data["t"], where=np.isfinite(particle_data["t"]), out=None)
            )  # or dt is NaN and t matches particle_data["t"]
        )
        & (np.isfinite(particle_data["particle_id"]))
        & (np.isfinite(particle_data["t"]))
    )[0]


def read_particlefile(path: PathLike, decode_times: bool = True) -> pd.DataFrame:
    """Read a Parcels particlefile (Parquet format) into a pandas DataFrame.

    Parameters
    ----------
    path : PathLike
        Path to the ``.parquet`` particlefile.
    decode_times : bool, optional
        If ``True`` (default), use Xarray to decode the numeric ``t`` column from CF
        conventions into ``datetime`` or ``cftime.datetime`` values using the units stored in
        the column metadata.  If ``False``, the raw numeric values are
        returned unchanged.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the particle data.  When *decode_times* is
        ``True``, the ``t`` column contains datetime-like values;
        otherwise it contains the original numeric representation.

    Notes
    -----
    For larger datasets, consider using `Polars <https://docs.pola.rs/>`_ directly,
    e.g. ``polars.read_parquet(path)``, which offers better performance and lower
    memory usage than pandas for large Parquet files.
    """
    path = Path(path)

    assert path.suffix == ".parquet", "Only Parquet files are supported"

    table = pq.read_table(path)

    try:
        time_field = table.field("t")
    except KeyError as e:
        raise ValueError(f"Could not find 't' column in parquet file. Are you sure {path=!r} is a particlefile?") from e

    assert pa.types.is_floating(time_field.type) or pa.types.is_integer(time_field.type), (
        f"'t' column must be numeric, got {time_field.type}"
    )

    try:
        assert b"units" in time_field.metadata
    except AssertionError as e:
        raise ValueError(f"Could not find 'units' in the 't' column metadata for parquet {path=!r}.") from e

    attrs = {k.decode(): v.decode() for k, v in time_field.metadata.items()}

    df = pl.read_parquet(path)
    if not decode_times:
        return df

    values = table.column("t").to_numpy()
    var = xr.Variable(("t",), values, attrs)
    values = xr.coders.CFDatetimeCoder(time_unit="ns").decode(var).values
    if "since" in attrs["units"]:
        values = values.astype("datetime64[ns]")
        df = df.with_columns(pl.Series("t", values, dtype=pl.Datetime("ns")))
    else:
        values = values.astype("timedelta64[ns]") * 1e9
        df = df.with_columns(pl.Series("t", values, dtype=pl.Duration("ns")))

    return df
