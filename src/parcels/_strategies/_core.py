import io

import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from parcels._core.particle import ParticleClass
from parcels._core.particlefile import get_schema

from .particle import particle_class
from .time import time_interval as st_time_interval

__all__ = [
    "particlefile_output",
]


def _generate_dummy_data(particle: ParticleClass, nparticles=10, nobs=10) -> pd.DataFrame:
    """Build a pandas dataframe from a particleclass.

    Only variables with ``to_write=True`` are included.
    """
    columns: dict[str, np.ndarray] = {}
    variables = {var.name: var for var in particle.variables if var.to_write}
    try:
        particle_id = variables["particle_id"]
        t = variables["t"]
    except KeyError as e:
        e.add_note("This function requires 'particle_id' and 't' to be set")

    nobs_total = nparticles * nobs
    columns = {}
    columns["particle_id"] = np.repeat(
        np.arange(0, nparticles, dtype=particle_id.dtype).reshape((-1, 1)),
        nobs,
        axis=1,
    ).flatten()
    columns["t"] = np.repeat(
        np.linspace(0, nparticles * 3, num=nparticles, dtype=t.dtype).reshape((-1, 1)),
        nobs,
        axis=1,
    ).flatten()

    data_vars = set(variables.keys()) - {"particle_id", "t"}

    for name in data_vars:
        var = variables[name]
        columns[name] = np.linspace(0, 10000, num=nobs_total, dtype=var.dtype)

    return pd.DataFrame(columns)


@st.composite
def particlefile_output(draw, nobs=None, nparticles=None) -> io.BytesIO:
    particle = draw(particle_class())
    time_interval = draw(st_time_interval())
    if nobs is None:
        nobs = draw(st.integers(min_value=5, max_value=100))
    if nparticles is None:
        nparticles = draw(st.integers(min_value=5, max_value=100))

    df = _generate_dummy_data(particle, nparticles, nobs)
    schema = get_schema(particle, {}, time_interval)
    buf = io.BytesIO()
    pq.write_table(
        pa.table(df, schema=schema),
        buf,
    )
    buf.seek(0)
    return buf
