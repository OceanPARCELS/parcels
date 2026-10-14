import io
import tempfile
from datetime import timedelta
from pathlib import Path

import numpy as np
import xarray as xr
from hypothesis import example, given, settings

import parcels._strategies as pst
from parcels import FieldSet, ParticleFile, ParticleSet, StatusCode
from parcels._compat_v3 import particlefile_to_v3_zarr
from parcels._core.particle import Particle
from parcels._datasets.structured.generic import datasets as datasets_structured


def example_particlefile() -> io.BytesIO:
    ds = datasets_structured["ds_2d_left"].copy()
    ds = ds[["U_A_grid", "V_A_grid", "grid"]].rename({"U_A_grid": "U", "V_A_grid": "V"})
    fieldset = FieldSet.from_sgrid_conventions(ds, mesh="flat")

    npart = 10
    pset = ParticleSet(fieldset, pclass=Particle, x=np.zeros(npart), y=np.zeros(npart))

    def RandomDelete(particles, fieldset):  # pragma: no cover
        particles.state = np.where(
            np.random.rand(len(particles)) < 0.3,
            StatusCode.Delete,
            particles.state,
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = Path(tmpdir) / "output.parquet"
        ofile = ParticleFile(parquet_path, outputdt=np.timedelta64(1, "s"))
        pset.execute(RandomDelete, runtime=np.timedelta64(5, "s"), dt=np.timedelta64(1, "s"), output_file=ofile)

        buf = io.BytesIO(parquet_path.read_bytes())

    return buf


def assert_valid_v3_particlefile_structure(ds: xr.Dataset):
    for var in ["lat", "lon", "z", "time"]:
        assert var in ds.variables

    assert set(ds.dims) == {"obs", "trajectory"}
    assert set(ds.coords) == {"obs", "trajectory"}

    assert ds["lat"].attrs["axis"] == "Y"  # attrs are copied accross correctly


@settings(deadline=timedelta(seconds=3))
@example(buf=example_particlefile())
@given(buf=pst.particlefile_output())
def test_particlefile_to_v3_zarr(buf):
    # The @example buffer is shared between runs of the test (e.g., Hypothesis may re-run it),
    # and reading it advances its position. Pass a fresh copy so that each run reads the whole file.
    buf = io.BytesIO(buf.getvalue())
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_zarr = Path(tmpdir) / "output.zarr"

        particlefile_to_v3_zarr(from_parquet=buf, to_zarr=tmp_zarr)
        ds = xr.open_zarr(tmp_zarr)
        assert_valid_v3_particlefile_structure(ds)
