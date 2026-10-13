import io
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import parcels
import parcels.tutorial
from parcels._datasets.structured.generated import simple_UV_dataset
from parcels.kernels import AdvectionRK2, AdvectionRK4

BackendT = Literal["WindowedArray", "Dask", "Zarr", "NumPy", "ChunkCachedArray"]
BACKENDS: set[BackendT] = {"WindowedArray", "Dask", "Zarr", "NumPy", "ChunkCachedArray"}


@pytest.fixture(scope="module")
def nemo_dataset() -> xr.Dataset:
    ds_u = parcels.tutorial.open_dataset("NemoNorthSeaORCA025-N006_data/U")
    ds_v = parcels.tutorial.open_dataset("NemoNorthSeaORCA025-N006_data/V")
    ds_w = parcels.tutorial.open_dataset("NemoNorthSeaORCA025-N006_data/W")
    ds_coords = parcels.tutorial.open_dataset("NemoNorthSeaORCA025-N006_data/mesh_mask")[["glamf", "gphif"]]

    ds_fset = parcels.convert.nemo_to_sgrid(
        fields={"U": ds_u["uo"], "V": ds_v["vo"], "W": ds_w["wo"]},
        coords=ds_coords,
    )
    return ds_fset


@pytest.fixture(scope="module")
def nemo_results(tmp_path_factory, nemo_dataset) -> tuple[xr.Dataset, Path]:
    ref_parquet = tmp_path_factory.mktemp("nemo_ref") / "ref.parquet"
    run_simulation(
        nemo_dataset.copy(), ref_parquet, "NumPy"
    )  # pass in a copy to avoid in-place modifications to the dataset
    return nemo_dataset, ref_parquet


def fieldset_uses_backend(fset: parcels.FieldSet, backend: BackendT):
    # a bit of a hacky way to check for the backend.... probably better for us to change how backends are stored
    buf = io.StringIO()
    fset.describe(buf)
    return backend in buf.getvalue()


def run_simulation(ds: xr.Dataset, output_path: Path, backend: BackendT) -> Path:
    if backend == "Zarr":
        raise NotImplementedError("Doesn't work at this level of execution. Also will likely remove Zarr backend.")

    if backend == "NumPy":
        ds.load()

    fset = parcels.FieldSet.from_sgrid_conventions(ds)

    if backend == "WindowedArray":
        fset.to_windowed_arrays()
    if backend == "ChunkCachedArray":
        fset.to_chunk_cached_arrays()

    assert fieldset_uses_backend(fset, backend)

    npart = 1000
    lons = np.linspace(1.9, 3.4, npart)
    lats = np.linspace(51.6, 52.5, npart)
    z = np.ones(npart)
    pset = parcels.ParticleSet(fset, x=lons, y=lats, z=z)

    def delete_particle(particles, fieldset):
        error_states = (
            parcels.StatusCode.ErrorOutOfBounds,
            parcels.StatusCode.ErrorGridSearching,
        )
        for error in error_states:
            particles.state = np.where(
                particles.state == error,
                parcels.StatusCode.Delete,
                particles.state,
            )

    pfile = parcels.ParticleFile(output_path, outputdt=np.timedelta64(6, "h"))
    pset.execute(
        [AdvectionRK4, delete_particle],
        runtime=np.timedelta64(1, "D"),
        dt=np.timedelta64(1, "h"),
        output_file=pfile,
    )

    return output_path


@pytest.mark.parametrize(
    "backend",
    BACKENDS
    - {
        "NumPy",  # reference point
        "Zarr",  # not supported
    },
)
def test_nemo_identical_across_backends(nemo_results, tmp_parquet, backend):
    ds = nemo_results[0]
    ref_parquet = nemo_results[1]

    assert str(ref_parquet) != str(tmp_parquet)  # just covering my bases with Pytest fixture usage

    run_simulation(ds, tmp_parquet, backend)

    ref_df = pd.read_parquet(ref_parquet)
    test_df = pd.read_parquet(tmp_parquet)

    ref_df = ref_df.sort_values(["particle_id", "t"]).reset_index(drop=True)
    test_df = test_df.sort_values(["particle_id", "t"]).reset_index(drop=True)

    np.testing.assert_allclose(test_df["x"].values, ref_df["x"].values, atol=1e-5)
    np.testing.assert_allclose(test_df["y"].values, ref_df["y"].values, atol=1e-5)
    np.testing.assert_allclose(test_df["z"].values, ref_df["z"].values, atol=1e-5)


@pytest.mark.parametrize(
    "backend",
    BACKENDS
    - {
        "NumPy",  # reference point
        "Zarr",  # not supported
    },
)
def test_field_without_depth_identical_across_backends(backend: BackendT):
    """Fields without a depth dimension get a size-1 ``mockZ`` dimension, which must
    also be indexable by every backend (gh-2897).
    """
    ds = simple_UV_dataset(n_times=10, nz=1, ny=20, nx=20, mesh="flat").isel(depth=0, drop=True)
    ds["U"].data[:] = 1.0 + ds["XG"].values / 20  # spatially varying, so the corners matter
    ds["V"].data[:] = 0.5

    def run(backend: BackendT) -> tuple[np.ndarray, np.ndarray]:
        fset = parcels.FieldSet.from_sgrid_conventions(ds if backend == "NumPy" else ds.chunk(), mesh="flat")
        assert "mockZ" in fset.U.data.dims
        if backend == "WindowedArray":
            fset.to_windowed_arrays()
        if backend == "ChunkCachedArray":
            fset.to_chunk_cached_arrays()
        assert fieldset_uses_backend(fset, backend)

        pset = parcels.ParticleSet(fset, x=np.linspace(2, 6, 10), y=np.linspace(2, 6, 10))
        pset.execute(AdvectionRK2, runtime=np.timedelta64(2, "s"), dt=np.timedelta64(1, "s"))
        return pset.x, pset.y

    ref_x, ref_y = run("NumPy")
    x, y = run(backend)
    np.testing.assert_allclose(x, ref_x)
    np.testing.assert_allclose(y, ref_y)


@pytest.mark.parametrize(
    "backend",
    BACKENDS
    - {
        "NumPy",  # reference point
        "Zarr",  # not supported with this input data
    },
)
def test_fieldset_describe_backend(nemo_dataset, backend: BackendT):
    if backend == "NumPy":
        nemo_dataset.load()

    fieldset = parcels.FieldSet.from_sgrid_conventions(nemo_dataset)

    if backend == "WindowedArray":
        fieldset.to_windowed_arrays()
    if backend == "ChunkCachedArray":
        fieldset.to_chunk_cached_arrays()

    assert fieldset_uses_backend(fieldset, backend)

    for other_backend in BACKENDS - {backend}:
        assert not fieldset_uses_backend(fieldset, other_backend)
