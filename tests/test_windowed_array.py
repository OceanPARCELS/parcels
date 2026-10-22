"""Tests for the transparent rolling time-window cache (WindowedArray)."""

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from parcels import FieldSet, ParticleSet
from parcels._core._windowed_array import WindowedArray, maybe_windowed
from parcels._core.mesh import FlatMesh
from parcels._datasets.structured.generated import simple_UV_dataset
from parcels._datasets.unstructured.generic import _ux_constant_flow_face_centered_2D
from parcels.kernels import AdvectionRK2


def test_windowed_isel_matches_dask_loads_once_and_evicts():
    """WindowedArray.isel must equal dask isel, load each level once, keep <=2 resident."""
    ntime, n, npart = 20, 64, 200
    rng = np.random.default_rng(0)
    base = rng.standard_normal((ntime, 3, n, n))
    lazy = xr.DataArray(da.from_array(base, chunks=(1, 3, n, n)), dims=("time", "depth", "lat", "lon"))
    win = WindowedArray(lazy)

    worst, max_cache = 0.0, 0
    for step in range(40):
        ti = min(step // 2, ntime - 2)  # advancing clock, 2 sub-steps per level
        yi, xi = rng.integers(0, n, npart), rng.integers(0, n, npart)
        zi = np.zeros(npart, dtype=int)
        sel = dict(
            time=xr.DataArray(np.r_[np.full(npart, ti), np.full(npart, ti + 1)], dims="p"),
            depth=xr.DataArray(np.r_[zi, zi], dims="p"),
            lat=xr.DataArray(np.r_[yi, yi], dims="p"),
            lon=xr.DataArray(np.r_[xi, xi], dims="p"),
        )
        got = win.isel(sel).data
        ref = lazy.isel(sel).data.compute()
        worst = max(worst, float(np.abs(got - ref).max()))
        max_cache = max(max_cache, win._cache.sizes["time"])

    assert worst == 0.0  # byte-identical to dask
    assert win.loads == ntime  # each time level read exactly once
    assert max_cache <= 2  # only the bracketing levels resident


def test_windowed_isel_backward_clock_loads_once_and_evicts():
    """A backward-running clock (dt < 0) must also load each level once and keep <=2
    resident: eviction is symmetric, so no integration-direction flag is needed.
    """
    ntime, n, npart = 20, 64, 200
    rng = np.random.default_rng(0)
    base = rng.standard_normal((ntime, 3, n, n))
    lazy = xr.DataArray(da.from_array(base, chunks=(1, 3, n, n)), dims=("time", "depth", "lat", "lon"))
    win = WindowedArray(lazy)

    worst, max_cache = 0.0, 0
    for step in range(40):
        ti = max(ntime - 2 - step // 2, 0)  # receding clock, 2 sub-steps per level
        yi, xi = rng.integers(0, n, npart), rng.integers(0, n, npart)
        zi = np.zeros(npart, dtype=int)
        sel = dict(
            time=xr.DataArray(np.r_[np.full(npart, ti), np.full(npart, ti + 1)], dims="p"),
            depth=xr.DataArray(np.r_[zi, zi], dims="p"),
            lat=xr.DataArray(np.r_[yi, yi], dims="p"),
            lon=xr.DataArray(np.r_[xi, xi], dims="p"),
        )
        got = win.isel(sel).data
        ref = lazy.isel(sel).data.compute()
        worst = max(worst, float(np.abs(got - ref).max()))
        max_cache = max(max_cache, win._cache.sizes["time"])

    assert worst == 0.0  # byte-identical to dask
    assert win.loads == ntime  # each time level read exactly once, going backward
    assert max_cache <= 2  # only the bracketing levels resident


@pytest.mark.parametrize(
    "fset_convention, ds, mesh",
    [
        (FieldSet.from_sgrid_conventions, simple_UV_dataset(mesh="flat"), FlatMesh()),
        (FieldSet.from_ugrid_conventions, _ux_constant_flow_face_centered_2D(), "flat"),
    ],
    ids=["structured", "unstructured"],
)
def test_windowed_arrays_wraps_dask_but_not_numpy(fset_convention: callable, ds: xr.Dataset, mesh):
    fset_np = fset_convention(ds, mesh=mesh)
    fset_dk = fset_convention(ds.chunk({"time": 1}), mesh=mesh)

    # construction is never windowing -- it is opt-in via the fieldset method
    assert not isinstance(fset_np.U.data, WindowedArray)
    assert not isinstance(fset_dk.U.data, WindowedArray)
    assert isinstance(fset_dk.U.data.data, da.Array)  # chunked input stays lazy (dask-backed)

    assert fset_np.to_windowed_arrays() is fset_np  # chainable
    fset_dk.to_windowed_arrays()

    # numpy-backed field is left eager; dask-backed field gets wrapped
    assert not isinstance(fset_np.U.data, WindowedArray)
    assert isinstance(fset_dk.U.data, WindowedArray)
    # transparency: forwarded attributes still behave like the DataArray
    assert fset_dk.U.data.dims == fset_np.U.data.dims
    assert fset_dk.U.data.shape == fset_np.U.data.shape


@pytest.mark.parametrize(
    "fset_convention, ds, mesh",
    [
        (FieldSet.from_sgrid_conventions, simple_UV_dataset(mesh="flat"), FlatMesh()),
        (FieldSet.from_ugrid_conventions, _ux_constant_flow_face_centered_2D(), "flat"),
    ],
    ids=["structured", "unstructured"],
)
def test_to_windowed_arrays_is_idempotent_and_forwards_max_levels(fset_convention: callable, ds: xr.Dataset, mesh):
    fs = fset_convention(ds.chunk({"time": 1}), mesh=mesh)

    fs.to_windowed_arrays(max_levels=3)
    first = fs.U.data
    assert isinstance(first, WindowedArray)
    assert first._max == 3

    # re-wrapping returns the same object (idempotent, warm cache preserved)
    fs.to_windowed_arrays(max_levels=3)
    assert fs.U.data is first


def test_windowed_isel_empty_selection():
    """An empty pointwise selection (a kernel evaluating an empty particle subset,
    as in the Argo floats tutorial's phase kernels) must return an empty result
    without touching the cache, matching plain xarray/dask isel behaviour.
    """
    ntime, n = 4, 8
    base = np.arange(ntime * 3 * n * n, dtype=float).reshape(ntime, 3, n, n)
    lazy = xr.DataArray(da.from_array(base, chunks=(1, 3, n, n)), dims=("time", "depth", "lat", "lon"))
    win = WindowedArray(lazy)

    empty = xr.DataArray(np.array([], dtype=int), dims="p")
    sel = dict(time=empty, depth=empty, lat=empty, lon=empty)
    got = win.isel(sel)
    ref = lazy.isel(sel)

    assert got.shape == ref.shape == (0,)
    assert got.dtype == base.dtype
    assert win.loads == 0  # nothing read
    assert win._cache.sizes["time"] == 0  # nothing cached, nothing evicted

    # a warm cache must survive an interleaved empty call (no spurious eviction)
    full = xr.DataArray(np.zeros(5, dtype=int), dims="p")
    win.isel(dict(time=full, depth=full, lat=full, lon=full))
    assert list(win._cache["time"].values) == [0]
    win.isel(sel)
    assert list(win._cache["time"].values) == [0]


def test_maybe_windowed_passthrough_for_non_time_leading():
    da_no_time = xr.DataArray(da.zeros((3, 4), chunks=(3, 4)), dims=("lat", "lon"))
    assert maybe_windowed(da_no_time) is da_no_time  # not wrapped (no leading time dim)


@pytest.mark.parametrize("mesh", ["flat", "spherical"])
@pytest.mark.parametrize("dt_minutes", [15, -15], ids=["forward", "backward"])
def test_dask_advection_matches_numpy_on_structured_grids(mesh, dt_minutes):
    """An identical advection must give identical trajectories whether the field
    is numpy-backed or dask-backed (windowed) -- for both forward (dt > 0) and
    backward (dt < 0) integration on structured grids.
    """
    ds = simple_UV_dataset(mesh=mesh)
    ds["U"].data[:] = 1.0  # steady zonal flow -> in-bounds, deterministic

    def run(chunked):
        d = ds.chunk({"time": 1}) if chunked else ds
        fs = FieldSet.from_sgrid_conventions(d)
        if chunked:
            fs.to_windowed_arrays()
        pset = ParticleSet(fs, x=np.zeros(10), y=np.linspace(-10, 10, 10))
        pset.execute(AdvectionRK2, runtime=7200, dt=np.timedelta64(dt_minutes, "m"))
        return np.array(pset.x), np.array(pset.y)

    x_np, y_np = run(False)
    x_dk, y_dk = run(True)
    np.testing.assert_allclose(x_dk, x_np, atol=1e-9)
    np.testing.assert_allclose(y_dk, y_np, atol=1e-9)


@pytest.mark.parametrize("mesh", ["flat", "spherical"])
@pytest.mark.parametrize("dt_minutes", [15, -15], ids=["forward", "backward"])
def test_dask_advection_matches_numpy_on_unstructured_grids(mesh, dt_minutes):
    """An identical advection must give identical trajectories whether the field
    is numpy-backed or dask-backed (windowed) -- for both forward (dt > 0) and
    backward (dt < 0) integration on unstructured grids.
    """
    ds = _ux_constant_flow_face_centered_2D()

    def run(chunked):
        d = ds.chunk({"time": 1}) if chunked else ds
        fs = FieldSet.from_ugrid_conventions(d, mesh=mesh)
        if chunked:
            fs.to_windowed_arrays()
        pset = ParticleSet(fs, x=10 * np.ones(10), y=np.linspace(5, 15, 10))
        pset.execute(AdvectionRK2, runtime=3600, dt=np.timedelta64(dt_minutes, "m"))
        return np.array(pset.x), np.array(pset.y)

    x_np, y_np = run(False)
    x_dk, y_dk = run(True)
    np.testing.assert_allclose(x_dk, x_np, atol=1e-9)
    np.testing.assert_allclose(y_dk, y_np, atol=1e-9)
