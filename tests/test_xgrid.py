import itertools
from collections import namedtuple

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose
from re_assert import Matches

from parcels import FieldSet
from parcels._core.index_search import (
    LEFT_OUT_OF_BOUNDS,
    RIGHT_OUT_OF_BOUNDS,
    _search_1d_array,
    _search_1d_columns,
)
from parcels._core.utils.time import timedelta_to_float
from parcels._core.xgrid import (
    XGrid,
    _transpose_xfield_data_to_tzyx,
)
from parcels._datasets.structured.generic import X, Y, Z, datasets, datasets_sgrid
from tests import utils

GridTestCase = namedtuple("GridTestCase", ["ds", "attr", "expected"])

test_cases = [
    GridTestCase(datasets["ds_2d_left"], "lon", datasets["ds_2d_left"].XG.values),
    GridTestCase(datasets["ds_2d_left"], "lat", datasets["ds_2d_left"].YG.values),
    GridTestCase(datasets["ds_2d_left"], "depth", datasets["ds_2d_left"].ZG.values),
    GridTestCase(
        datasets["ds_2d_left"],
        "time",
        datasets["ds_2d_left"].time.values.astype(np.float64) / 1e9,
    ),
    GridTestCase(datasets["ds_2d_left"], "xdim", X - 1),
    GridTestCase(datasets["ds_2d_left"], "ydim", Y - 1),
    GridTestCase(datasets["ds_2d_left"], "zdim", Z - 1),
]


def assert_equal(actual, expected):
    if expected is None:
        assert actual is None
    elif isinstance(expected, np.ndarray):
        assert actual.shape == expected.shape
        assert_allclose(actual, expected)
    else:
        assert_allclose(actual, expected)


@pytest.mark.skip(
    "Needs updating after refactoring from https://github.com/Parcels-code/Parcels/pull/2646"
)  # TODO: Remove or replace
@pytest.mark.parametrize("ds", [datasets["ds_2d_left"]])
def test_grid_init_param_types(ds):
    with pytest.raises(ValueError, match="Invalid value 'invalid'. Valid options are.*"):
        XGrid.from_dataset(ds, mesh="invalid")


def test_xgrid_repr(fieldset):
    Matches(r"\<.*XGrid object at.*\>").assert_matches(repr(fieldset.U.grid))


@pytest.mark.parametrize("ds, attr, expected", test_cases)
def test_xgrid_properties_ground_truth(ds, attr, expected):
    grid = FieldSet.from_sgrid_conventions(ds, mesh="flat").data_g.grid
    actual = getattr(grid, attr)
    assert_equal(actual, expected)


def test_xgrid_axes(fieldset):
    assert fieldset.U.grid.axes == ["Z", "Y", "X"]


@pytest.mark.parametrize("ds", [datasets["ds_2d_left"]])
@pytest.mark.parametrize("mesh", ["flat", "spherical"])
def test_uxgrid_mesh(ds, mesh):
    grid = FieldSet.from_sgrid_conventions(ds, mesh=mesh).data_g.grid
    assert mesh in grid._mesh.__class__.__name__.lower()


@pytest.mark.skip(
    "Needs updating after refactoring from https://github.com/Parcels-code/Parcels/pull/2646"
)  # TODO: Remove or replace
@pytest.mark.parametrize("ds", [datasets["ds_2d_left"]])
def test_transpose_xfield_data_to_tzyx(ds):
    da = ds["data_g"]
    grid = XGrid.from_dataset(ds, mesh="flat")

    all_combinations = (itertools.combinations(da.dims, n) for n in range(len(da.dims)))
    all_combinations = itertools.chain(*all_combinations)
    for subset_dims in all_combinations:
        isel = {dim: 0 for dim in subset_dims}
        da_subset = da.isel(isel, drop=True)
        da_test = _transpose_xfield_data_to_tzyx(da_subset, grid.xgcm_grid)
        utils.assert_valid_field_data(da_test, grid)


def test_xgrid_get_axis_dim(fieldset):
    grid = fieldset.U.grid
    assert grid.get_axis_dim("Z") == Z - 1
    assert grid.get_axis_dim("Y") == Y - 1
    assert grid.get_axis_dim("X") == X - 1


def test_invalid_xgrid_field_array():
    """Stress test initialiser by creating incompatible datasets that test the edge cases"""
    ...


@pytest.mark.skip(
    "Needs updating after refactoring from https://github.com/Parcels-code/Parcels/pull/2646"
)  # TODO: Remove or replace
def test_invalid_lon_lat():
    """Stress test the grid initialiser by creating incompatible datasets that test the edge cases"""
    ds = datasets["ds_2d_left"].copy()
    ds["lon"], ds["lat"] = xr.broadcast(ds["YC"], ds["XC"])

    with pytest.raises(
        ValueError,
        match=r".*is defined on the center of the grid, but must be defined on the F points\.",
    ):
        XGrid.from_dataset(ds, mesh="flat")

    ds = datasets["ds_2d_left"].copy()
    ds["lon"], _ = xr.broadcast(ds["YG"], ds["XG"])
    with pytest.raises(
        ValueError,
        match=r".*have different dimensionalities\.",
    ):
        XGrid.from_dataset(ds, mesh="flat")

    ds = datasets["ds_2d_left"].copy()
    ds["lon"], ds["lat"] = xr.broadcast(ds["YG"], ds["XG"])
    ds["lon"], ds["lat"] = ds["lon"].transpose(), ds["lat"].transpose()

    with pytest.raises(
        ValueError,
        match=r".*must be defined on the X and Y axes and transposed to have dimensions in order of Y, X\.",
    ):
        XGrid.from_dataset(ds, mesh="flat")


@pytest.mark.skip(
    "Needs updating after refactoring from https://github.com/Parcels-code/Parcels/pull/2646"
)  # TODO: Remove or replace
def test_invalid_depth():
    ds = datasets["ds_2d_left"].copy()
    ds = ds.reindex({"ZG": ds.ZG[::-1]})

    with pytest.raises(ValueError, match="Depth DataArray .* must be strictly increasing*"):
        XGrid.from_dataset(ds, mesh="flat")


@pytest.mark.skip(
    "Needs updating after refactoring from https://github.com/Parcels-code/Parcels/pull/2646"
)  # TODO: I think we can just rely on the SGRID metadata for this (which already has robust error messaging). How should discrepencies between SGRID and axis attr be handled?
def test_dim_with_duplicate_axis():
    ds = datasets_sgrid["ds_2d_padded_low"].copy()

    # Add an extra Z axis
    ds = ds[["data_g", "grid"]]
    ds["data_g"] = ds["data_g"].expand_dims("vertical_dimensions_dim2", 1)

    z = ds["vertical_dimensions_dim2"]
    z.attrs.update({"axis": "Z"})
    ds["vertical_dimensions_dim2"] = z

    # TODO: Clean up this attribute setting (really, this should be on the source datasets)
    lon = ds["lon"]
    lat = ds["lat"]
    lon.attrs.update({"units": "metres"})
    lat.attrs.update({"units": "metres"})
    ds["lon"] = lon
    ds["lat"] = lat

    with pytest.raises(ValueError, match="Two dimensions .*provide values in the axis direction 'Z'."):
        FieldSet.from_sgrid_conventions(ds)


# TODO restructure: Look into the test below
# remove: eval with timedelta64 time fails; TimeInterval.is_all_time_in_interval expects float (seconds) not timedelta64; time arg type requirement changed
@pytest.mark.skip("remove: see comment above")
@pytest.mark.parametrize("ds", [datasets["ds_2d_left"]])
def test_vertical1D_field(ds):
    ds = ds.drop_vars(set(ds.data_vars) - {"grid"})
    ds["depth"] = (["ZG"], np.linspace(0, 1, ds["depth"].size), {"axis": "Z"})
    ds["z1d"] = xr.DataArray(np.linspace(0, 10, ds["depth"].size), dims=("ZG",))
    ds = ds.reset_coords("z1d")

    fieldset = FieldSet.from_sgrid_conventions(ds, mesh="flat")
    field = fieldset.z1d

    np.testing.assert_almost_equal(field.eval(np.timedelta64(0, "s"), 0.45, 0, 0), np.array([4.5]))


def test_time1D_field():
    ds = datasets["ds_2d_left"].sgrid.isel(XC=0, YC=0, ZC=0)[["data_g", "grid"]]
    ds["data_g"] = (["time"], np.arange(0, ds["time"].size))
    ds["time"] = xr.date_range("2000-01-01", "2000-01-13")

    field = FieldSet.from_sgrid_conventions(ds, mesh="flat").data_g
    time = timedelta_to_float(np.datetime64("2000-01-10T12:00:00") - field.time_interval.left)
    assert field.eval(time, -20, 5, 6) == 9.5


@pytest.mark.parametrize(
    "ds",
    [
        pytest.param(datasets["ds_2d_left"], id="1D lon/lat"),
        pytest.param(datasets["2d_left_rotated"], id="2D lon/lat"),
    ],
)  # for key, ds in datasets.items()])
def test_xgrid_search_cpoints(ds):
    fieldset = FieldSet.from_sgrid_conventions(ds, mesh="flat")
    grid = fieldset.U_A_grid.grid
    lat_array, lon_array = get_2d_fpoint_mesh(grid)
    lat_array, lon_array = corner_to_cell_center_points(lat_array, lon_array)

    for xi in range(grid.xdim - 1):
        for yi in range(grid.ydim - 1):
            axis_indices = {"Z": 0, "Y": yi, "X": xi}

            lat, lon = lat_array[yi, xi], lon_array[yi, xi]
            axis_indices_bcoords = grid.search(0, np.atleast_1d(lat), np.atleast_1d(lon), ei=None)
            axis_indices_test = {k: v["index"] for k, v in axis_indices_bcoords.items() if k in axis_indices}
            assert axis_indices == axis_indices_test

            # assert np.isclose(bcoords[0], 0.5) #? Should this not be the case with the cell center points?
            # assert np.isclose(bcoords[1], 0.5)


def get_2d_fpoint_mesh(grid: XGrid):
    lat, lon = grid.lat, grid.lon
    if lon.ndim == 1:
        lat, lon = np.meshgrid(lat, lon, indexing="ij")
    return lat, lon


def corner_to_cell_center_points(lat, lon):
    """Convert F points to C points."""
    lon_c = (lon[:-1, :-1] + lon[:-1, 1:]) / 2
    lat_c = (lat[:-1, :-1] + lat[1:, :-1]) / 2
    return lat_c, lon_c


def _search_shared_column(array, x):
    """Run ``_search_1d_columns`` with every query point searching the same column ``array``."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    return _search_1d_columns(np.broadcast_to(array, (x.size, array.size)), x)


SEARCH_1D_FUNCTIONS = [
    pytest.param(_search_1d_array, id="search_1d_array"),
    pytest.param(_search_shared_column, id="search_1d_columns"),
]


@pytest.mark.parametrize("search_1d", SEARCH_1D_FUNCTIONS)
@pytest.mark.parametrize(
    "array, x, expected_xi, expected_xsi",
    [
        (np.array([1, 2, 3, 4, 5]), (1.1, 2.1), (0, 1), (0.1, 0.1)),
        (np.array([1, 2, 3, 4, 5]), 2.1, 1, 0.1),
        (np.array([1, 2, 3, 4, 5]), 3.1, 2, 0.1),
        (np.array([1, 2, 3, 4, 5]), 4.5, 3, 0.5),
    ],
)
def test_search_1d_array(search_1d, array, x, expected_xi, expected_xsi):
    xi, xsi = search_1d(array, x)
    np.testing.assert_array_equal(xi, expected_xi)
    np.testing.assert_allclose(xsi, expected_xsi)


@pytest.mark.parametrize("search_1d", SEARCH_1D_FUNCTIONS)
@pytest.mark.parametrize(
    "array, x, expected_xi",
    [
        (np.array([1, 2, 3, 4, 5]), -0.1, LEFT_OUT_OF_BOUNDS),
        (np.array([1, 2, 3, 4, 5]), 6.5, RIGHT_OUT_OF_BOUNDS),
    ],
)
def test_search_1d_array_out_of_bounds(search_1d, array, x, expected_xi):
    xi, _xsi = search_1d(array, x)
    assert xi == expected_xi


@pytest.mark.parametrize("search_1d", SEARCH_1D_FUNCTIONS)
@pytest.mark.parametrize(
    "array, x, expected_xi",
    [
        (np.array([1, 2, 3, 4, 5]), (-0.1, 2.5), (LEFT_OUT_OF_BOUNDS, 1)),
        (np.array([1, 2, 3, 4, 5]), (6.5, 1), (RIGHT_OUT_OF_BOUNDS, 0)),
    ],
)
def test_search_1d_array_some_out_of_bounds(search_1d, array, x, expected_xi):
    xi, _ = search_1d(array, x)
    np.testing.assert_array_equal(xi, expected_xi)


@pytest.mark.xfail(reason="grid.localize not yet adapted to xi, xsi, yi, eta, zi, zeta position keys")
@pytest.mark.parametrize(
    "ds, da_name, expected",
    [
        pytest.param(
            datasets["ds_2d_left"],
            "U_C_grid",
            {
                "XG": (np.int64(0), np.float64(0.0)),
                "YC": (np.int64(-1), np.float64(0.5)),
                "ZG": (np.int64(0), np.float64(0.0)),
            },
            id="MITgcm indexing style U_C_grid",
        ),
        pytest.param(
            datasets["ds_2d_left"],
            "V_C_grid",
            {
                "XC": (np.int64(-1), np.float64(0.5)),
                "YG": (np.int64(0), np.float64(0.0)),
                "ZG": (np.int64(0), np.float64(0.0)),
            },
            id="MITgcm indexing style V_C_grid",
        ),
        pytest.param(
            datasets["ds_2d_right"],
            "U_C_grid",
            {
                "XG": (np.int64(0), np.float64(0.0)),
                "YC": (np.int64(0), np.float64(0.5)),
                "ZG": (np.int64(0), np.float64(0.0)),
            },
            id="NEMO indexing style U_C_grid",
        ),
        pytest.param(
            datasets["ds_2d_right"],
            "V_C_grid",
            {
                "XC": (np.int64(0), np.float64(0.5)),
                "YG": (np.int64(0), np.float64(0.0)),
                "ZG": (np.int64(0), np.float64(0.0)),
            },
            id="NEMO indexing style V_C_grid",
        ),
    ],
)
def test_xgrid_localize_zero_position(ds, da_name, expected):
    """Test localize function using left and right datasets."""
    grid = XGrid.from_dataset(ds, mesh="flat")
    da = ds[da_name]
    position = grid.search(0, 0, 0)

    local_position = grid.localize(position, da.dims)
    assert local_position == expected, f"Expected {expected}, got {local_position}"
