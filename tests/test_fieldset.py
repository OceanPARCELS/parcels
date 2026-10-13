from datetime import timedelta
from io import StringIO

import cf_xarray  # noqa: F401
import cftime
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from re_assert import Matches

import parcels.tutorial
import tests
from parcels import ParticleFile, ParticleSet, convert
from parcels._core.fieldset import FieldSet, IncompatibleMeshesException, _datetime_to_msg
from parcels._core.mesh import SphericalMesh
from parcels._core.model import _default_vector_field_components
from parcels._datasets.structured.generic import datasets as datasets_structured
from parcels._datasets.structured.generic import datasets_sgrid
from parcels._datasets.unstructured.generic import datasets as datasets_unstructured

ds = datasets_structured["ds_2d_left"]


@pytest.fixture
def fieldset_two_models():
    ds1 = datasets_structured["ds_2d_left"][["U_A_grid", "V_A_grid", "grid"]].rename({"U_A_grid": "U", "V_A_grid": "V"})
    ds2 = datasets_structured["ds_2d_left"][["U_A_grid", "V_A_grid", "grid"]].rename(
        {"U_A_grid": "U_wind", "V_A_grid": "V_wind"}
    )

    fset1 = FieldSet.from_sgrid_conventions(ds1, mesh="flat")
    fset2 = FieldSet.from_sgrid_conventions(ds2, mesh="flat", vector_fields={"UV_wind": ("U_wind", "V_wind")})
    fset2.add_context("my_value", 2.0)
    fset2.add_context("my_list", [1, 2, "hello"])
    fset2.add_constant_field("constant_field", 3.0)
    return fset1 + fset2


def test_fieldset_init_wrong_types():
    with pytest.raises(ValueError, match="Expected `model` to be a ModelData object. Got .*"):
        FieldSet([1.0, 2.0, 3.0])


def test_fieldset_repr(fieldset):
    Matches(r"\<.*FieldSet object at.*\>").assert_matches(repr(fieldset))


def test_fieldset_add_context(fieldset):
    fieldset.add_context("test_context", 1.0)
    assert fieldset.test_context == 1.0


def test_fieldset_add_context_int_name(fieldset):
    with pytest.raises(TypeError, match="Expected a string for variable name, got int instead."):
        fieldset.add_context(123, 1.0)


def test_fieldset_setattr_new(fieldset):
    fieldset.context = {"new_field": 1.0}
    assert fieldset.context == {"new_field": 1.0}


def test_fieldset_setattr_context(fieldset):
    fieldset.add_context("test_context", 1.0)
    with pytest.raises(AttributeError, match=r"Cannot assign .* directly.*context"):
        fieldset.test_context = 2.0


@pytest.mark.parametrize("name", ["a b", "123", "while"])
def test_fieldset_add_context_invalid_name(fieldset, name):
    with pytest.raises(ValueError, match=r"Received invalid Python variable name.*"):
        fieldset.add_context(name, 1.0)


def test_fieldset_add_constant_field(fieldset):
    fieldset.add_constant_field("test_constant_field", 1.0)

    # Get a point in the domain
    time = ds["time"].mean()
    z = ds["depth"].mean()
    lat = ds["lat"].mean()
    lon = ds["lon"].mean()

    assert fieldset.test_constant_field[time, z, lat, lon] == 1.0


def test_fieldset_gridset(fieldset):
    assert fieldset.fields["U"].grid in fieldset.gridset
    assert fieldset.fields["V"].grid in fieldset.gridset
    assert fieldset.fields["UV"].grid in fieldset.gridset
    assert len(fieldset.gridset) == 1

    fieldset.add_constant_field("constant_field", 1.0)
    assert len(fieldset.gridset) == 2


def test_fieldset_no_UV(tmp_parquet):
    fieldset = FieldSet.from_sgrid_conventions(ds[["U_A_grid", "grid"]].rename({"U_A_grid": "P"}), mesh="flat")

    def SampleP(particles, fieldset):
        particles.dx += fieldset.P[particles]

    pset = ParticleSet(fieldset, x=0, y=0)
    ofile = ParticleFile(tmp_parquet, outputdt=np.timedelta64(1, "s"))
    pset.execute(SampleP, runtime=np.timedelta64(1, "s"), dt=np.timedelta64(1, "s"), output_file=ofile)

    df = pd.read_parquet(tmp_parquet)
    assert len(df["x"]) == 2


@pytest.mark.parametrize("ds", [pytest.param(ds, id=k) for k, ds in datasets_structured.items()])
def test_fieldset_from_structured_generic_datasets(ds):
    fieldset = FieldSet.from_sgrid_conventions(ds, mesh="flat")

    assert len(fieldset.fields) == len(ds.data_vars) - 1  # `-1` for the SGRID metadata
    for field in fieldset.fields.values():
        tests.utils.assert_valid_field_data(field.data, field.grid)

    assert len(fieldset.gridset) == 1


@pytest.mark.parametrize(
    "vector_fields,ctx",
    [
        pytest.param(
            {"UV": ("U",)},
            pytest.raises(ValueError, match="must have either 2 or 3 components"),
            id="single-component",
        ),
        pytest.param(
            {"UV": ("U", "missing")},
            pytest.raises(ValueError, match="not present in the source dataset"),
            id="component-not-in-dataset",
        ),
        pytest.param(
            {"UV": ("U", "U", "U", "U")},
            pytest.raises(ValueError, match="must have either 2 or 3 components"),
            id="too-many-components",
        ),
        pytest.param(
            None,
            pytest.raises(ValueError, match="vector_fields must be a dictionary"),
            id="None",
        ),
    ],
)
def test_fieldset_invalid_vector_fields(vector_fields, ctx):
    ds = datasets_structured["ds_2d_left"][["U_A_grid", "V_A_grid", "grid"]].rename({"U_A_grid": "U", "V_A_grid": "V"})

    with ctx:
        FieldSet.from_sgrid_conventions(ds, mesh="flat", vector_fields=vector_fields)


def test_fieldset_structured_vectorfield_default():
    ds = datasets_structured["ds_2d_left"][["U_A_grid", "V_A_grid", "grid"]].rename({"U_A_grid": "U", "V_A_grid": "V"})

    fset = FieldSet.from_sgrid_conventions(ds, mesh="flat")

    assert "U" in fset.fields
    assert "V" in fset.fields
    assert "UV" in fset.fields


def test_fieldset_structured_vectorfield_custom():
    ds = datasets_structured["ds_2d_left"][["U_A_grid", "V_A_grid", "grid"]].rename({"U_A_grid": "U", "V_A_grid": "V"})
    ds = ds.rename({"U": "U_wind", "V": "V_wind"})

    fset = FieldSet.from_sgrid_conventions(ds, mesh="flat", vector_fields={"UV_wind": ("U_wind", "V_wind")})

    assert "U_wind" in fset.fields
    assert "V_wind" in fset.fields
    assert "UV_wind" in fset.fields


def test_fieldset_structured_vectorfield_empty():
    ds = datasets_structured["ds_2d_left"][["U_A_grid", "V_A_grid", "grid"]].rename({"U_A_grid": "U", "V_A_grid": "V"})

    fset = FieldSet.from_sgrid_conventions(ds, mesh="flat", vector_fields={})

    assert "U" in fset.fields
    assert "V" in fset.fields
    assert "UV" not in fset.fields


def test_fieldset_unstructured_vectorfield_default():
    ds = datasets_unstructured["stommel_gyre_delaunay"]
    fset = FieldSet.from_ugrid_conventions(ds, mesh="spherical")

    assert "U" in fset.fields
    assert "V" in fset.fields
    assert "UV" in fset.fields


def test_fieldset_unstructured_vectorfield_custom():
    ds = datasets_unstructured["stommel_gyre_delaunay"]
    ds = ds.rename({"U": "U_wind", "V": "V_wind"})

    fset = FieldSet.from_ugrid_conventions(ds, mesh="spherical", vector_fields={"UV_wind": ("U_wind", "V_wind")})

    assert "U_wind" in fset.fields
    assert "V_wind" in fset.fields
    assert "UV_wind" in fset.fields


def test_fieldset_unstructured_vectorfield_empty():
    ds = datasets_unstructured["stommel_gyre_delaunay"]

    fset = FieldSet.from_ugrid_conventions(ds, mesh="spherical", vector_fields={})

    assert "U" in fset.fields
    assert "V" in fset.fields
    assert "UV" not in fset.fields


@pytest.mark.parametrize(
    "data_vars,expected",
    [
        (["U", "V", "land_mask"], {"UV": ("U", "V")}),
        (["U", "V", "W", "land_mask"], {"UV": ("U", "V"), "UVW": ("U", "V", "W")}),
        (["field1", "field2", "field3"], {}),
    ],
)
def test_default_vector_field_components(data_vars, expected):
    got = _default_vector_field_components(data_vars)
    assert got == expected


def test_multi_model_time_interval():
    ds1 = datasets_structured["ds_2d_left"][["U_A_grid", "V_A_grid", "grid"]]
    fieldset = FieldSet.from_sgrid_conventions(ds1, mesh="flat")

    ds2 = ds1.copy().rename({"U_A_grid": "U2", "V_A_grid": "V2"})
    ds2["time"] = (ds2["time"].dims, ds2["time"].data + np.timedelta64(timedelta(days=1)), ds2["time"].attrs)
    fieldset += FieldSet.from_sgrid_conventions(ds2, mesh="flat")

    ds3 = ds1.copy().rename({"U_A_grid": "U3", "V_A_grid": "V3"})
    ds3["time"] = (ds3["time"].dims, ds3["time"].data + np.timedelta64(timedelta(days=2)), ds3["time"].attrs)
    fieldset += FieldSet.from_sgrid_conventions(ds3, mesh="flat")

    fieldset.add_constant_field("constant_field", 1.0)

    assert len(fieldset.models) == 3
    assert fieldset.constant_model is not None
    assert fieldset.time_interval.left == np.datetime64("2000-01-03")
    assert fieldset.time_interval.right == np.datetime64("2001-01-01")


def test_multi_model_nonoverlapping_time_interval():
    ds1 = datasets_structured["ds_2d_left"][["U_A_grid", "V_A_grid", "grid"]]
    fieldset = FieldSet.from_sgrid_conventions(ds1, mesh="flat")

    ds2 = ds1.copy().rename({"U_A_grid": "U2", "V_A_grid": "V2"})
    ds2["time"] = (ds2["time"].dims, ds2["time"].data + np.timedelta64(timedelta(days=1000)), ds2["time"].attrs)
    fieldset += FieldSet.from_sgrid_conventions(ds2, mesh="flat")

    ds3 = ds1.copy().rename({"U_A_grid": "U3", "V_A_grid": "V3"})
    ds3["time"] = (ds3["time"].dims, ds3["time"].data + np.timedelta64(timedelta(days=2000)), ds3["time"].attrs)
    fieldset += FieldSet.from_sgrid_conventions(ds3, mesh="flat")

    fieldset.add_constant_field("constant_field", 1.0)

    assert len(fieldset.models) == 3
    assert fieldset.constant_model is not None
    assert fieldset.time_interval is None


def test_fieldset_add_incompatible_calendars():
    # tests the adding of fieldsets that have incompatible calendars
    ...


@pytest.mark.parametrize(
    "input_, expected",
    [
        (cftime.DatetimeNoLeap(2000, 1, 1), "<class 'cftime._cftime.DatetimeNoLeap'> with cftime calendar noleap'"),
        (cftime.Datetime360Day(2000, 1, 1), "<class 'cftime._cftime.Datetime360Day'> with cftime calendar 360_day'"),
        (cftime.DatetimeJulian(2000, 1, 1), "<class 'cftime._cftime.DatetimeJulian'> with cftime calendar julian'"),
        (
            cftime.DatetimeGregorian(2000, 1, 1),
            "<class 'cftime._cftime.DatetimeGregorian'> with cftime calendar standard'",
        ),
        (np.datetime64("2000-01-01"), "<class 'numpy.datetime64'>"),
        (cftime.datetime(2000, 1, 1), "<class 'cftime._cftime.datetime'> with cftime calendar standard'"),
    ],
)
def test_datetime_to_msg(input_, expected):
    assert _datetime_to_msg(input_) == expected


def test_fieldset_samegrids_UV():
    """Test that if a simple fieldset with U and V is created, that only one grid object is defined."""
    ...


def test_fieldset_grid_deduplication():
    """Tests that for a full fieldset that the number of grid objects is as expected
    (sharing of grid objects so that the particle location is not duplicated).

    When grid deduplication is actually implemented, this might need to be refactored
    into multiple tests (/more might be needed).
    """
    ...


def test_fieldset_from_icon():
    ds = convert.icon_to_ugrid(datasets_unstructured["icon_square_delaunay_uniform_z_coordinate"])
    fieldset = FieldSet.from_ugrid_conventions(ds)
    assert "U" in fieldset.fields
    assert "V" in fieldset.fields
    assert "UVW" in fieldset.fields


def test_fieldset_from_fesom2():
    ds = convert.fesom_to_ugrid(datasets_unstructured["fesom2_square_delaunay_uniform_z_coordinate"])
    fieldset = FieldSet.from_ugrid_conventions(ds)
    assert "U" in fieldset.fields
    assert "V" in fieldset.fields
    assert "UV" in fieldset.fields
    assert "UVW" in fieldset.fields


def test_fieldset_from_fesom2_missingUV():
    ds = convert.fesom_to_ugrid(datasets_unstructured["fesom2_square_delaunay_uniform_z_coordinate"])
    # Intentionally create a dataset that is missing the U field
    localds = ds.rename({"U": "notU"})
    with pytest.raises(ValueError) as info:
        _ = FieldSet.from_ugrid_conventions(localds)
    assert "Dataset has only one of the two variables 'U' and 'V'" in str(info)

    # Intentionally create a dataset that is missing the V field
    localds = ds.rename({"V": "notV"})
    with pytest.raises(ValueError) as info:
        _ = FieldSet.from_ugrid_conventions(localds)
    assert "Dataset has only one of the two variables 'U' and 'V'" in str(info)


@pytest.mark.parametrize("ds_name", list(datasets_sgrid.keys()))
def test_fieldset_from_sgrid_conventions(ds_name):
    ds = datasets_sgrid[ds_name]
    fieldset = FieldSet.from_sgrid_conventions(ds, mesh="flat")
    assert isinstance(fieldset, FieldSet)
    assert len(fieldset.fields) > 0


def test_fieldset_skip_field_data_validation():
    ds = datasets_structured["ds_2d_left"]
    ds["U_A_grid"][:] = np.nan

    fieldset = FieldSet.from_sgrid_conventions(ds, mesh="flat")
    assert np.isfinite(fieldset.U_A_grid.data).all()
    fieldset = FieldSet.from_sgrid_conventions(ds, mesh="flat", skip_field_data_validation=True)
    assert np.isnan(fieldset.U_A_grid.data).all()


def test_fieldset_add_error_on_duplicate_fields():
    """Test that adding FieldSets with overlapping field names raises a ValueError."""
    ds1 = datasets_structured["ds_2d_left"][["U_A_grid", "V_A_grid", "grid"]].rename({"U_A_grid": "U", "V_A_grid": "V"})
    ds2 = ds1.copy()

    fset1 = FieldSet.from_sgrid_conventions(ds1, mesh="flat")
    fset2 = FieldSet.from_sgrid_conventions(ds2, mesh="flat")

    with pytest.raises(ValueError, match="field names in common.*'U'"):
        fset1 + fset2


def test_fieldset_add():
    """Test that two FieldSets can be combined with + (fset1 + fset2)."""
    ds1 = datasets_structured["ds_2d_left"][["U_A_grid", "V_A_grid", "grid"]].rename({"U_A_grid": "U", "V_A_grid": "V"})
    ds2 = datasets_structured["ds_2d_left"][["U_A_grid", "V_A_grid", "grid"]].rename(
        {"U_A_grid": "U_wind", "V_A_grid": "V_wind"}
    )

    fset1 = FieldSet.from_sgrid_conventions(ds1, mesh="flat")
    fset2 = FieldSet.from_sgrid_conventions(ds2, mesh="flat", vector_fields={"UV_wind": ("U_wind", "V_wind")})

    fset = fset1 + fset2

    assert len(fset.models) == len(fset1.models) + len(fset2.models)

    fields_before = list(fset1.fields.keys()) + list(fset2.fields.keys())
    assert len(fields_before) == len(fset.fields)
    assert set(fields_before) == set(fset.fields.keys())


def test_fieldset_add_different_meshes():
    ds1 = datasets_structured["ds_2d_left"][["U_A_grid", "V_A_grid", "grid"]].rename({"U_A_grid": "U", "V_A_grid": "V"})
    ds2 = datasets_structured["ds_2d_left"][["U_A_grid", "V_A_grid", "grid"]].rename(
        {"U_A_grid": "U_wind", "V_A_grid": "V_wind"}
    )

    fset1 = FieldSet.from_sgrid_conventions(
        ds1,
        mesh=SphericalMesh(71_492_000),  # Jupiter
    )
    fset2 = FieldSet.from_sgrid_conventions(
        ds2,
        mesh="spherical",  # earth
        vector_fields={"UV_wind": ("U_wind", "V_wind")},
    )

    with pytest.raises(IncompatibleMeshesException, match="All ModelData objects must have the same meshes."):
        _ = fset1 + fset2


def test_vectorfields_without_time():
    """Test that vector fields without a time dimension can be evaluated."""
    ds1 = datasets_structured["ds_2d_left"][["U_A_grid", "V_A_grid", "grid"]].rename({"U_A_grid": "U", "V_A_grid": "V"})
    ds2 = ds1.isel(time=0).drop_vars("time").rename({"U": "U_const", "V": "V_const"})
    ds = xr.merge([ds1, ds2])

    fset = FieldSet.from_sgrid_conventions(ds, mesh="flat", vector_fields={"UV_const": ("U_const", "V_const")})
    fset.UV_const.eval(t=0, z=0, y=0, x=0)
    fset.U_const.eval(t=0, z=0, y=0, x=0)


def test_fieldset_add_error_on_duplicate_context_values():
    """Test that adding FieldSets with overlapping context value names raises a ValueError."""
    ds1 = datasets_structured["ds_2d_left"][["U_A_grid", "grid"]].rename({"U_A_grid": "U1"})
    ds2 = datasets_structured["ds_2d_left"][["V_A_grid", "grid"]].rename({"V_A_grid": "V2"})

    fset1 = FieldSet.from_sgrid_conventions(ds1, mesh="flat")
    fset1.add_context("kh", 1.0)

    fset2 = FieldSet.from_sgrid_conventions(ds2, mesh="flat")
    fset2.add_context("kh", 2.0)

    with pytest.raises(ValueError, match="context value names in common.*'kh'"):
        fset1 + fset2


def test_fieldset_add_context_values():
    """Test that context values from both FieldSets are present in the combined FieldSet."""
    ds1 = datasets_structured["ds_2d_left"][["U_A_grid", "grid"]].rename({"U_A_grid": "U1"})
    ds2 = datasets_structured["ds_2d_left"][["V_A_grid", "grid"]].rename({"V_A_grid": "V2"})

    fset1 = FieldSet.from_sgrid_conventions(ds1, mesh="flat")
    fset1.add_context("c1", 1.0)

    fset2 = FieldSet.from_sgrid_conventions(ds2, mesh="flat")
    fset2.add_context("c2", 2.0)

    fset = fset1 + fset2

    assert fset.context["c1"] == 1.0
    assert fset.context["c2"] == 2.0


def test_fieldset_describe(fieldset_two_models: FieldSet):
    fieldset = fieldset_two_models
    io = StringIO()
    expected = """\
| Name           | Type        | Grid number   | Interp method / value   | Parcels backend   |
|:---------------|:------------|:--------------|:------------------------|:------------------|
| my_list        | Context     | -             | [1, 2, 'hello']         | -                 |
| my_value       | Context     | -             | 2.0                     | -                 |
| U              | Field       | 0             | XLinear(...)            | NumPy             |
| V              | Field       | 0             | XLinear(...)            | NumPy             |
| UV             | VectorField | 0             | XLinear_Velocity(...)   | -                 |
| U_wind         | Field       | 1             | XLinear(...)            | NumPy             |
| V_wind         | Field       | 1             | XLinear(...)            | NumPy             |
| UV_wind        | VectorField | 1             | XLinear_Velocity(...)   | -                 |
| constant_field | Field       | 2             | XConstantField(...)     | NumPy             |

mesh: FlatMesh()
time interval: (np.datetime64('2000-01-01T00:00:00.000000000'), np.datetime64('2001-01-01T00:00:00.000000000'))
"""
    fieldset.describe(io)
    actual = io.getvalue()
    assert actual == expected


@tests.mark.zarr_filterwarning_consolidated_metadata
def test_fieldset_describe_backends(tmp_path):
    ds_u = parcels.tutorial.open_dataset("NemoNorthSeaORCA025-N006_data/U")
    ds_v = parcels.tutorial.open_dataset("NemoNorthSeaORCA025-N006_data/V")
    ds_w = parcels.tutorial.open_dataset("NemoNorthSeaORCA025-N006_data/W")
    ds_coords = parcels.tutorial.open_dataset("NemoNorthSeaORCA025-N006_data/mesh_mask")[["glamf", "gphif"]]

    ds_fset = convert.nemo_to_sgrid(
        fields={"U": ds_u["uo"], "V": ds_v["vo"], "W": ds_w["wo"]},
        coords=ds_coords,
    )
    fieldset = FieldSet.from_sgrid_conventions(ds_fset)

    io = StringIO()
    expected = """\
| Name   | Type        |   Grid number | Interp method / value   | Parcels backend   |
|:-------|:------------|--------------:|:------------------------|:------------------|
| U      | Field       |             0 | XLinear(...)            | Dask              |
| V      | Field       |             0 | XLinear(...)            | Dask              |
| W      | Field       |             0 | XLinear(...)            | Dask              |
| UV     | VectorField |             0 | CGrid_Velocity(...)     | -                 |
| UVW    | VectorField |             0 | CGrid_Velocity(...)     | -                 |

mesh: SphericalMesh(radius=6366707.019493707)
time interval: (np.datetime64('2000-01-02T12:00:00.000000000'), np.datetime64('2000-01-12T12:00:00.000000000'))
"""
    fieldset.describe(io)
    actual = io.getvalue()
    assert actual == expected

    # Also run with WindowedArray backend
    fieldset = fieldset.to_windowed_arrays()

    io = StringIO()
    expected = """\
| Name   | Type        |   Grid number | Interp method / value   | Parcels backend   |
|:-------|:------------|--------------:|:------------------------|:------------------|
| U      | Field       |             0 | XLinear(...)            | WindowedArray     |
| V      | Field       |             0 | XLinear(...)            | WindowedArray     |
| W      | Field       |             0 | XLinear(...)            | WindowedArray     |
| UV     | VectorField |             0 | CGrid_Velocity(...)     | -                 |
| UVW    | VectorField |             0 | CGrid_Velocity(...)     | -                 |

mesh: SphericalMesh(radius=6366707.019493707)
time interval: (np.datetime64('2000-01-02T12:00:00.000000000'), np.datetime64('2000-01-12T12:00:00.000000000'))
"""
    fieldset.describe(io)
    actual = io.getvalue()
    assert actual == expected

    # TODO: Add test for the ChunkedArray backend (can also refactor this test at the same time)
