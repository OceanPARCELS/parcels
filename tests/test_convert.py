import numpy as np
import pytest
import uxarray as ux
import xarray as xr

import parcels
import parcels.convert as convert
import parcels.tutorial
from parcels import FieldSet
from parcels._datasets.remote import open_remote_dataset
from parcels._datasets.structured.circulation_models import datasets as datasets_circulation_models
from parcels.interpolators._xinterpolators import _get_offsets_dictionary


@pytest.mark.parametrize(
    "U, V, coords",
    [
        pytest.param(
            parcels.tutorial.open_dataset("NemoCurvilinear_data_zonal/U"),
            parcels.tutorial.open_dataset("NemoCurvilinear_data_zonal/V"),
            parcels.tutorial.open_dataset("NemoCurvilinear_data_zonal/mesh_mask"),
            id="NemoCurvilinear_data_zonal",
        ),
    ],
)
def test_nemo_to_sgrid_2d(U, V, coords):  # noqa: N803
    ds = convert.nemo_to_sgrid(fields=dict(U=U, V=V), coords=coords)

    assert ds["grid"].attrs == {
        "cf_role": "grid_topology",
        "topology_dimension": 2,
        "node_dimensions": "x y",
        "face_dimensions": "x_center:x (padding:low) y_center:y (padding:low)",
        "node_coordinates": "lon lat",
        "vertical_dimensions": "depth_center:depth (padding:high)",
    }

    meta = ds.sgrid.metadata

    # Assuming that node_dimension1 and node_dimension2 correspond to X and Y respectively
    # check that U and V are properly defined on the staggered grid
    assert {
        meta.get_value_by_id("node_dimension1"),  # X edge
        meta.get_value_by_id("face_dimension2"),  # Y center
    }.issubset(set(ds["U"].dims))
    assert {
        meta.get_value_by_id("face_dimension1"),  # X center
        meta.get_value_by_id("node_dimension2"),  # Y edge
    }.issubset(set(ds["V"].dims))

    parcels.FieldSet.from_sgrid_conventions(ds)


@pytest.mark.parametrize(
    "U, V, depth, coords",
    [
        (
            open_remote_dataset("Benchmarks_MOi_data_metadata-only/U")[["vozocrtx"]].rename_vars({"vozocrtx": "U"}),
            open_remote_dataset("Benchmarks_MOi_data_metadata-only/V")[["vomecrty"]].rename_vars({"vomecrty": "V"}),
            open_remote_dataset("Benchmarks_MOi_data_metadata-only/W")["depthw"],
            open_remote_dataset("Benchmarks_MOi_data_metadata-only/mesh")[["glamf", "gphif"]].isel(t=0),
        ),
    ],
)
def test_nemo_to_sgrid_with_depth(U, V, depth, coords):  # noqa: N803
    coords["depthw"] = depth
    ds = parcels.convert.nemo_to_sgrid(fields=dict(U=U, V=V), coords=coords)

    assert ds["grid"].attrs == {
        "cf_role": "grid_topology",
        "topology_dimension": 2,
        "node_dimensions": "x y",
        "face_dimensions": "x_center:x (padding:low) y_center:y (padding:low)",
        "node_coordinates": "lon lat",
        "vertical_dimensions": "depth_center:depth (padding:high)",
    }

    meta = ds.sgrid.metadata

    # Assuming that node_dimension1 and node_dimension2 correspond to X and Y respectively
    # check that U and V are properly defined on the staggered grid
    assert {
        meta.get_value_by_id("node_dimension1"),  # X edge
        meta.get_value_by_id("face_dimension2"),  # Y center
    }.issubset(set(ds["U"].dims))
    assert {
        meta.get_value_by_id("face_dimension1"),  # X center
        meta.get_value_by_id("node_dimension2"),  # Y edge
    }.issubset(set(ds["V"].dims))

    parcels.FieldSet.from_sgrid_conventions(ds)


def test_convert_nemo_offsets():
    U = parcels.tutorial.open_dataset("NemoCurvilinear_data_zonal/U")
    V = parcels.tutorial.open_dataset("NemoCurvilinear_data_zonal/V")
    coords = parcels.tutorial.open_dataset("NemoCurvilinear_data_zonal/mesh_mask")

    ds = convert.nemo_to_sgrid(fields=dict(U=U, V=V), coords=coords)
    fieldset = FieldSet.from_sgrid_conventions(ds)

    offsets = _get_offsets_dictionary(fieldset.UV.grid)
    assert offsets["X"] == 1
    assert offsets["Y"] == 1
    assert offsets["Z"] == 0


def test_convert_mitgcm_offsets():
    ds_fields = parcels.tutorial.open_dataset("MITgcm_example_data/mitgcm_UV_surface_zonally_reentrant")
    coords = ds_fields[["XG", "YG", "Zl", "time"]]
    ds_fset = convert.mitgcm_to_sgrid(fields={"U": ds_fields.UVEL, "V": ds_fields.VVEL}, coords=coords)
    fieldset = FieldSet.from_sgrid_conventions(ds_fset)
    offsets = _get_offsets_dictionary(fieldset.UV.grid)
    assert offsets["X"] == 0
    assert offsets["Y"] == 0
    assert offsets["Z"] == 0


def test_convert_croco_offsets():
    ds = datasets_circulation_models["ds_CROCO_idealized"]
    coords = ds[["x_rho", "y_rho", "s_w", "time"]]

    ds = convert.croco_to_sgrid(fields={"U": ds["u"], "V": ds["v"]}, coords=coords)
    fieldset = FieldSet.from_sgrid_conventions(ds)

    offsets = _get_offsets_dictionary(fieldset.UV.grid)
    assert offsets["X"] == 0
    assert offsets["Y"] == 0
    assert offsets["Z"] == 0


_COPERNICUS_DATASETS = [
    datasets_circulation_models["ds_copernicusmarine"],
    datasets_circulation_models["ds_copernicusmarine_waves"],
]


@pytest.mark.parametrize("ds", _COPERNICUS_DATASETS)
def test_convert_copernicusmarine(ds, caplog):
    if "uo" in ds:
        fields = {"U": ds["uo"], "V": ds["vo"]}
    elif "VSDX" in ds:
        fields = {"U": ds["VSDX"], "V": ds["VSDY"]}
    else:
        raise ValueError("Test dataset does not contain recognized current variables.")
    ds_fset = convert.copernicusmarine_to_sgrid(fields=fields)
    fieldset = FieldSet.from_sgrid_conventions(ds_fset)
    assert "U" in fieldset.fields
    assert "V" in fieldset.fields
    assert "UV" in fieldset.fields


def test_convert_copernicusmarine_no_currents(caplog):
    ds = datasets_circulation_models["ds_copernicusmarine"]
    ds_fset = convert.copernicusmarine_to_sgrid(fields={"do": ds["uo"]})
    fieldset = FieldSet.from_sgrid_conventions(ds_fset)
    assert "U" not in fieldset.fields
    assert "V" not in fieldset.fields
    assert "UV" not in fieldset.fields
    assert caplog.text == ""


@pytest.mark.filterwarnings("ignore:The delft3d_to_sgrid function is experimental")
def test_convert_structured_delft3d():
    ds = open_remote_dataset("Delft3D_data/Rotterdam_tiny")
    coords = ds[["XZETA", "YZETA", "SIGMA_C"]]
    ds_fset = convert.delft3d_to_sgrid(fields={"U": ds["VELU"], "V": ds["VELV"]}, coords=coords)
    fieldset = FieldSet.from_sgrid_conventions(ds_fset)
    assert "U" in fieldset.fields
    assert "V" in fieldset.fields
    assert "UV" in fieldset.fields


@pytest.mark.parametrize("ds", _COPERNICUS_DATASETS)
def test_convert_copernicusmarine_no_logs(ds, caplog):
    ds = ds.copy()
    zeros = xr.zeros_like(list(ds.data_vars.values())[0])
    ds["U"] = zeros
    ds["V"] = zeros

    ds_fset = convert.copernicusmarine_to_sgrid(fields={"U": ds["U"], "V": ds["V"]})
    fieldset = FieldSet.from_sgrid_conventions(ds_fset)
    assert "U" in fieldset.fields
    assert "V" in fieldset.fields
    assert "UV" in fieldset.fields
    assert caplog.text == ""


def test_convert_copernicusmarine_nodepth(caplog):
    ds = datasets_circulation_models["ds_copernicusmarine"]
    ds = ds.isel(depth=0).drop_vars("depth")
    ds_fset = convert.copernicusmarine_to_sgrid(fields={"uo": ds["uo"]})
    FieldSet.from_sgrid_conventions(ds_fset)
    assert (
        "No depth dimension found in dataset. Assuming surface data and adding a singleton depth dimension."
        in caplog.text
    )


def test_convert_fesom_to_ugrid():
    grid_file = open_remote_dataset("Benchmarks_FESOM2-baroclinic-gyre/grid")
    data_files = open_remote_dataset("Benchmarks_FESOM2-baroclinic-gyre/data")

    grid = ux.open_grid(grid_file)
    uxds = ux.UxDataset(data_files, uxgrid=grid)
    uxds = convert.fesom_to_ugrid(uxds)
    FieldSet.from_ugrid_conventions(uxds)


@pytest.mark.parametrize("converter", ["copernicusmarine_to_sgrid", "nemo_to_sgrid"])
def test_convert_does_not_modify_input_w(converter):
    if converter == "copernicusmarine_to_sgrid":
        ds = datasets_circulation_models["ds_copernicusmarine"].copy(deep=True)
        W = 0.5 * ds["uo"]
        kwargs = dict(fields={"U": ds["uo"], "V": ds["vo"], "W": W})
    else:
        U, V, W, coords = (
            parcels.tutorial.open_dataset(f"NemoNorthSeaORCA025-N006_data/{name}")
            for name in ("U", "V", "W", "mesh_mask")
        )
        W = W["wo"].load()
        kwargs = dict(fields={"U": U["uo"], "V": V["vo"], "W": W}, coords=coords)
    W_before = W.copy(deep=True)

    getattr(convert, converter)(**kwargs)

    xr.testing.assert_identical(W, W_before)


def test_convert_raises_on_existing_grid_metadata():
    ds = datasets_circulation_models["ds_copernicusmarine"]
    coords = ds[["time", "depth", "latitude", "longitude"]]
    coords["some_grid"] = xr.DataArray(0, attrs={"cf_role": "grid_topology"})
    with pytest.raises(ValueError, match="Dataset already has SGRID metadata"):
        convert.copernicusmarine_to_sgrid(fields={"U": ds["uo"], "V": ds["vo"]}, coords=coords)


@pytest.mark.filterwarnings("ignore:The delft3d_to_sgrid function is experimental")
def test_convert_sets_only_conventions_global_attr():
    U = parcels.tutorial.open_dataset("NemoCurvilinear_data_zonal/U")
    V = parcels.tutorial.open_dataset("NemoCurvilinear_data_zonal/V")
    coords = parcels.tutorial.open_dataset("NemoCurvilinear_data_zonal/mesh_mask")
    U.attrs["source"] = "NEMO"
    ds = convert.nemo_to_sgrid(fields=dict(U=U, V=V), coords=coords)
    assert ds.attrs == {"Conventions": "SGRID"}

    ds = open_remote_dataset("Delft3D_data/Rotterdam_tiny")
    ds_fset = convert.delft3d_to_sgrid(fields={"U": ds["VELU"], "V": ds["VELV"]}, coords=ds[["XZETA", "YZETA"]])
    assert ds_fset.attrs == {"Conventions": "SGRID"}


def test_convert_nemo_surface_adds_depth_after_time():
    U = parcels.tutorial.open_dataset("NemoCurvilinear_data_zonal/U")
    V = parcels.tutorial.open_dataset("NemoCurvilinear_data_zonal/V")
    coords = parcels.tutorial.open_dataset("NemoCurvilinear_data_zonal/mesh_mask")
    ds = convert.nemo_to_sgrid(fields=dict(U=U, V=V), coords=coords)
    assert ds["U"].dims[:2] == ("time", "depth")
    assert ds["V"].dims[:2] == ("time", "depth")


def test_convert_dataset_field_without_matching_variable():
    ds = datasets_circulation_models["ds_copernicusmarine"]
    with pytest.raises(ValueError, match=r"fields\['U'\] is a Dataset without a variable named 'U'"):
        convert.copernicusmarine_to_sgrid(fields={"U": ds, "V": ds})


def test_mom6_to_sgrid_cefi_example():
    U = parcels.tutorial.open_dataset("MOM6_example_data/U")["ssu"]
    V = parcels.tutorial.open_dataset("MOM6_example_data/V")["ssv"]
    coords = parcels.tutorial.open_dataset("MOM6_example_data/static")
    ds = convert.mom6_to_sgrid(fields={"U": U, "V": V}, coords=coords)

    # CEFI NEP output is symmetric: one more corner point than cell centres along each axis
    assert ds["grid"].attrs["face_dimensions"] == "xh:xq (padding:none) yh:yq (padding:none)"

    fieldset = FieldSet.from_sgrid_conventions(ds)
    lon = float(ds["lon"].mean())
    lat = float(ds["lat"].mean())
    pset = parcels.ParticleSet(fieldset, x=[lon], y=[lat])
    pset.execute(parcels.kernels.AdvectionRK4, runtime=np.timedelta64(1, "D"), dt=np.timedelta64(10, "m"))
    assert np.isfinite(pset.x).all() and np.isfinite(pset.y).all()


def _mom6_synthetic(nx=4, ny=3, symmetric=True):
    """Small MOM6-like C-grid: U on (yh, xq), V on (yq, xh), corners on (yq, xq)."""
    nxq = nx + 1 if symmetric else nx
    nyq = ny + 1 if symmetric else ny
    lon_c, lat_c = np.meshgrid(np.linspace(-130.0, -125.0, nxq), np.linspace(40.0, 45.0, nyq))
    time = np.array(["2022-01-01", "2022-01-02"], dtype="datetime64[ns]")
    coords = xr.Dataset(
        {
            "geolon_c": (("yq", "xq"), lon_c),
            "geolat_c": (("yq", "xq"), lat_c),
            "areacello": (("yh", "xh"), np.ones((ny, nx))),  # unused static variable, should be dropped
        }
    )
    U = xr.DataArray(np.full((2, ny, nxq), 0.1), dims=("time", "yh", "xq"), coords={"time": time}, name="ssu")
    V = xr.DataArray(np.full((2, nyq, nx), 0.1), dims=("time", "yq", "xh"), coords={"time": time}, name="ssv")
    return U, V, coords


@pytest.mark.parametrize("symmetric, padding", [(True, "none"), (False, "low")])
def test_mom6_to_sgrid(symmetric, padding):
    U, V, coords = _mom6_synthetic(symmetric=symmetric)
    ds = convert.mom6_to_sgrid(fields={"U": U, "V": V}, coords=coords)

    assert ds["grid"].attrs == {
        "cf_role": "grid_topology",
        "topology_dimension": 2,
        "node_dimensions": "xq yq",
        "face_dimensions": f"xh:xq (padding:{padding}) yh:yq (padding:{padding})",
        "node_coordinates": "lon lat",
    }
    assert "areacello" not in ds

    meta = ds.sgrid.metadata
    assert {
        meta.get_value_by_id("node_dimension1"),  # X edge
        meta.get_value_by_id("face_dimension2"),  # Y center
    }.issubset(set(ds["U"].dims))
    assert {
        meta.get_value_by_id("face_dimension1"),  # X center
        meta.get_value_by_id("node_dimension2"),  # Y edge
    }.issubset(set(ds["V"].dims))

    FieldSet.from_sgrid_conventions(ds)


def test_mom6_to_sgrid_inconsistent_sizes():
    U, V, coords = _mom6_synthetic(symmetric=True)
    V = V.isel(xh=slice(0, -2))  # two fewer centres than corners along X
    with pytest.raises(ValueError, match="Unexpected MOM6 grid sizes"):
        convert.mom6_to_sgrid(fields={"U": U, "V": V}, coords=coords)


def test_mom6_to_sgrid_withdepth():
    U, V, coords = _mom6_synthetic(symmetric=True)
    U = U.expand_dims(depth=[0], axis=1)
    V = V.expand_dims(depth=[0], axis=1)
    with pytest.raises(ValueError, match="has more than 2 spatial dimensions"):
        convert.mom6_to_sgrid(fields={"U": U, "V": V}, coords=coords)
