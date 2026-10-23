import pytest

from parcels import UxGrid
from parcels._core.mesh import FlatMesh, SphericalMesh
from parcels._datasets.unstructured.generic import datasets as uxdatasets

GENERIC_Z_COORDS = ["nz", "zf", "depth_2"]


@pytest.mark.parametrize("uxds", [pytest.param(uxds, id=key) for key, uxds in uxdatasets.items()])
def test_uxgrid_init_on_generic_datasets(uxds):
    vertical_coord = next((z_coord for z_coord in uxds.coords if z_coord in GENERIC_Z_COORDS), None)
    UxGrid(uxds.uxgrid, z=uxds.coords[vertical_coord])


@pytest.mark.parametrize("uxds", [uxdatasets["stommel_gyre_delaunay"]])
def test_uxgrid_axes(uxds):
    grid = UxGrid(uxds.uxgrid, z=uxds.coords["zf"])
    assert grid.axes == ["Z", "FACE"]


@pytest.mark.parametrize("uxds", [uxdatasets["stommel_gyre_delaunay"]])
@pytest.mark.parametrize("mesh", [FlatMesh(), SphericalMesh()])
def test_uxgrid_mesh(uxds, mesh):
    grid = UxGrid(uxds.uxgrid, z=uxds.coords["zf"], mesh=mesh)
    assert grid.mesh == mesh


@pytest.mark.parametrize("uxds", [uxdatasets["stommel_gyre_delaunay"]])
def test_uxgrid_mesh_inferred_from_units(uxds):
    grid = UxGrid(uxds.uxgrid, z=uxds.coords["zf"])
    assert isinstance(grid.mesh, SphericalMesh)  # uxarray sets degrees_east/north units


@pytest.mark.parametrize("uxds", [uxdatasets["stommel_gyre_delaunay"]])
def test_uxgrid_mesh_string_not_allowed(uxds):
    with pytest.raises(TypeError):
        UxGrid(uxds.uxgrid, z=uxds.coords["zf"], mesh="flat")


@pytest.mark.parametrize("uxds", [uxdatasets["stommel_gyre_delaunay"]])
def test_xgrid_get_axis_dim(uxds):
    grid = UxGrid(uxds.uxgrid, z=uxds.coords["zf"])

    assert grid.get_axis_dim("FACE") == 721
    assert grid.get_axis_dim("Z") == 2
