import pytest

from parcels import UxGrid
from parcels._datasets.unstructured.generic import datasets as uxdatasets

GENERIC_Z_COORDS = ["nz", "zf", "depth_2"]


@pytest.mark.parametrize("uxds", [pytest.param(uxds, id=key) for key, uxds in uxdatasets.items()])
def test_uxgrid_init_on_generic_datasets(uxds):
    vertical_coord = next((z_coord for z_coord in uxds.coords if z_coord in GENERIC_Z_COORDS), None)
    UxGrid(uxds.uxgrid, z=uxds.coords[vertical_coord], mesh="flat")


@pytest.mark.parametrize("uxds", [uxdatasets["stommel_gyre_delaunay"]])
def test_uxgrid_axes(uxds):
    grid = UxGrid(uxds.uxgrid, z=uxds.coords["zf"], mesh="flat")
    assert grid.axes == ["Z", "FACE"]


@pytest.mark.parametrize("uxds", [uxdatasets["stommel_gyre_delaunay"]])
@pytest.mark.parametrize("mesh", ["flat", "spherical"])
def test_uxgrid_mesh(uxds, mesh):
    grid = UxGrid(uxds.uxgrid, z=uxds.coords["zf"], mesh=mesh)

    assert mesh in grid.mesh.__class__.__name__.lower()


@pytest.mark.parametrize("uxds", [uxdatasets["stommel_gyre_delaunay"]])
def test_xgrid_get_axis_dim(uxds):
    grid = UxGrid(uxds.uxgrid, z=uxds.coords["zf"], mesh="flat")

    assert grid.get_axis_dim("FACE") == 721
    assert grid.get_axis_dim("Z") == 2
