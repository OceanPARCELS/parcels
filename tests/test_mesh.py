import numpy as np
import pytest

from parcels import FieldSet, ParticleSet
from parcels._core.mesh import EARTH_RADIUS, FlatMesh, SphericalMesh
from parcels._datasets.structured.generated import simple_UV_dataset
from parcels.kernels import AdvectionRK4

EARTH_DEG2M = EARTH_RADIUS * np.pi / 180


def test_spherical_mesh_deg2m():
    assert SphericalMesh().radius == EARTH_RADIUS
    assert SphericalMesh().deg2m == EARTH_DEG2M
    r = 3389500.0  # Mars radius
    assert SphericalMesh(radius=r).deg2m == pytest.approx(r * np.pi / 180)


@pytest.mark.parametrize(
    "mesh, exp_radius, exp_deg2m",
    [
        (SphericalMesh(), EARTH_RADIUS, EARTH_DEG2M),
        (SphericalMesh(radius=3389500.0), 3389500.0, 3389500.0 * np.pi / 180),
    ],
)
def test_xgrid_radius_and_deg2m(mesh, exp_radius, exp_deg2m):
    grid = FieldSet.from_sgrid_conventions(simple_UV_dataset(), mesh=mesh).U.grid
    assert isinstance(grid.mesh, SphericalMesh)
    assert grid.mesh.radius == exp_radius
    assert grid.deg2m == pytest.approx(exp_deg2m)


@pytest.mark.parametrize(
    "mesh, deg2m",
    [
        (SphericalMesh(), EARTH_DEG2M),  # No radius entered
        (SphericalMesh(radius=3389500.0), 3389500.0 * np.pi / 180),  # Mars
        (SphericalMesh(radius=6051800.0), 6051800.0 * np.pi / 180),  # Venus
        (SphericalMesh(radius=EARTH_RADIUS), EARTH_DEG2M),  # Explicit Earth
    ],
)
def test_advection_uses_custom_radius(mesh, deg2m, npart=10):
    ds = simple_UV_dataset()
    ds["U"].data[:] = 1.0
    fieldset = FieldSet.from_sgrid_conventions(ds, mesh=mesh)

    runtime = 7200
    startlat = np.linspace(0, 80, npart)
    startlon = 20.0 + np.zeros(npart)
    pset = ParticleSet(fieldset, x=startlon, y=startlat)
    pset.execute(AdvectionRK4, runtime=runtime, dt=np.timedelta64(15, "m"))

    expected_dlon = runtime / (deg2m * np.cos(np.deg2rad(pset.y)))
    np.testing.assert_allclose(pset.x - startlon, expected_dlon, atol=1e-5)
    np.testing.assert_allclose(pset.y, startlat, atol=1e-5)


def test_advection_flat_mesh(npart=10):
    ds = simple_UV_dataset(mesh_str="flat")
    ds["U"].data[:] = 1.0
    fieldset = FieldSet.from_sgrid_conventions(ds)

    runtime = 7200
    startlat = np.linspace(0, 80, npart)
    startlon = 20.0 + np.zeros(npart)
    pset = ParticleSet(fieldset, x=startlon, y=startlat)
    pset.execute(AdvectionRK4, runtime=runtime, dt=np.timedelta64(15, "m"))

    assert fieldset.U.grid.deg2m == 1.0  # flat mesh deg2m
    np.testing.assert_allclose(pset.x - startlon, runtime, atol=1e-5)
    np.testing.assert_allclose(pset.y, startlat, atol=1e-5)


@pytest.mark.parametrize("bad_radius", ["6371000", [6371000], (1, 2), {}])
def test_spherical_mesh_rejects_non_numeric_radius(bad_radius):
    with pytest.raises(TypeError):
        SphericalMesh(radius=bad_radius)


@pytest.mark.parametrize("bad_radius", [0, -1.0, -6371000])
def test_spherical_mesh_rejects_nonpos_radius(bad_radius):
    with pytest.raises(ValueError):
        SphericalMesh(radius=bad_radius)


@pytest.mark.parametrize(
    "lhs, op, rhs",
    [
        (FlatMesh(), "==", FlatMesh()),
        (FlatMesh(), "!=", SphericalMesh()),
        (SphericalMesh(), "==", SphericalMesh()),
        (SphericalMesh(20_000), "!=", SphericalMesh(30_000)),
    ],
)
def test_mesh_equality(lhs, op, rhs):
    match op:
        case "==":
            assert lhs == rhs
        case "!=":
            assert lhs != rhs
        case _:
            raise NotImplementedError
