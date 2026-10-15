import numpy as np
import pandas as pd
import pytest

import parcels
from parcels._datasets.unstructured.generated import sigma_coordinate_lattice_dataset
from parcels._datasets.unstructured.generic import datasets as datasets_unstructured
from parcels.kernels import (
    AdvectionEE,
    AdvectionRK2,
    AdvectionRK4,
    AdvectionRK4_3D,
)


@pytest.mark.parametrize("integrator", [AdvectionEE, AdvectionRK2, AdvectionRK4])
def test_ux_constant_flow_face_centered_2D(integrator, tmp_parquet):
    ds = datasets_unstructured["ux_constant_flow_face_centered_2D"]
    T = np.timedelta64(3600, "s")
    dt = np.timedelta64(300, "s")

    fieldset = parcels.FieldSet.from_ugrid_conventions(ds, mesh="flat")
    pset = parcels.ParticleSet(fieldset, x=[5.0], y=[5.0])
    pfile = parcels.ParticleFile(path=tmp_parquet, outputdt=dt)
    pset.execute(integrator, runtime=T, dt=dt, output_file=pfile, verbose_progress=False)
    expected_lon = 8.6
    np.testing.assert_allclose(pset.x, expected_lon, atol=1e-5)

    df = pd.read_parquet(tmp_parquet)
    np.testing.assert_allclose(df["x"].iloc[-1], expected_lon, atol=1e-5)


@pytest.mark.filterwarnings("ignore:Time-varying \\(3D\\) z coordinates are experimental")
def test_ux_advection_on_moving_sigma_grid_keeps_depth():
    """With uniform u and w = 0, particles keep their depth while the sigma levels move past them."""
    nx, nz = 5, 6
    x_nodes, y_nodes = np.meshgrid(np.linspace(0.0, 4e3, nx), np.linspace(0.0, 4e3, nx), indexing="ij")
    bottom_depth = 20.0 + 1e-2 * x_nodes + 2e-3 * y_nodes
    eta = np.stack([np.sin(snapshot + 1e-3 * x_nodes) for snapshot in range(3)])
    ds = sigma_coordinate_lattice_dataset(nx, (0.0, 4e3), (0.0, 4e3), nz, bottom_depth, eta)
    u0 = 0.5
    ds["U"].values[:] = u0
    fieldset = parcels.FieldSet.from_ugrid_conventions(ds, mesh="flat")

    x0, z0 = 200.0, np.array([2.0, 8.0, 15.0])
    pset = parcels.ParticleSet(fieldset, x=np.full(z0.size, x0), y=np.full(z0.size, 2e3), z=z0)
    pset.execute(AdvectionRK4_3D, runtime=np.timedelta64(2, "h"), dt=np.timedelta64(300, "s"), verbose_progress=False)

    np.testing.assert_array_equal(pset.z, z0)
    np.testing.assert_allclose(pset.x, x0 + u0 * 7200.0)
    assert np.all(pset.state < parcels.StatusCode.Error)
