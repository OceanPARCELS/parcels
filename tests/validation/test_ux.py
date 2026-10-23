"""Validation tests for the analytical unstructured datasets in
``parcels._datasets.unstructured.generated``.

Each test runs a particle through a generated field and asserts the result against the
closed-form trajectory at the strongest accuracy the configuration permits:

* **T1-1** (uniform translation): the field is constant in space and time, so every
  interpolator and every Runge-Kutta stage evaluates the same value. All four
  combinations of face/node placement and EE/RK4 must hit machine precision.
* **T1-2** (2D solid-body rotation): the field is linear in space, so only the
  node-centered grid (barycentric interpolation) reproduces it exactly. Trajectory
  error is then attributable purely to the time integrator. RK4 on the node-centered
  field must return a particle to its starting point after one full orbit.
* **T1-3** (3D helix): horizontal field as T1-2 plus a constant vertical velocity.
  The vertical ODE has constant RHS so depth advection is exact for both face- and
  node-centered datasets under any 3D integrator.
"""

import math

import numpy as np
import pytest

import parcels
from parcels._core.mesh import FlatMesh
from parcels._datasets.unstructured.generated import (
    solid_body_rotation_3d_face_centered,
    solid_body_rotation_3d_node_centered,
    solid_body_rotation_face_centered,
    solid_body_rotation_node_centered,
    uniform_translation_face_centered,
    uniform_translation_node_centered,
)
from parcels.kernels import (
    AdvectionEE,
    AdvectionRK4,
    AdvectionRK4_3D,
)

pytestmark = pytest.mark.validation

# Uniform translation parameters
T1_1_U0 = 0.001
T1_1_V0 = 0.0005
T1_1_RUNTIME = np.timedelta64(3600, "s")
T1_1_DT = np.timedelta64(300, "s")
T1_1_LON0 = 3.0
T1_1_LAT0 = 3.0
T1_1_TOL = 1e-5  # particle positions are float32; ~6 decimal digits of precision

# Solid body rotation - 2D parameters
T1_2_OMEGA = 2.0 * math.pi / 3600.0
T1_2_PERIOD = 2.0 * math.pi / T1_2_OMEGA
T1_2_RUNTIME = np.timedelta64(int(T1_2_PERIOD), "s")
T1_2_DT = np.timedelta64(15, "s")
T1_2_LON0 = 2.0
T1_2_LAT0 = 0.0

# Solid body rotation - 3D parameters
T1_3_W0 = 0.005
T1_3_Z0 = 50.0
T1_3_DT = np.timedelta64(15, "s")
T1_3_RUNTIME = T1_2_RUNTIME


@pytest.mark.parametrize(
    "dataset_fn",
    [uniform_translation_face_centered, uniform_translation_node_centered],
    ids=["face-centered", "node-centered"],
)
@pytest.mark.parametrize("integrator", [AdvectionEE, AdvectionRK4], ids=["EE", "RK4"])
def test_uniform_translation_exact(dataset_fn, integrator):
    ds = dataset_fn(nx=20, u0=T1_1_U0, v0=T1_1_V0)
    fieldset = parcels.FieldSet.from_ugrid_conventions(ds, mesh=FlatMesh())

    pset = parcels.ParticleSet(fieldset, x=[T1_1_LON0], y=[T1_1_LAT0])
    pset.execute(integrator, runtime=T1_1_RUNTIME, dt=T1_1_DT, verbose_progress=False)

    t = T1_1_RUNTIME / np.timedelta64(1, "s")
    np.testing.assert_allclose(pset.x, T1_1_LON0 + T1_1_U0 * t, atol=T1_1_TOL)
    np.testing.assert_allclose(pset.y, T1_1_LAT0 + T1_1_V0 * t, atol=T1_1_TOL)


def test_solid_body_rotation_node_centered_rk4_returns_to_start():
    ds = solid_body_rotation_node_centered(nx=40, omega=T1_2_OMEGA)
    fieldset = parcels.FieldSet.from_ugrid_conventions(ds, mesh=FlatMesh())

    pset = parcels.ParticleSet(fieldset, x=[T1_2_LON0], y=[T1_2_LAT0])
    pset.execute(AdvectionRK4, runtime=T1_2_RUNTIME, dt=T1_2_DT, verbose_progress=False)

    np.testing.assert_allclose(pset.x, T1_2_LON0, atol=1e-6)
    np.testing.assert_allclose(pset.y, T1_2_LAT0, atol=1e-6)


@pytest.mark.parametrize("integrator", [AdvectionEE, AdvectionRK4], ids=["EE", "RK4"])
def test_solid_body_rotation_face_centered_runs_bounded(integrator):
    # Piecewise-constant interpolation has spatial truncation error on a linear field,
    # so we only assert the trajectory stays inside the [-5, 5] mesh.
    ds = solid_body_rotation_face_centered(nx=40, omega=T1_2_OMEGA)
    fieldset = parcels.FieldSet.from_ugrid_conventions(ds, mesh=FlatMesh())

    pset = parcels.ParticleSet(fieldset, x=[T1_2_LON0], y=[T1_2_LAT0])
    pset.execute(integrator, runtime=T1_2_RUNTIME, dt=T1_2_DT, verbose_progress=False)
    assert abs(float(pset.x[0])) < 5.0
    assert abs(float(pset.y[0])) < 5.0


def test_helix_node_centered_rk4_3d_returns_to_start_with_exact_depth():
    ds = solid_body_rotation_3d_node_centered(nx=40, nz=10, omega=T1_2_OMEGA, w0=T1_3_W0)
    fieldset = parcels.FieldSet.from_ugrid_conventions(ds, mesh=FlatMesh())

    pset = parcels.ParticleSet(fieldset, x=[T1_2_LON0], y=[T1_2_LAT0], z=[T1_3_Z0])
    pset.execute(AdvectionRK4_3D, runtime=T1_3_RUNTIME, dt=T1_3_DT, verbose_progress=False)

    t = T1_3_RUNTIME / np.timedelta64(1, "s")
    np.testing.assert_allclose(pset.x, T1_2_LON0, atol=1e-6)
    np.testing.assert_allclose(pset.y, T1_2_LAT0, atol=1e-6)
    np.testing.assert_allclose(pset.z, T1_3_Z0 + T1_3_W0 * t, atol=1e-4)


@pytest.mark.parametrize(
    "dataset_fn",
    [solid_body_rotation_3d_face_centered, solid_body_rotation_3d_node_centered],
    ids=["face-centered", "node-centered"],
)
def test_helix_constant_vertical_velocity_exact_depth(dataset_fn):
    # Constant w implies linear-in-z interpolation and constant-RHS depth ODE are both exact,
    # independent of horizontal grid placement.
    ds = dataset_fn(nx=40, nz=10, omega=T1_2_OMEGA, w0=T1_3_W0)
    fieldset = parcels.FieldSet.from_ugrid_conventions(ds, mesh=FlatMesh())

    pset = parcels.ParticleSet(fieldset, x=[T1_2_LON0], y=[T1_2_LAT0], z=[T1_3_Z0])
    pset.execute(AdvectionRK4_3D, runtime=T1_3_RUNTIME, dt=T1_3_DT, verbose_progress=False)

    t = T1_3_RUNTIME / np.timedelta64(1, "s")
    np.testing.assert_allclose(pset.z, T1_3_Z0 + T1_3_W0 * t, atol=1e-4)
