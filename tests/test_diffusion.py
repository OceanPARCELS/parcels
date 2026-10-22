import random

import numpy as np
import pytest
from scipy import stats

from parcels import (
    FieldSet,
    Particle,
    ParticleSet,
    Variable,
)
from parcels._core.utils.time import timedelta_to_float
from parcels._datasets.structured.generated import simple_UV_dataset
from parcels.kernels import AdvectionDiffusionEM, AdvectionDiffusionM1, DiffusionUniformKh
from tests.utils import create_fieldset_zeros_conversion


@pytest.mark.parametrize("mesh", ["spherical", "flat"])
def test_fieldKh_Brownian(mesh):
    kh_zonal = 100
    kh_meridional = 50
    mesh_conversion = 1 / 1852.0 / 60 if mesh == "spherical" else 1

    ds = simple_UV_dataset(n_times=2, nz=1, ny=2, nx=2, mesh=mesh)
    ds["lon"].data = np.array([-1e6, 1e6])
    ds["lat"].data = np.array([-1e6, 1e6])
    fieldset = FieldSet.from_sgrid_conventions(ds)
    fieldset.add_constant_field("Kh_zonal", kh_zonal)
    fieldset.add_constant_field("Kh_meridional", kh_meridional)

    npart = 100
    runtime = np.timedelta64(2, "h")

    np.random.seed(1234)
    pset = ParticleSet(fieldset=fieldset, x=np.zeros(npart), y=np.zeros(npart))
    pset.execute(DiffusionUniformKh, runtime=runtime, dt=np.timedelta64(1, "h"))

    expected_std_lon = np.sqrt(2 * kh_zonal * mesh_conversion**2 * timedelta_to_float(runtime))
    expected_std_lat = np.sqrt(2 * kh_meridional * mesh_conversion**2 * timedelta_to_float(runtime))

    tol = 500 * mesh_conversion  # effectively 500 m errors
    np.testing.assert_allclose(np.std(pset.y), expected_std_lat, atol=tol)
    np.testing.assert_allclose(np.std(pset.x), expected_std_lon, atol=tol)
    np.testing.assert_allclose(np.mean(pset.x), 0, atol=tol)
    np.testing.assert_allclose(np.mean(pset.y), 0, atol=tol)


@pytest.mark.parametrize("mesh", ["spherical", "flat"])
@pytest.mark.parametrize("kernel", [AdvectionDiffusionM1, AdvectionDiffusionEM])
def test_fieldKh_SpatiallyVaryingDiffusion(mesh, kernel):
    """Test advection-diffusion kernels on a non-uniform diffusivity field with a linear gradient in one direction."""
    ydim, xdim = 100, 200

    mesh_conversion = 1 / 1852.0 / 60 if mesh == "spherical" else 1
    ds = simple_UV_dataset(n_times=2, nz=1, ny=ydim, nx=xdim, mesh=mesh)
    ds["lon"].data = np.linspace(-1e6, 1e6, xdim)
    ds["lat"].data = np.linspace(-1e6, 1e6, ydim)

    Kh = np.zeros((ydim, xdim), dtype=np.float32)
    for x in range(xdim):
        Kh[:, x] = np.tanh(ds["lon"][x] / ds["lon"][-1] * 10.0) * xdim / 2.0 + xdim / 2.0 + 100.0

    ds["Kh_zonal"] = (["time", "depth", "YG", "XG"], np.full((2, 1, ydim, xdim), Kh))
    ds["Kh_meridional"] = (["time", "depth", "YG", "XG"], np.full((2, 1, ydim, xdim), Kh))
    fieldset = FieldSet.from_sgrid_conventions(ds)
    fieldset.add_context("dres", float(ds["lon"][1] - ds["lon"][0]))

    npart = 10000

    np.random.seed(1636)
    pset = ParticleSet(fieldset=fieldset, x=np.zeros(npart), y=np.zeros(npart))
    pset.execute(kernel, runtime=np.timedelta64(3, "h"), dt=np.timedelta64(1, "h"))

    tol = 2000 * mesh_conversion  # effectively 2000 m errors (because of low numbers of particles)
    assert np.allclose(np.mean(pset.x), 0, atol=tol)
    assert np.allclose(np.mean(pset.y), 0, atol=tol)
    assert abs(stats.skew(pset.x)) > abs(stats.skew(pset.y))


@pytest.mark.parametrize("lambd", [1, 5])
def test_randomexponential(lambd):
    fieldset = create_fieldset_zeros_conversion()
    npart = 1000

    # Rate parameter for random.expovariate
    fieldset.add_context("lambd", lambd)

    # Set random seed
    np.random.seed(1234)

    pset = ParticleSet(fieldset=fieldset, x=np.zeros(npart), y=np.zeros(npart), z=np.zeros(npart))

    def vertical_randomexponential(particles, fieldset):  # pragma: no cover
        # Kernel for random exponential variable in z direction
        particles.dz += np.random.exponential(scale=1 / fieldset.lambd, size=len(particles))

    pset.execute(vertical_randomexponential, runtime=np.timedelta64(1, "s"), dt=np.timedelta64(1, "s"))

    expected_mean = 1.0 / fieldset.lambd
    assert np.allclose(np.mean(pset.z), expected_mean, rtol=0.1)


@pytest.mark.parametrize("mu", [0.8 * np.pi, np.pi])
@pytest.mark.parametrize("kappa", [2, 4])
def test_randomvonmises(mu, kappa):
    npart = 10000
    fieldset = create_fieldset_zeros_conversion()

    # Parameters for random.vonmisesvariate
    fieldset.mu = mu
    fieldset.kappa = kappa

    # Set random seed
    np.random.seed(1234)

    AngleParticle = Particle.add_variable(Variable("angle"))
    pset = ParticleSet(fieldset=fieldset, pclass=AngleParticle, x=np.zeros(npart), y=np.zeros(npart), z=np.zeros(npart))

    def vonmises(particles, fieldset):  # pragma: no cover
        particles.angle = np.array([random.vonmisesvariate(fieldset.mu, fieldset.kappa) for _ in range(len(particles))])

    pset.execute(vonmises, runtime=np.timedelta64(1, "s"), dt=np.timedelta64(1, "s"))

    assert np.allclose(np.mean(pset.angle), mu, atol=0.1)
    vonmises_mean = stats.vonmises.mean(kappa=kappa, loc=mu)
    assert np.allclose(np.mean(pset.angle), vonmises_mean, atol=0.1)
    vonmises_var = stats.vonmises.var(kappa=kappa, loc=mu)
    assert np.allclose(np.var(pset.angle), vonmises_var, atol=0.1)
