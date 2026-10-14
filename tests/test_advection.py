import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import parcels
import parcels.tutorial
from parcels import (
    FieldSet,
    Particle,
    ParticleFile,
    ParticleSet,
    StatusCode,
    Variable,
    convert,
)
from parcels._core.utils.time import timedelta_to_float
from parcels._core.warnings import FieldEvalWarning
from parcels._datasets.structured.generated import (
    decaying_moving_eddy_dataset,
    moving_eddy_dataset,
    peninsula_dataset,
    radial_rotation_dataset,
    simple_UV_dataset,
    stommel_gyre_dataset,
)
from parcels._datasets.structured.generic import datasets_sgrid
from parcels.kernels import (
    RK2,
    RK4,
    AdvectionDiffusionEM,
    AdvectionDiffusionM1,
    AdvectionEE,
    AdvectionRK2,
    AdvectionRK2_3D,
    AdvectionRK4,
    AdvectionRK4_3D,
    AdvectionRK45,
)
from tests.mark import ignore_kernel_warnings
from tests.utils import DEFAULT_PARTICLES, assert_cftime_like_particlefile


@pytest.mark.parametrize("mesh", ["spherical", "flat"])
def test_advection_zonal(mesh, npart=10):
    """Particles at high latitude move geographically faster due to the pole correction."""
    ds = simple_UV_dataset(mesh=mesh)
    ds["U"].data[:] = 1.0
    fieldset = FieldSet.from_sgrid_conventions(ds, mesh=mesh)

    runtime = 7200
    startlat = np.linspace(0, 80, npart)
    startlon = 20.0 + np.zeros(npart)
    pset = ParticleSet(fieldset, x=startlon, y=startlat)
    pset.execute(AdvectionRK4, runtime=runtime, dt=np.timedelta64(15, "m"))

    expected_dlon = runtime
    if mesh == "spherical":
        expected_dlon /= 1852 * 60 * np.cos(np.deg2rad(pset.y))

    np.testing.assert_allclose(pset.x - startlon, expected_dlon, atol=1e-5)
    np.testing.assert_allclose(pset.y, startlat, atol=1e-5)


def test_advection_zonal_with_particlefile(tmp_parquet):
    """Particles at high latitude move geographically faster due to the pole correction."""
    npart = 10
    ds = simple_UV_dataset(mesh="flat")
    ds["U"].data[:] = 1.0
    fieldset = FieldSet.from_sgrid_conventions(ds, mesh="flat")

    pset = ParticleSet(fieldset, x=np.zeros(npart) + 20.0, y=np.linspace(0, 80, npart))
    pfile = ParticleFile(tmp_parquet, outputdt=np.timedelta64(30, "m"))
    pset.execute(AdvectionRK4, runtime=np.timedelta64(2, "h"), dt=np.timedelta64(15, "m"), output_file=pfile)

    assert (np.diff(pset.x) < 1.0e-4).all()
    df = pd.read_parquet(tmp_parquet)
    final_time = df["t"].max()
    np.testing.assert_allclose(df[df["t"] == final_time]["x"].values, pset.x, atol=1e-5)
    assert_cftime_like_particlefile(tmp_parquet)


def periodicBC(particles, fieldset):
    particles.total_dlon += particles.dx
    particles.x = np.fmod(particles.x, 2)


@ignore_kernel_warnings
def test_advection_zonal_periodic():
    ds = simple_UV_dataset(dims=(2, 2, 2, 2), mesh="flat")
    ds["U"].data[:] = 0.1
    ds["lon"].data = np.array([0, 2])
    ds["lat"].data = np.array([0, 2])

    # add a halo
    halo = ds.isel(XG=0)
    halo.lon.values = ds.lon.values[1] + 1
    halo.XG.values = ds.XG.values[1] + 2
    ds = xr.concat([ds, halo], dim="XG", data_vars="all")

    fieldset = FieldSet.from_sgrid_conventions(ds, mesh="flat")

    PeriodicParticle = Particle.add_variable(Variable("total_dlon", initial=0))
    startlon = np.array([0.5, 0.4])
    pset = ParticleSet(fieldset, pclass=PeriodicParticle, x=startlon, y=[0.5, 0.5])
    pset.execute([AdvectionEE, periodicBC], runtime=np.timedelta64(40, "s"), dt=np.timedelta64(1, "s"))
    np.testing.assert_allclose(pset.total_dlon, 4.0, atol=1e-5)
    np.testing.assert_allclose(pset.x, startlon, atol=1e-5)
    np.testing.assert_allclose(pset.y, 0.5, atol=1e-5)


@pytest.mark.parametrize("mesh", ["spherical", "flat"])
def test_advection_meridional(mesh, npart=10):
    """All particles move the same in meridional direction, regardless of latitude."""
    ds = simple_UV_dataset(mesh=mesh)
    ds["V"].data[:] = 1.0
    fieldset = FieldSet.from_sgrid_conventions(ds, mesh=mesh)

    runtime = 7200
    startlat = np.linspace(0, 80, npart)
    startlon = 20.0 + np.zeros(npart)
    pset = ParticleSet(fieldset, x=startlon, y=startlat)
    pset.execute(AdvectionRK4, runtime=runtime, dt=np.timedelta64(15, "m"))

    expected_dlat = runtime
    if mesh == "spherical":
        expected_dlat /= 1852 * 60

    np.testing.assert_allclose(pset.x, startlon, atol=1e-5)
    np.testing.assert_allclose(pset.y - startlat, expected_dlat, atol=1e-4)


@pytest.mark.parametrize("mesh", ["spherical", "flat"])
def test_horizontal_advection_in_3D_flow(mesh, npart=10):
    """2D zonal flow that increases linearly with z from 0 m/s to 1 m/s."""
    ds = simple_UV_dataset(mesh=mesh)
    ds["U"].data[:] = 1.0
    ds["U"].data[:, 0, :, :] = 0.0  # Set U to 0 at the surface
    fieldset = FieldSet.from_sgrid_conventions(ds, mesh=mesh)

    pset = ParticleSet(fieldset, x=np.zeros(npart), y=np.zeros(npart), z=np.linspace(0.1, 0.9, npart))
    pset.execute(AdvectionRK4, runtime=np.timedelta64(2, "h"), dt=np.timedelta64(15, "m"))

    expected_lon = pset.z * pset.t
    if mesh == "spherical":
        expected_lon /= 1852 * 60 * np.cos(np.deg2rad(pset.y))
    np.testing.assert_allclose(pset.x, expected_lon, atol=1.0e-1)


@pytest.mark.parametrize("direction", ["up", "down"])
@pytest.mark.parametrize("resubmerge_particle", [True, False])
def test_advection_3D_outofbounds(direction, resubmerge_particle):
    ds = simple_UV_dataset(mesh="flat")
    ds["W"] = ds["V"].copy()  # Just to have W field present
    ds["U"].data[:] = 0.01  # Set U to small value (to avoid horizontal out of bounds)
    ds["W"].data[:] = -1.0 if direction == "up" else 1.0
    fieldset = FieldSet.from_sgrid_conventions(ds, mesh="flat")

    def DeleteParticle(particles, fieldset):  # pragma: no cover
        particles.state = np.where(particles.state == StatusCode.ErrorOutOfBounds, StatusCode.Delete, particles.state)
        particles.state = np.where(
            particles.state == StatusCode.ErrorThroughSurface, StatusCode.Delete, particles.state
        )

    def SubmergeParticle(particles, fieldset):  # pragma: no cover
        if len(particles.state) == 0:
            return
        inds = np.argwhere(particles.state == StatusCode.ErrorThroughSurface).flatten()
        if len(inds) == 0:
            return
        (u, v) = fieldset.UV[particles[inds]]
        particles[inds].dx = u * particles.dt
        particles[inds].dy = v * particles.dt
        particles[inds].dz = 0.0
        particles[inds].z = 0
        particles[inds].state = StatusCode.Success

    kernels = [AdvectionRK4_3D]
    if resubmerge_particle:
        kernels.append(SubmergeParticle)
    kernels.append(DeleteParticle)

    pset = ParticleSet(fieldset=fieldset, x=0.5, y=0.5, z=0.9)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FieldEvalWarning)
        pset.execute(kernels, runtime=np.timedelta64(10, "s"), dt=np.timedelta64(1, "s"))

    if direction == "up" and resubmerge_particle:
        np.testing.assert_allclose(pset.x[0], 0.6, atol=1e-5)
        np.testing.assert_allclose(pset.z[0], 0, atol=1e-5)
    else:
        assert len(pset) == 0


@pytest.mark.parametrize(
    "u_value, x_slice", [(-0.03, slice(0, 1)), (0.02, slice(None))], ids=["single_u_layer", "full_u"]
)
@pytest.mark.parametrize(
    "v_value, y_slice", [(0.02, slice(0, 1)), (0.1, slice(None))], ids=["single_v_layer", "full_v"]
)
@pytest.mark.parametrize(
    "w_value, z_slice",
    [(None, None), (-0.02, slice(0, 1)), (0.07, slice(None))],
    ids=["no_vertical", "single_w_layer", "full_w"],
)
def test_length1dimensions(u_value, x_slice, v_value, y_slice, w_value, z_slice):
    ds = datasets_sgrid["ds_2d_padded_high"].copy()[["U_A_grid", "grid"]]
    ds = ds.isel(time=slice(2))  # TODO make this also work for length-1 time dimensions
    ds = ds.rename({"U_A_grid": "U"})
    ds["U"] = xr.full_like(ds["U"], u_value)
    ds["V"] = xr.full_like(ds["U"], v_value)
    if w_value is not None:
        ds["W"] = xr.full_like(ds["U"], w_value)

    # Slice dataset
    indexers = {"node_dimension1": x_slice, "node_dimension2": y_slice}
    if w_value:
        indexers.update({"vertical_dimensions_dim1": z_slice})
    ds = ds.sgrid.isel(indexers)

    fieldset = FieldSet.from_sgrid_conventions(ds, mesh="flat")

    x0, y0, z0 = 3, 3, 20
    pset = ParticleSet(fieldset, x=x0, y=y0, z=z0)
    kernel = AdvectionRK4 if w_value is None else AdvectionRK4_3D
    pset.execute(kernel, runtime=np.timedelta64(4, "s"), dt=np.timedelta64(1, "s"))

    assert len(pset.x) == len([p.x for p in pset])
    np.testing.assert_allclose(np.array([p.x - x0 for p in pset]), 4 * u_value, atol=1e-5)
    np.testing.assert_allclose(np.array([p.y - y0 for p in pset]), 4 * v_value, atol=1e-5)
    if w_value:
        np.testing.assert_allclose(np.array([p.z - z0 for p in pset]), 4 * w_value, atol=1e-5)


@pytest.mark.parametrize("npart", [1, 2, 3, 10])
@pytest.mark.parametrize("integrator", [RK2, RK4])
def test_advection_with_integrators_singlefield(npart, integrator):
    """Test that the RK2 and RK4 integrators don't work with a single field (e.g. U)"""
    ds = simple_UV_dataset(mesh="flat").rename({"U": "T"})
    fset = parcels.FieldSet.from_sgrid_conventions(ds, mesh="flat")

    pset = parcels.ParticleSet(fset, x=np.zeros(npart), y=np.arange(npart))

    def rhs(fieldset, t, z, y, x, particles):
        return fieldset.T[t, z, y, x, particles]

    def SingleField(particles, fieldset):
        integrator(particles, fieldset, rhs)

    with pytest.raises(TypeError):
        pset.execute(SingleField, runtime=np.timedelta64(1, "s"), dt=np.timedelta64(1, "s"))


def test_radialrotation(npart=10):
    ds = radial_rotation_dataset()
    fieldset = parcels.FieldSet.from_sgrid_conventions(ds, mesh="flat")

    dt = np.timedelta64(30, "s")
    lon = np.linspace(32, 50, npart)
    lat = np.ones(npart) * 30
    starttime = np.arange(np.timedelta64(0, "s"), npart * dt, dt)
    endtime = np.timedelta64(10, "m")

    pset = parcels.ParticleSet(fieldset, x=lon, y=lat, t=starttime)
    pset.execute(parcels.kernels.AdvectionRK4, endtime=endtime, dt=dt)

    theta = 2 * np.pi * (pset.t - timedelta_to_float(starttime)) / (24 * 3600)
    true_lon = (lon - 30.0) * np.cos(theta) + 30.0
    true_lat = -(lon - 30.0) * np.sin(theta) + 30.0

    np.testing.assert_allclose(pset.x, true_lon, atol=5e-2)
    np.testing.assert_allclose(pset.y, true_lat, atol=5e-2)


@pytest.mark.parametrize(
    "kernel, rtol",
    [
        (AdvectionEE, 1e-2),
        (AdvectionDiffusionEM, 1e-2),
        (AdvectionDiffusionM1, 1e-2),
        (AdvectionRK2, 1e-4),
        (AdvectionRK2_3D, 1e-4),
        (AdvectionRK4, 1e-5),
        (AdvectionRK4_3D, 1e-5),
        (AdvectionRK45, 1e-4),
    ],
)
def test_moving_eddy(kernel, rtol):
    ds = moving_eddy_dataset()
    if kernel in [AdvectionRK2_3D, AdvectionRK4_3D]:
        # Using W to test 3D advection (assuming same velocity as V)
        ds["W"] = ds["V"]

    if kernel in [AdvectionDiffusionEM, AdvectionDiffusionM1]:
        # Add zero diffusivity field for diffusion kernels
        ds["Kh_zonal"] = (["time", "depth", "YG", "XG"], np.full(ds["U"].shape, 0))
        ds["Kh_meridional"] = ds["Kh_zonal"]

    fieldset = FieldSet.from_sgrid_conventions(ds, mesh="flat")

    if kernel in [AdvectionDiffusionEM, AdvectionDiffusionM1]:
        fieldset.add_context("dres", 0.1)

    start_lon, start_lat, start_z = 12000, 12500, 12500
    dt = np.timedelta64(30, "m")
    endtime = np.timedelta64(1, "h")

    if kernel == AdvectionRK45:
        fieldset.add_context("RK45_tol", rtol)
        fieldset.add_context("RK45_min_dt", 1)
        fieldset.add_context("RK45_max_dt", 24 * 60 * 60)

    pset = ParticleSet(
        fieldset, pclass=DEFAULT_PARTICLES[kernel], x=start_lon, y=start_lat, z=start_z, t=np.timedelta64(0, "s")
    )
    pset.execute(kernel, dt=dt, endtime=endtime)

    def truth_moving(x_0, y_0, t):
        t /= np.timedelta64(1, "s")
        lat = y_0 - (ds.u_0 - ds.u_g) / ds.f * (1 - np.cos(ds.f * t))
        lon = x_0 + ds.u_g * t + (ds.u_0 - ds.u_g) / ds.f * np.sin(ds.f * t)
        return lon, lat

    exp_lon, exp_lat = truth_moving(start_lon, start_lat, endtime)
    np.testing.assert_allclose(pset.x, exp_lon, rtol=rtol)
    np.testing.assert_allclose(pset.y, exp_lat, rtol=rtol)
    if kernel == AdvectionRK4_3D:
        np.testing.assert_allclose(pset.z, exp_lat, rtol=rtol)


@pytest.mark.parametrize(
    "kernel, rtol",
    [
        (AdvectionEE, 1e-1),
        (AdvectionRK2, 3e-3),
        (AdvectionRK4, 1e-5),
        (AdvectionRK45, 1e-4),
    ],
)
def test_decaying_moving_eddy(kernel, rtol):
    ds = decaying_moving_eddy_dataset()
    fieldset = FieldSet.from_sgrid_conventions(ds, mesh="flat")

    start_lon, start_lat = 10000, 10000
    dt = np.timedelta64(60, "m")
    endtime = np.timedelta64(23, "h")

    if kernel == AdvectionRK45:
        fieldset.add_context("RK45_tol", rtol)
        fieldset.add_context("RK45_min_dt", 10 * 60)
        fieldset.add_context("RK45_max_dt", 24 * 60 * 60)

    pset = ParticleSet(fieldset, pclass=DEFAULT_PARTICLES[kernel], x=start_lon, y=start_lat, t=np.timedelta64(0, "s"))
    pset.execute(kernel, dt=dt, endtime=endtime)

    def truth_moving(x_0, y_0, t):
        t /= np.timedelta64(1, "s")
        lon = (
            x_0
            + (ds.u_g / ds.gamma_g) * (1 - np.exp(-ds.gamma_g * t))
            + ds.f
            * ((ds.u_0 - ds.u_g) / (ds.f**2 + ds.gamma**2))
            * ((ds.gamma / ds.f) + np.exp(-ds.gamma * t) * (np.sin(ds.f * t) - (ds.gamma / ds.f) * np.cos(ds.f * t)))
        )
        lat = y_0 - ((ds.u_0 - ds.u_g) / (ds.f**2 + ds.gamma**2)) * ds.f * (
            1 - np.exp(-ds.gamma * t) * (np.cos(ds.f * t) + (ds.gamma / ds.f) * np.sin(ds.f * t))
        )
        return lon, lat

    exp_lon, exp_lat = truth_moving(start_lon, start_lat, endtime)
    np.testing.assert_allclose(pset.x, exp_lon, rtol=rtol)
    np.testing.assert_allclose(pset.y, exp_lat, rtol=rtol)


@pytest.mark.parametrize(
    "kernel, rtol",
    [
        (AdvectionRK2, 0.1),
        (AdvectionRK4, 0.1),
        (AdvectionRK45, 0.1),
    ],
)
@pytest.mark.parametrize("grid_type", ["A", "C"])
def test_stommelgyre_fieldset(kernel, rtol, grid_type):
    npart = 2
    fieldset = FieldSet.from_sgrid_conventions(stommel_gyre_dataset(grid_type=grid_type), mesh="flat")

    dt = np.timedelta64(30, "m")
    runtime = np.timedelta64(1, "D")
    start_lon = np.linspace(10e3, 100e3, npart)
    start_lat = np.ones_like(start_lon) * 5000e3

    SampleParticle = DEFAULT_PARTICLES[kernel].add_variable(
        [Variable("p", initial=0.0, dtype=np.float32), Variable("p_start", initial=0.0, dtype=np.float32)]
    )

    if kernel == AdvectionRK45:
        fieldset.add_context("RK45_tol", rtol)
        fieldset.add_context("RK45_min_dt", 1)
        fieldset.add_context("RK45_max_dt", 24 * 60 * 60)

    def UpdateP(particles, fieldset):  # pragma: no cover
        particles.p = fieldset.P[particles.t, particles.z, particles.y, particles.x]
        particles.p_start = np.where(particles.t == 0, particles.p, particles.p_start)

    pset = ParticleSet(fieldset, pclass=SampleParticle, x=start_lon, y=start_lat, t=np.timedelta64(0, "s"))
    pset.execute([kernel, UpdateP], dt=dt, runtime=runtime)
    np.testing.assert_allclose(pset.p, pset.p_start, rtol=rtol)


@pytest.mark.parametrize(
    "kernel, rtol",
    [
        (AdvectionRK2, 2e-2),
        (AdvectionRK4, 1e-2),
        (AdvectionRK45, 1e-3),
    ],
)
@pytest.mark.parametrize("mesh", ["flat", "spherical"])
@pytest.mark.parametrize("grid_type", ["A", "C"])
def test_peninsula_fieldset(kernel, rtol, grid_type, mesh):
    npart = 2
    ds = peninsula_dataset(grid_type=grid_type)
    fieldset = FieldSet.from_sgrid_conventions(ds, mesh=mesh)

    dt = np.timedelta64(30, "m")
    runtime = np.timedelta64(23, "h")
    start_lat = np.linspace(3e3, 47e3, npart)
    start_lon = 3e3 * np.ones_like(start_lat)

    SampleParticle = DEFAULT_PARTICLES[kernel].add_variable(
        [Variable("p", initial=0.0, dtype=np.float32), Variable("p_start", initial=0.0, dtype=np.float32)]
    )

    if kernel == AdvectionRK45:
        fieldset.add_context("RK45_tol", rtol)
        fieldset.add_context("RK45_min_dt", 10)
        fieldset.add_context("RK45_max_dt", 24 * 60 * 60)

    def UpdateP(particles, fieldset):  # pragma: no cover
        particles.p = fieldset.P[particles.t, particles.z, particles.y, particles.x]
        particles.p_start = np.where(particles.t == 0, particles.p, particles.p_start)

    pset = ParticleSet(fieldset, pclass=SampleParticle, x=start_lon, y=start_lat, t=np.timedelta64(0, "s"))
    pset.execute([kernel, UpdateP], dt=dt, runtime=runtime)
    np.testing.assert_allclose(pset.p, pset.p_start, rtol=rtol)


def test_nemo_curvilinear_fieldset():
    U = parcels.tutorial.open_dataset("NemoCurvilinear_data_zonal/U")
    V = parcels.tutorial.open_dataset("NemoCurvilinear_data_zonal/V")
    coords = parcels.tutorial.open_dataset("NemoCurvilinear_data_zonal/mesh_mask")

    ds = parcels.convert.nemo_to_sgrid(fields=dict(U=U, V=V), coords=coords)

    fieldset = parcels.FieldSet.from_sgrid_conventions(ds)

    npart = 20
    lonp = 30 * np.ones(npart)
    latp = np.linspace(-70, 88, npart)
    runtime = np.timedelta64(160, "D")

    pset = parcels.ParticleSet(fieldset, x=lonp, y=latp)
    pset.execute(AdvectionEE, runtime=runtime, dt=np.timedelta64(10, "D"))
    np.testing.assert_allclose(pset.y, latp, atol=1e-1)


@pytest.mark.parametrize("kernel", [AdvectionRK4, AdvectionRK4_3D])
def test_nemo_3D_curvilinear_fieldset(kernel):
    U = parcels.tutorial.open_dataset("NemoNorthSeaORCA025-N006_data/U")
    V = parcels.tutorial.open_dataset("NemoNorthSeaORCA025-N006_data/V")
    W = parcels.tutorial.open_dataset("NemoNorthSeaORCA025-N006_data/W")
    coords = parcels.tutorial.open_dataset("NemoNorthSeaORCA025-N006_data/mesh_mask")

    ds = parcels.convert.nemo_to_sgrid(fields=dict(U=U["uo"], V=V["vo"], W=W["wo"]), coords=coords)

    fieldset = parcels.FieldSet.from_sgrid_conventions(ds)

    npart = 10
    lons_initial = np.linspace(1.9, 3.4, npart)
    lats_initial = np.linspace(52.5, 51.6, npart)
    z_initial = np.ones_like(lons_initial)
    pset = parcels.ParticleSet(fieldset, x=lons_initial, y=lats_initial, z=z_initial)

    pset.execute(kernel, runtime=np.timedelta64(3, "D") + np.timedelta64(18, "h"), dt=np.timedelta64(6, "h"))

    if kernel == AdvectionRK4:
        np.testing.assert_allclose([p.z for p in pset], z_initial)
    elif kernel == AdvectionRK4_3D:
        depths_from_v3 = [
            0.66616202,
            0.86671308,
            0.92108645,
            0.95940743,
            0.95945352,
            1.00413373,
            1.02847297,
            1.00335434,
            1.27260261,
            1.38021829,
        ]  # depths from cell 1 of https://docs.parcels-code.org/en/v3.1.4/examples/tutorial_nemo_3D.html
        np.testing.assert_allclose([p.z for p in pset], depths_from_v3, rtol=2e-7)


def test_mitgcm():
    ds_fields = parcels.tutorial.open_dataset("MITgcm_example_data/mitgcm_UV_surface_zonally_reentrant")

    ds_fset = convert.mitgcm_to_sgrid(fields={"U": ds_fields.UVEL, "V": ds_fields.VVEL}, coords=ds_fields)
    fieldset = FieldSet.from_sgrid_conventions(ds_fset)

    npart = 10
    lon = [24e3] * npart
    lat = np.linspace(22e3, 1950e3, npart)

    pset = parcels.ParticleSet(fieldset, x=lon, y=lat)
    pset.execute(AdvectionRK4, runtime=np.timedelta64(1, "D"), dt=np.timedelta64(1, "h"))
    lat_v3 = [
        22363.3109351,
        241677.27730161,
        461322.50669352,
        664760.42373974,
        886760.91201257,
        1087116.48719691,
        1274515.29539529,
        1542216.3950654,
        1741733.45906654,
        1952691.93845841,
    ]
    np.testing.assert_allclose(pset.y, lat_v3, atol=1)
