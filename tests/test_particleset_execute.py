from contextlib import nullcontext as does_not_raise
from datetime import datetime, timedelta

import numpy as np
import pytest

from parcels import (
    FieldInterpolationError,
    FieldOutOfBoundError,
    FieldSet,
    OutsideTimeInterval,
    Particle,
    ParticleFile,
    ParticleSet,
    StatusCode,
    Variable,
)
from parcels._core.statuscodes import GridSearchingError
from parcels._core.utils.time import timedelta_to_float
from parcels._datasets.structured.generated import simple_UV_dataset
from parcels._datasets.structured.generic import datasets as datasets_structured
from parcels._datasets.unstructured.generic import datasets as datasets_unstructured
from parcels.interpolators import Ux_Velocity, UxConstantFaceConstantZC
from parcels.interpolators._base import VectorInterpolator
from parcels.kernels import AdvectionEE, AdvectionRK2, AdvectionRK4, AdvectionRK4_3D, AdvectionRK45
from tests.common_kernels import DoNothing
from tests.utils import DEFAULT_PARTICLES


@pytest.fixture
def fieldset_no_time_interval() -> FieldSet:
    # i.e., no time variation
    ds = datasets_structured["ds_2d_left"].isel(time=0).drop_vars("time")

    ds = ds[["U_A_grid", "V_A_grid", "grid"]].rename(
        {
            "U_A_grid": "U",
            "V_A_grid": "V",
        }
    )
    return FieldSet.from_sgrid_conventions(ds, mesh="flat")


@pytest.fixture
def zonal_flow_fieldset() -> FieldSet:
    ds = simple_UV_dataset(mesh="flat")
    ds["U"].data[:] = 1.0
    return FieldSet.from_sgrid_conventions(ds, mesh="flat")


@pytest.fixture
def time_varying_zonal_flow_fieldset() -> FieldSet:
    """Flat A-grid whose U varies in time only (V = 0), sampled 3-hourly.

    Because U is spatially uniform, a particle's displacement depends on its own
    clock and nothing else.
    """
    nt = 25
    ds = simple_UV_dataset(n_times=nt, nz=2, ny=6, nx=6, mesh="flat")
    times = np.array([np.timedelta64(3 * i, "h") for i in range(nt)])
    ds["time"] = ("time", times, {"axis": "T"})
    u = np.cos(2 * np.pi * (times / np.timedelta64(1, "s")) / 86400.0)  # 1 m/s, 1-day period
    ds["U"].data[:] = u[:, None, None, None]
    return FieldSet.from_sgrid_conventions(ds, mesh="flat")


def test_execute_trajectory_independent_of_other_particles_release_times(time_varying_zonal_flow_fieldset):
    """A particle's trajectory must not depend on when its batch-mates were released.

    ``ParticleSet`` accepts a per-particle ``t``, so a set can hold particles on
    different time indices. Interpolation is per-particle, hence batching a
    particle with later-released mates must not change its own result.
    """
    fieldset = time_varying_zonal_flow_fieldset
    t0 = np.timedelta64(0, "s")

    def run(release_times):
        npart = len(release_times)
        pset = ParticleSet(
            fieldset,
            pclass=Particle,
            t=np.array(release_times),
            z=np.zeros(npart),
            y=np.zeros(npart),
            x=np.zeros(npart),
        )
        pset.execute(AdvectionRK4, dt=np.timedelta64(1, "h"), endtime=np.timedelta64(48, "h"))
        return pset.x[0], pset.y[0]

    alone = run([t0])
    uniform = run([t0] * 4)
    staggered = run([t0] + [t0 + np.timedelta64(3, "h")] * 3)

    assert uniform == pytest.approx(alone)
    assert staggered == pytest.approx(alone)


def test_pset_execute_invalid_arguments(fieldset, fieldset_no_time_interval):
    with pytest.raises(RuntimeWarning, match="invalid value encountered in cast.*"):
        ParticleSet(fieldset, x=[0.2], y=[5.0], pclass=Particle).execute(AdvectionRK4, dt=np.timedelta64(None))

    with pytest.raises(
        ValueError,
        match="dt must be a non-zero datetime.timedelta or np.timedelta64 object, got .*",
    ):
        ParticleSet(fieldset, x=[0.2], y=[5.0], pclass=Particle).execute(AdvectionRK4, dt=np.timedelta64(0, "s"))

    with pytest.raises(
        ValueError,
        match="runtime and endtime are mutually exclusive - provide one or the other. Got .*",
    ):
        ParticleSet(fieldset, x=[0.2], y=[5.0], pclass=Particle).execute(
            AdvectionRK4, runtime=np.timedelta64(1, "s"), endtime=np.datetime64("2100-01-01"), dt=np.timedelta64(1, "s")
        )

    msg = """Calculated/provided end time of .* is not in fieldset time interval .* Either reduce your runtime, modify your provided endtime, or change your release timing.*"""
    with pytest.raises(
        ValueError,
        match=msg,
    ):
        ParticleSet(fieldset, x=[0.2], y=[5.0], pclass=Particle).execute(
            AdvectionRK4, endtime=np.datetime64("1990-01-01"), dt=np.timedelta64(1, "s")
        )

    with pytest.raises(
        ValueError,
        match=msg,
    ):
        ParticleSet(fieldset, x=[0.2], y=[5.0], pclass=Particle).execute(
            AdvectionRK4, endtime=np.datetime64("2100-01-01"), dt=np.timedelta64(-1, "s")
        )

    with pytest.raises(
        ValueError,
        match="The endtime must be of the same type as the fieldset.time_interval start time. Got .*",
    ):
        ParticleSet(fieldset, x=[0.2], y=[5.0], pclass=Particle).execute(
            AdvectionRK4, endtime=12345, dt=np.timedelta64(1, "s")
        )

    with pytest.raises(
        ValueError,
        match="The runtime must be provided when the time_interval is not defined for a fieldset.",
    ):
        ParticleSet(fieldset_no_time_interval, x=[0.2], y=[5.0], pclass=Particle).execute(
            AdvectionRK4, dt=np.timedelta64(1, "s")
        )


@pytest.mark.parametrize(
    "runtime, expectation",
    [
        (np.timedelta64(5, "s"), does_not_raise()),
        (timedelta(seconds=2), does_not_raise()),
        (5.0, does_not_raise()),
        (np.datetime64("2001-01-02T00:00:00"), pytest.raises(ValueError)),
        (datetime(2000, 1, 2, 0, 0, 0), pytest.raises(ValueError)),
    ],
)
def test_particleset_runtime_type(fieldset, runtime, expectation):
    pset = ParticleSet(fieldset, x=[0.2], y=[5.0], z=[50.0], pclass=Particle)
    with expectation:
        pset.execute(runtime=runtime, dt=np.timedelta64(10, "s"), kernels=DoNothing)


@pytest.mark.parametrize(
    "endtime, expectation",
    [
        (np.datetime64("2000-01-02T00:00:00"), does_not_raise()),
        (5.0, pytest.raises(ValueError)),
        (np.timedelta64(5, "s"), pytest.raises(ValueError)),
        (timedelta(seconds=2), pytest.raises(ValueError)),
        (datetime(2000, 1, 2, 0, 0, 0), pytest.raises(ValueError)),
    ],
)
def test_particleset_endtime_type(fieldset, endtime, expectation):
    pset = ParticleSet(fieldset, x=[0.2], y=[5.0], z=[50.0], pclass=Particle)
    with expectation:
        pset.execute(endtime=endtime, dt=np.timedelta64(10, "m"), kernels=DoNothing)


def test_sampleUonly(fieldset):

    def SampleU(particles, fieldset):  # pragma: no cover
        _ = fieldset.U[particles]

    pset = ParticleSet(fieldset, x=[0.2], y=[5.0])
    with pytest.raises(
        RuntimeWarning,
        match="Sampling of velocities should normally be done using fieldset.UV or fieldset.UVW object; tread carefully",
    ):
        pset.execute(SampleU, runtime=np.timedelta64(1, "D"), dt=np.timedelta64(1, "D"))


def test_particleset_run_to_endtime(fieldset):
    starttime = fieldset.time_interval.left
    endtime = fieldset.time_interval.right

    def SampleUV(particles, fieldset):  # pragma: no cover
        _, _ = fieldset.UV[particles]

    pset = ParticleSet(fieldset, x=[0.2], y=[5.0], t=[starttime])
    pset.execute(SampleUV, endtime=endtime, dt=np.timedelta64(1, "D"))
    assert np.timedelta64(int(pset[0].t), "s") + fieldset.time_interval.left == endtime


@pytest.mark.parametrize("kernel", [AdvectionEE, AdvectionRK2, AdvectionRK4, AdvectionRK45])
@pytest.mark.parametrize("dt", [np.timedelta64(10, "D"), np.timedelta64(1, "D")])
def test_particleset_run_RK_to_endtime_fwd_bwd(fieldset, kernel, dt):
    """Test that RK kernels can be run to the endtime of a fieldset (and not throw OutsideTimeInterval)"""
    starttime = fieldset.time_interval.left
    endtime = fieldset.time_interval.right

    if kernel == AdvectionRK45:
        fieldset.add_context("RK45_tol", 10)
        fieldset.add_context("RK45_min_dt", 1)
        fieldset.add_context("RK45_max_dt", 24 * 60 * 60)

    # Setting zero velocities to avoid OutofBoundsErrors
    fieldset.U.data[:] = 0.0
    fieldset.V.data[:] = 0.0

    pset = ParticleSet(fieldset, pclass=DEFAULT_PARTICLES[kernel], x=[0.2], y=[5.0], t=[starttime])
    pset.execute(kernel, endtime=endtime, dt=dt)
    assert pset[0].t == fieldset.time_interval.time_length_as_flt

    pset._requires_prepended_positionupdate_kernel = False  # Reset positionupdate_kernel use for backward

    pset.execute(kernel, endtime=starttime, dt=-dt)
    assert pset[0].t == 0.0


def test_particleset_interpolate_on_domainedge(zonal_flow_fieldset):
    fieldset = zonal_flow_fieldset

    MyParticle = Particle.add_variable(Variable("var"))

    def SampleUV(particles, fieldset):  # pragma: no cover
        particles.var, _ = fieldset.UV[particles]

    pset = ParticleSet(fieldset, pclass=MyParticle, x=fieldset.U.grid.lon[-1], y=fieldset.U.grid.lat[-1])
    pset.execute(SampleUV, runtime=np.timedelta64(1, "D"), dt=np.timedelta64(1, "D"))
    np.testing.assert_equal(pset[0].var, 1)


def test_particleset_interpolate_outside_domainedge(zonal_flow_fieldset):
    fieldset = zonal_flow_fieldset

    def SampleU(particles, fieldset):  # pragma: no cover
        particles.dx, _ = fieldset.UV[particles]

    dlat = 1e-3
    pset = ParticleSet(fieldset, x=fieldset.U.grid.lon[-1], y=fieldset.U.grid.lat[-1] + dlat)

    with pytest.raises(FieldOutOfBoundError):
        pset.execute(SampleU, runtime=np.timedelta64(2, "D"), dt=np.timedelta64(1, "D"))


@pytest.mark.parametrize(
    "dt", [np.timedelta64(1, "s"), np.timedelta64(1, "ms"), np.timedelta64(10, "ms"), np.timedelta64(1, "ns")]
)
def test_pset_execute_subsecond_dt(fieldset, dt):
    def AddDt(particles, fieldset):  # pragma: no cover
        particles.added_dt += particles.dt

    pclass = Particle.add_variable(Variable("added_dt", dtype=np.float32, initial=0))
    pset = ParticleSet(fieldset, pclass=pclass, x=0, y=0)
    pset.execute(AddDt, runtime=dt * 10, dt=dt)
    np.testing.assert_allclose(pset[0].added_dt, 10.0 * timedelta_to_float(dt), atol=1e-5)


def test_pset_remove_particle_in_kernel(fieldset):
    npart = 100
    pset = ParticleSet(fieldset, x=np.linspace(0, 1, npart), y=np.linspace(1, 0, npart))

    def DeleteKernel(particles, fieldset):  # pragma: no cover
        particles.state = np.where((particles.x >= 0.4) & (particles.x <= 0.6), StatusCode.Delete, particles.state)

    pset.execute(DeleteKernel, runtime=np.timedelta64(1, "s"), dt=np.timedelta64(1, "s"))
    indices = [i for i in range(npart) if not (40 <= i < 60)]
    assert [p.particle_id for p in pset] == indices
    assert pset[70].particle_id == 90
    assert pset[-1].particle_id == npart - 1
    assert pset.size == 80


@pytest.mark.parametrize("npart", [1, 100])
def test_pset_stop_simulation(fieldset, npart):
    pset = ParticleSet(fieldset, x=np.zeros(npart), y=np.zeros(npart), pclass=Particle)

    def Delete(particles, fieldset):  # pragma: no cover
        particles[particles.t >= 4].state = StatusCode.StopExecution

    pset.execute(Delete, dt=np.timedelta64(1, "s"), runtime=np.timedelta64(21, "s"))
    assert pset[0].t == 4


@pytest.mark.parametrize("with_delete", [True, False])
def test_pset_multi_execute(fieldset, with_delete, npart=10, n=5):
    pset = ParticleSet(fieldset, x=np.linspace(0, 1, npart), y=np.zeros(npart))

    def AddLat(particles, fieldset):  # pragma: no cover
        particles.dy += 0.1

    for _ in range(n):
        pset.execute(AddLat, runtime=np.timedelta64(1, "s"), dt=np.timedelta64(1, "s"))
        if with_delete:
            pset.remove_indices(len(pset) - 1)
    if with_delete:
        assert np.allclose(pset.y, n * 0.1, atol=1e-12)
    else:
        assert np.allclose([p.y - n * 0.1 for p in pset], np.zeros(npart), rtol=1e-12)


@pytest.mark.parametrize(
    "starttime, endtime, dt",
    [(0, 10, 1), (0, 10, 3), (2, 16, 3), (20, 10, -1), (20, 0, -2), (5, 15, 1)],
)
def test_execution_endtime(fieldset, starttime, endtime, dt):
    starttime = fieldset.time_interval.left + np.timedelta64(starttime, "s")
    endtime = fieldset.time_interval.left + np.timedelta64(endtime, "s")
    dt = np.timedelta64(dt, "s")
    pset = ParticleSet(fieldset, t=starttime, x=0, y=0)
    pset.execute(DoNothing, endtime=endtime, dt=dt)
    assert pset.t == timedelta_to_float(endtime - fieldset.time_interval.left)


def test_dont_run_particles_outside_starttime(fieldset):
    # Test forward in time (note third particle is outside endtime)
    start_times = [fieldset.time_interval.left + np.timedelta64(t, "s") for t in [0, 2, 10]]
    endtime = fieldset.time_interval.left + np.timedelta64(8, "s")

    def AddLon(particles, fieldset):  # pragma: no cover
        particles.dx += 1

    pset = ParticleSet(fieldset, x=np.zeros(len(start_times)), y=np.zeros(len(start_times)), t=start_times)
    pset.execute(AddLon, dt=np.timedelta64(1, "s"), endtime=endtime)

    np.testing.assert_array_equal(pset.x, [8, 6, 0])
    assert pset.t[0:1] == timedelta_to_float(endtime - fieldset.time_interval.left)
    assert pset.t[2] == timedelta_to_float(
        start_times[2] - fieldset.time_interval.left
    )  # this particle has not been executed

    # Test backward in time (note third particle is outside endtime)
    start_times = [fieldset.time_interval.right - np.timedelta64(t, "s") for t in [0, 2, 10]]
    endtime = fieldset.time_interval.right - np.timedelta64(8, "s")

    pset = ParticleSet(fieldset, x=np.zeros(len(start_times)), y=np.zeros(len(start_times)), t=start_times)
    pset.execute(AddLon, dt=-np.timedelta64(1, "s"), endtime=endtime)

    np.testing.assert_array_equal(pset.x, [8, 6, 0])
    assert pset.t[0:1] == timedelta_to_float(endtime - fieldset.time_interval.left)
    assert pset.t[2] == timedelta_to_float(
        start_times[2] - fieldset.time_interval.left
    )  # this particle has not been executed


def test_some_particles_throw_outofbounds(zonal_flow_fieldset):
    npart = 100
    lon = np.linspace(0, 9e5, npart)
    pset = ParticleSet(zonal_flow_fieldset, x=lon, y=np.zeros_like(lon))

    with pytest.raises(FieldOutOfBoundError):
        pset.execute(AdvectionEE, runtime=np.timedelta64(1_000_000, "s"), dt=np.timedelta64(10_000, "s"))


def test_delete_on_all_errors(fieldset):
    def MoveRight(particles, fieldset):  # pragma: no cover
        particles.dx += 1
        fieldset.UV[particles.t, particles.z, particles.y, particles.x, particles]

    def DeleteAllErrorParticles(particles, fieldset):  # pragma: no cover
        particles[particles.state > 20].state = StatusCode.Delete

    pset = ParticleSet(fieldset, x=[1e5, 2], y=[0, 0])
    pset.execute([MoveRight, DeleteAllErrorParticles], runtime=np.timedelta64(10, "s"), dt=np.timedelta64(1, "s"))
    assert len(pset) == 0


def test_some_particles_throw_outoftime(fieldset):
    time = [fieldset.time_interval.left + np.timedelta64(t, "D") for t in [0, 350]]
    pset = ParticleSet(fieldset, x=np.zeros_like(time), y=np.zeros_like(time), t=time)

    def FieldAccessOutsideTime(particles, fieldset):  # pragma: no cover
        fieldset.UV[particles.t + 400 * 86400, particles.z, particles.y, particles.x, particles]

    with pytest.raises(OutsideTimeInterval):
        pset.execute(FieldAccessOutsideTime, runtime=np.timedelta64(1, "D"), dt=np.timedelta64(10, "D"))


def test_raise_grid_searching_error(): ...


def test_raise_general_error(): ...


def test_errorinterpolation(fieldset):
    class NaNInterpolator(VectorInterpolator):  # pragma: no cover
        def interp(self, particle_positions, grid_positions, field):
            nanvals = np.nan * np.zeros_like(particle_positions["x"])
            return nanvals, nanvals, nanvals

    def SampleUV(particles, fieldset):  # pragma: no cover
        fieldset.UV[particles.t, particles.z, particles.y, particles.x, particles]

    fieldset.UV.interp_method = NaNInterpolator()
    pset = ParticleSet(fieldset, x=[0, 2], y=[0, 0])
    with pytest.raises(FieldInterpolationError):
        pset.execute(SampleUV, runtime=np.timedelta64(2, "s"), dt=np.timedelta64(1, "s"))


def test_execution_check_stopallexecution(fieldset):
    def addoneLon(particles, fieldset):  # pragma: no cover
        particles.dx += 1
        particles[particles.x + particles.dx >= 10].state = StatusCode.StopAllExecution

    pset = ParticleSet(fieldset, x=[0, 0], y=[0, 0])
    pset.execute(addoneLon, runtime=np.timedelta64(20, "s"), dt=np.timedelta64(1, "s"))
    np.testing.assert_allclose(pset.x, 9)
    np.testing.assert_allclose(pset.t, 9)


def test_execution_recover_out_of_bounds(fieldset):
    npart = 2

    def MoveRight(particles, fieldset):  # pragma: no cover
        fieldset.UV[particles.t, particles.z, particles.y, particles.x + 0.1, particles]
        particles.dx += 0.1

    def MoveLeft(particles, fieldset):  # pragma: no cover
        inds = np.where(particles.state == StatusCode.ErrorOutOfBounds)
        particles[inds].dx -= 1.0
        particles[inds].state = StatusCode.Success

    lon = np.linspace(0.05, 6.95, npart)
    lat = np.linspace(1, 0, npart)
    pset = ParticleSet(fieldset, x=lon, y=lat)
    pset.execute([MoveRight, MoveLeft], runtime=np.timedelta64(60, "s"), dt=np.timedelta64(1, "s"))
    assert len(pset) == npart
    np.testing.assert_allclose(pset.x, [6.05, 5.95], rtol=1e-5)
    np.testing.assert_allclose(pset.y, lat, rtol=1e-5)


@pytest.mark.parametrize(
    "starttime_flt, runtime_flt, dt",
    [(0, 10, 1), (0, 10, 3), (2, 16, 3), (20, 10, -1), (20, 0, -2), (5, 15, 1)],
)
@pytest.mark.parametrize("npart", [1, 10])
def test_execution_runtime(fieldset, starttime_flt, runtime_flt, dt, npart):
    starttime = fieldset.time_interval.left + np.timedelta64(starttime_flt, "s")
    runtime = np.timedelta64(runtime_flt, "s")
    sign_dt = np.sign(dt)
    dt = np.timedelta64(dt, "s")
    pset = ParticleSet(fieldset, t=starttime, x=np.zeros(npart), y=np.zeros(npart))
    pset.execute(DoNothing, runtime=runtime, dt=dt)
    assert all([abs(p.t - starttime_flt - runtime_flt * sign_dt) < 1e-3 for p in pset])


def test_changing_dt_in_kernel(fieldset):
    def KernelCounter(particles, fieldset):  # pragma: no cover
        particles.dx = 1

    pset = ParticleSet(fieldset, x=np.zeros(1), y=np.zeros(1))
    pset.execute(KernelCounter, dt=np.timedelta64(2, "s"), runtime=np.timedelta64(5, "s"))
    assert pset.x == 3
    assert pset.dt == 2
    assert pset.t == 5


@pytest.mark.parametrize("npart", [1, 100])
def test_execution_fail_python_exception(fieldset, npart):
    pset = ParticleSet(fieldset, x=np.linspace(0, 1, npart), y=np.linspace(1, 0, npart))

    def PythonFail(particles, fieldset):  # pragma: no cover
        inds = np.argwhere(particles.t >= 10)
        if inds.size > 0:
            raise RuntimeError("Enough is enough!")

    with pytest.raises(RuntimeError):
        pset.execute(PythonFail, runtime=np.timedelta64(20, "s"), dt=np.timedelta64(2, "s"))
    assert len(pset) == npart
    assert all(pset.t == 10)


@pytest.mark.parametrize(
    "kernel_names, expected",
    [
        ("Lat1", [0, 1]),
        ("Lat2", [2, 0]),
        ("Lat1and2", [2, 1]),
        ("Lat1then2", [2, 1]),
    ],
)
def test_execution_update_particle_in_kernel_function(fieldset, kernel_names, expected):
    npart = 2

    pset = ParticleSet(fieldset, x=np.linspace(0, 1, npart), y=np.zeros(npart))

    def Lat1(particles, fieldset):  # pragma: no cover
        def SetLat1(p):
            p.y = 1

        SetLat1(particles[(particles.y == 0) & (particles.x > 0.5)])

    def Lat2(particles, fieldset):  # pragma: no cover
        def SetLat2(p):
            p.y = 2

        SetLat2(particles[(particles.y == 0) & (particles.x < 0.5)])

    def Lat1and2(particles, fieldset):  # pragma: no cover
        def SetLat1(p):
            p.y = 1

        def SetLat2(p):
            p.y = 2

        SetLat1(particles[(particles.y == 0) & (particles.x > 0.5)])
        SetLat2(particles[(particles.y == 0) & (particles.x < 0.5)])

    if kernel_names == "Lat1":
        kernels = [Lat1]
    elif kernel_names == "Lat2":
        kernels = [Lat2]
    elif kernel_names == "Lat1and2":
        kernels = [Lat1and2]
    elif kernel_names == "Lat1then2":
        kernels = [Lat1, Lat2]

    pset.execute(kernels, runtime=np.timedelta64(2, "s"), dt=np.timedelta64(1, "s"))
    np.testing.assert_allclose(pset.y, expected, rtol=1e-5)


def test_uxstommelgyre_pset_execute():
    ds = datasets_unstructured["stommel_gyre_delaunay"]
    fieldset = FieldSet.from_ugrid_conventions(ds, mesh="spherical")
    pset = ParticleSet(
        fieldset,
        x=[30.0],
        y=[5.0],
        z=[50.0],
        t=[np.timedelta64(0, "s")],
        pclass=Particle,
    )
    pset.execute(
        AdvectionEE,
        runtime=np.timedelta64(10, "m"),
        dt=np.timedelta64(60, "s"),
    )
    np.testing.assert_allclose(pset[0].x, 29.997387, atol=1e-3)
    np.testing.assert_allclose(pset[0].y, 4.998546, atol=1e-3)


def test_uxstommelgyre_multiparticle_pset_execute():
    ds = datasets_unstructured["stommel_gyre_delaunay"]
    fieldset = FieldSet.from_ugrid_conventions(ds, mesh="spherical")
    pset = ParticleSet(
        fieldset,
        x=[30.0, 32.0],
        y=[5.0, 5.0],
        z=[50.0, 50.0],
        t=[np.timedelta64(0, "s")],
        pclass=Particle,
    )
    pset.execute(
        runtime=np.timedelta64(10, "m"),
        dt=np.timedelta64(60, "s"),
        kernels=AdvectionRK4_3D,
    )


@pytest.mark.xfail(reason="Output file not implemented yet")
def test_uxstommelgyre_pset_execute_output():
    ds = datasets_unstructured["stommel_gyre_delaunay"]
    fieldset = FieldSet.from_ugrid_conventions(ds, mesh="spherical")
    pset = ParticleSet(
        fieldset,
        x=[30.0],
        y=[5.0],
        z=[50.0],
        t=[0.0],
        pclass=Particle,
    )
    output_file = ParticleFile(
        name="stommel_uxarray_particles.zarr",  # the file name
        outputdt=np.timedelta64(5, "m"),  # the time step of the outputs
    )
    pset.execute(
        runtime=np.timedelta64(10, "m"), dt=np.timedelta64(60, "s"), kernels=AdvectionEE, output_file=output_file
    )


def test_uxgrid_particle_leaving_domain_raises():
    """A particle advecting out of an unstructured (UxGrid) domain must be flagged.

    Once the particle crosses the outflow boundary the FACE search returns
    ``GRID_SEARCH_ERROR`` and ``pset.execute`` should raise a GridSearchingError.
    """
    ds = datasets_unstructured["ux_constant_flow_face_centered_2D"]
    fieldset = FieldSet.from_ugrid_conventions(ds, mesh="flat")
    assert isinstance(fieldset.U.interp_method, UxConstantFaceConstantZC)
    assert isinstance(fieldset.V.interp_method, UxConstantFaceConstantZC)
    assert isinstance(fieldset.UV.interp_method, Ux_Velocity)

    # Uniform eastward flow (U0 = 0.001 deg/s); release 0.1 deg inside the eastern
    # outflow boundary (domain spans lon, lat in [0, 20]). The particle crosses
    # lon=20 after 100 s of travel.
    lon_max = float(ds.uxgrid.node_lon.max())
    pset = ParticleSet(fieldset, x=[lon_max - 0.1], y=[10.0], z=[0.5], pclass=Particle)

    with pytest.raises(GridSearchingError):
        pset.execute(AdvectionEE, runtime=np.timedelta64(120, "s"), dt=np.timedelta64(10, "s"))
