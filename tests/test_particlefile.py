import tempfile
from contextlib import nullcontext as does_not_raise
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from re_assert import Matches

import parcels.tutorial
from parcels import (
    FieldSet,
    Particle,
    ParticleFile,
    ParticleSet,
    ParticleSetWarning,
    StatusCode,
    Variable,
    convert,
)
from parcels._core.particle import get_default_particle
from parcels._core.particlefile import get_schema
from parcels._core.utils.time import TimeInterval, timedelta_to_float
from parcels._datasets.structured.generated import peninsula_dataset
from parcels.kernels import AdvectionRK4
from tests.common_kernels import DoNothing


def test_particlefile_repr(tmp_parquet):
    pfile_repr = repr(ParticleFile(tmp_parquet, outputdt=np.timedelta64(1, "s")))
    match = Matches(
        r"""\<ParticleFile\>
    path                : .*
    outputdt            : 1.0
    metadata            : .*""",
    )
    match.assert_matches(pfile_repr)


def test_metadata(fieldset, tmp_parquet):
    pset = ParticleSet(fieldset, pclass=Particle, x=0, y=0)

    ofile = ParticleFile(tmp_parquet, outputdt=np.timedelta64(1, "s"))
    pset.execute(DoNothing, runtime=np.timedelta64(1, "s"), dt=np.timedelta64(1, "s"), output_file=ofile)

    tab = pq.read_table(tmp_parquet)
    assert tab.schema.metadata[b"parcels_kernels"].decode().lower() == "DoNothing".lower()


@pytest.mark.parametrize("compression", ["zstd", "gzip", "snappy", "brotli", None])
def test_compression(fieldset, tmp_parquet, compression):
    pset = ParticleSet(fieldset, pclass=Particle, x=0, y=0)

    ofile = ParticleFile(tmp_parquet, outputdt=np.timedelta64(1, "s"), compression=compression)
    pset.execute(DoNothing, runtime=np.timedelta64(1, "s"), dt=np.timedelta64(1, "s"), output_file=ofile)

    tab = pq.ParquetFile(tmp_parquet)
    for i in range(tab.num_row_groups):
        row_group = tab.metadata.row_group(i)
        for j in range(row_group.num_columns):
            col = row_group.column(j)
            assert col.compression.lower() == compression or (
                compression is None and col.compression.lower() == "uncompressed"
            )


def test_write_fieldset_without_time(tmp_parquet):
    ds = peninsula_dataset()  # DataSet without time
    assert "time" not in ds.dims
    fieldset = FieldSet.from_sgrid_conventions(ds)

    pset = ParticleSet(fieldset, pclass=Particle, x=0, y=0)

    ofile = ParticleFile(tmp_parquet, outputdt=np.timedelta64(1, "s"))
    pset.execute(DoNothing, runtime=np.timedelta64(1, "s"), dt=np.timedelta64(1, "s"), output_file=ofile)

    table = pq.read_table(tmp_parquet)
    assert table.schema.field("t").metadata[b"units"] == b"seconds"
    assert b"calendar" not in table.schema.field("t").metadata
    assert table["t"].to_numpy()[1] == 1.0


def test_pfile_array_remove_particles(fieldset, tmp_parquet):
    """If a particle from the middle of a particleset is removed, that writing doesn't crash"""
    npart = 10
    pset = ParticleSet(
        fieldset,
        pclass=Particle,
        x=np.linspace(0, 1, npart),
        y=0.5 * np.ones(npart),
        t=fieldset.time_interval.left,
    )
    pfile = ParticleFile(tmp_parquet, outputdt=np.timedelta64(1, "s"))
    pfile.set_time_interval(fieldset.time_interval)
    pset._data["t"][:] = 0
    pfile.write(pset, t=fieldset.time_interval.left)
    pset.remove_indices(3)
    new_time = 86400  # s in a day
    pset._data["t"][:] = new_time
    pfile.write(pset, new_time)
    pfile.close()


def test_pfile_array_remove_all_particles(fieldset, tmp_parquet):
    npart = 10
    pset = ParticleSet(
        fieldset,
        pclass=Particle,
        x=np.linspace(0, 1, npart),
        y=0.5 * np.ones(npart),
        t=fieldset.time_interval.left,
    )
    pfile = ParticleFile(tmp_parquet, outputdt=np.timedelta64(1, "s"))
    pfile.write(pset, t=0)
    for _ in range(npart):
        pset.remove_indices(-1)
    pfile.set_time_interval(fieldset.time_interval)
    pfile.write(pset, fieldset.time_interval.left + np.timedelta64(1, "D"))
    pfile.write(pset, fieldset.time_interval.left + np.timedelta64(2, "D"))
    pfile.close()

    df = pd.read_parquet(tmp_parquet)
    assert df["particle_id"].nunique() == npart


def test_write_dtypes_pfile(fieldset, tmp_parquet):
    dtypes = [
        np.float32,
        np.float64,
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
        np.bool_,
        np.int8,
        np.uint8,
        np.int16,
        np.uint16,
    ]

    extra_vars = [Variable(f"v_{d.__name__}", dtype=d, initial=0.0) for d in dtypes]
    MyParticle = Particle.add_variable(extra_vars)

    pset = ParticleSet(fieldset, pclass=MyParticle, x=0, y=0, t=fieldset.time_interval.left)
    pfile = ParticleFile(tmp_parquet, outputdt=np.timedelta64(1, "s"))
    pfile.set_time_interval(fieldset.time_interval)
    pfile.write(pset, t=fieldset.time_interval.left)
    pfile.close()

    tab = pq.read_table(tmp_parquet)
    for d in dtypes:
        assert tab[f"v_{d.__name__}"].type == pa.from_numpy_dtype(d)


def test_file_warnings(fieldset, tmp_parquet):
    pset = ParticleSet(fieldset, x=[0, 0], y=[0, 0], t=[np.timedelta64(0, "s"), np.timedelta64(1, "s")])
    pfile = ParticleFile(tmp_parquet, outputdt=np.timedelta64(2, "s"))
    with pytest.warns(ParticleSetWarning, match="Some of the particles have a start time difference.*"):
        pset.execute(AdvectionRK4, runtime=3, dt=1, output_file=pfile)


@pytest.mark.parametrize(
    "outputdt, expectation",
    [
        (np.timedelta64(5, "s"), does_not_raise()),
        (timedelta(seconds=2), does_not_raise()),
        (5.0, does_not_raise()),
        (np.datetime64("2001-01-02T00:00:00"), pytest.raises(ValueError)),
        (datetime(2000, 1, 2, 0, 0, 0), pytest.raises(ValueError)),
        (-np.timedelta64(5, "s"), pytest.raises(ValueError)),
    ],
)
def test_outputdt_types(outputdt, expectation, tmp_parquet):
    with expectation:
        pfile = ParticleFile(tmp_parquet, outputdt=outputdt)
        assert pfile.outputdt == timedelta_to_float(outputdt)


def test_write_timebackward(fieldset, tmp_parquet):
    release_time = fieldset.time_interval.left + [np.timedelta64(i + 1, "s") for i in range(3)]
    pset = ParticleSet(fieldset, y=[0, 1, 2], x=[0, 0, 0], t=release_time)
    pfile = ParticleFile(tmp_parquet, outputdt=np.timedelta64(1, "s"))
    pset.execute(DoNothing, runtime=np.timedelta64(3, "s"), dt=-np.timedelta64(1, "s"), output_file=pfile)

    df = pd.read_parquet(tmp_parquet)

    assert df["particle_id"].dtype == "int64"
    dt_per_particle = df.groupby("particle_id")["t"].diff().dropna()
    assert (dt_per_particle < 0).all()


@pytest.mark.parametrize("outputdt", [np.timedelta64(1, "s"), np.timedelta64(2, "s"), np.timedelta64(3, "s")])
def test_time_is_age(fieldset, tmp_parquet, outputdt):
    # Test that particle age is same as time - initial_time
    npart = 10

    AgeParticle = get_default_particle(np.float64).add_variable(Variable("age", initial=0.0))

    def IncreaseAge(particles, fieldset):  # pragma: no cover
        particles.age += particles.dt

    time = fieldset.time_interval.left + np.arange(npart) * np.timedelta64(1, "s")
    pset = ParticleSet(fieldset, pclass=AgeParticle, x=npart * [0], y=npart * [0], t=time)
    ofile = ParticleFile(tmp_parquet, outputdt=outputdt)

    if outputdt > np.timedelta64(1, "s"):
        warning_ctx = pytest.warns(ParticleSetWarning, match="Some of the particles have a start time difference.*")
    else:
        warning_ctx = does_not_raise()

    with warning_ctx:
        pset.execute(IncreaseAge, runtime=np.timedelta64(npart * 2, "s"), dt=np.timedelta64(1, "s"), output_file=ofile)

    df = parcels.read_particlefile(tmp_parquet)

    # Map sorted particle IDs to release times (0, 1, ..., npart-1 seconds)
    for i, df_traj in enumerate(df.partition_by("particle_id", maintain_order=True)):
        release_time = pd.Timestamp(time[i]).to_pydatetime()
        traj_time = (df_traj["t"] - release_time).dt.total_seconds()
        assert (df_traj["age"] == traj_time).all()


@pytest.mark.parametrize("npart", [1, 10])
def test_sampling_initial_value(fieldset, npart, tmp_parquet):
    # Test that inital value of a field gets sampled

    SampleParticle = get_default_particle(np.float64).add_variable(Variable("sample", initial=np.nan))

    def SampleKernel(particles, fieldset):  # pragma: no cover
        particles.sample, _ = fieldset.UV[particles]

    x = np.zeros(npart)
    y = np.zeros(npart)
    t = np.zeros(npart, dtype="timedelta64[s]")

    pset = ParticleSet(fieldset, pclass=SampleParticle, x=x, y=y, t=t)
    pset.sample, _ = fieldset.UV[pset]  # Sample initial value

    ofile = ParticleFile(tmp_parquet, outputdt=np.timedelta64(1, "s"))
    pset.execute(SampleKernel, runtime=np.timedelta64(2, "s"), dt=np.timedelta64(1, "s"), output_file=ofile)

    df = parcels.read_particlefile(tmp_parquet)
    np.testing.assert_allclose(df["sample"].is_finite().all(), True)


def test_reset_dt(fieldset, tmp_parquet):
    # Assert that p.dt gets reset when a write_time is not a multiple of dt
    # for p.dt=0.02 to reach outputdt=0.05 and endtime=0.1, the steps should be [0.2, 0.2, 0.1, 0.2, 0.2, 0.1], resulting in 6 kernel executions
    dt = np.timedelta64(20, "s")

    def Update_lon(particles, fieldset):  # pragma: no cover
        particles.dx += 0.1

    particle = get_default_particle(np.float64)
    pset = ParticleSet(fieldset, pclass=particle, x=[0], y=[0])
    ofile = ParticleFile(tmp_parquet, outputdt=np.timedelta64(50, "s"))
    pset.execute(Update_lon, runtime=5 * dt, dt=dt, output_file=ofile)

    assert np.allclose(pset.x, 0.6)


@pytest.mark.parametrize("dt", [100, 200])
def test_subsecond_outputdt(fieldset, dt, tmp_parquet):
    """Test that outputdt can be subsecond and that the output times are correct."""

    def Update_lon(particles, fieldset):  # pragma: no cover
        particles.dx += dt / 1000.0  # Move at a rate of 1 unit per second

    pset = ParticleSet(fieldset, x=[0], y=[0])
    ofile = ParticleFile(tmp_parquet, outputdt=np.timedelta64(dt, "ms"))
    pset.execute(Update_lon, runtime=np.timedelta64(1, "s"), dt=np.timedelta64(dt, "ms"), output_file=ofile)

    df = parcels.read_particlefile(tmp_parquet)
    np.testing.assert_allclose(df["x"], np.arange(0, 1 + 1e-6, dt / 1000.0), atol=1e-6)
    expected_t = np.arange(0, 1001, dt).astype("timedelta64[ms]")
    elapsed_t = (df["t"] - df["t"].min()).to_numpy().astype("timedelta64[ms]")
    np.testing.assert_allclose(
        elapsed_t.astype("timedelta64[ms]").astype("int"),
        expected_t.astype("timedelta64[ms]").astype("int"),
        atol=1,
    )


def test_correct_misaligned_outputdt_dt(fieldset, tmp_parquet):
    """Testing that outputdt does not need to be a multiple of dt."""

    def Update_lon(particles, fieldset):  # pragma: no cover
        particles.dx = particles.dt

    particle = get_default_particle(np.float64)
    pset = ParticleSet(fieldset, pclass=particle, x=[0], y=[0])
    ofile = ParticleFile(tmp_parquet, outputdt=np.timedelta64(3, "s"))
    pset.execute(Update_lon, runtime=np.timedelta64(11, "s"), dt=np.timedelta64(2, "s"), output_file=ofile)

    df = pd.read_parquet(tmp_parquet)
    assert np.allclose(df["x"].values, [0, 3, 6, 9])
    assert np.allclose(df["t"] - df["t"].min(), [0, 3, 6, 9])


def setup_pset_execute(*, fieldset: FieldSet, outputdt: timedelta, execute_kwargs, particle_class=Particle):
    npart = 10

    pset = ParticleSet(
        fieldset,
        pclass=particle_class,
        x=np.full(npart, fieldset.U.data.lon.mean()),
        y=np.full(npart, fieldset.U.data.lat.mean()),
    )

    with tempfile.TemporaryDirectory() as dir:
        name = f"{dir}/tmp.parquet"
        output_file = ParticleFile(name, outputdt=outputdt)

        pset.execute(DoNothing, output_file=output_file, **execute_kwargs)
        df = parcels.read_particlefile(name)

    return df


def test_pset_execute_outputdt_forwards(fieldset):
    """Testing output data dt matches outputdt in forward time."""
    outputdt = timedelta(hours=1)
    runtime = timedelta(hours=5)
    dt = timedelta(minutes=5)

    df = setup_pset_execute(fieldset=fieldset, outputdt=outputdt, execute_kwargs=dict(runtime=runtime, dt=dt))
    particle_0_times = df.filter(pl.col("particle_id") == 0)["t"]
    np.testing.assert_equal(np.diff(particle_0_times) / 1e9, outputdt.seconds)


def test_pset_execute_output_time_forwards(fieldset):
    """Testing output times start at initial time and end at initial time + runtime."""
    outputdt = np.timedelta64(1, "h")
    runtime = np.timedelta64(5, "h")
    dt = np.timedelta64(5, "m")

    df = setup_pset_execute(fieldset=fieldset, outputdt=outputdt, execute_kwargs=dict(runtime=runtime, dt=dt))
    assert df["t"].min() == pd.Timestamp(fieldset.time_interval.left)
    assert df["t"].max() - df["t"].min() == runtime


def test_pset_execute_outputdt_backwards(fieldset):
    """Testing output data dt matches outputdt in backwards time."""
    outputdt = timedelta(hours=1)
    runtime = timedelta(days=2)
    dt = -timedelta(minutes=5)

    df = setup_pset_execute(fieldset=fieldset, outputdt=outputdt, execute_kwargs=dict(runtime=runtime, dt=dt))
    particle_0_times = df.filter(pl.col("particle_id") == 0)["t"]
    np.testing.assert_equal(np.diff(particle_0_times) / 1e9, -outputdt.seconds)


def test_pset_execute_outputdt_backwards_fieldset_timevarying():
    """test_pset_execute_outputdt_backwards() still passed despite #1722 as it doesn't account for time-varying fields,
    which for some reason #1722
    """
    outputdt = timedelta(hours=1)
    runtime = timedelta(days=2)
    dt = -timedelta(minutes=5)

    # TODO: Not ideal using the `open_dataset` here, but I'm struggling to recreate this error using the test suite fieldsets we have
    ds_in = parcels.tutorial.open_dataset("CopernicusMarine_data_for_Argo_tutorial/data")
    fields = {"U": ds_in["uo"], "V": ds_in["vo"]}
    ds_fset = convert.copernicusmarine_to_sgrid(fields=fields)
    fieldset = FieldSet.from_sgrid_conventions(ds_fset)

    df = setup_pset_execute(outputdt=outputdt, execute_kwargs=dict(runtime=runtime, dt=dt), fieldset=fieldset)
    particle_0_times = df.filter(pl.col("particle_id") == 0)["t"]
    np.testing.assert_equal(np.diff(particle_0_times) / 1e9, -outputdt.seconds)


def test_particlefile_init(tmp_parquet):
    ParticleFile(tmp_parquet, outputdt=np.timedelta64(1, "s"))


def test_particlefile_init_existing_path_modes(fieldset, tmp_parquet):
    pset = ParticleSet(fieldset, pclass=Particle, x=0, y=0)

    first_file = ParticleFile(tmp_parquet, outputdt=np.timedelta64(1, "s"))
    pset.execute(DoNothing, runtime=np.timedelta64(10, "s"), dt=np.timedelta64(1, "s"), output_file=first_file)

    df_first = pd.read_parquet(tmp_parquet)

    with pytest.raises(ValueError, match="already exists"):
        ParticleFile(tmp_parquet, outputdt=np.timedelta64(1, "s"))

    overwrite_file = ParticleFile(tmp_parquet, outputdt=np.timedelta64(1, "s"), mode="w")
    pset.execute(DoNothing, runtime=np.timedelta64(10, "s"), dt=np.timedelta64(1, "s"), output_file=overwrite_file)

    df_overwrite = pd.read_parquet(tmp_parquet)

    assert len(df_first) == len(df_overwrite)


def test_particlefile_init_existing_path_no_mode(tmp_parquet):
    tmp_parquet.touch()
    with pytest.raises(ValueError, match="already exists"):
        ParticleFile(tmp_parquet, outputdt=np.timedelta64(1, "s"))


def test_particlefile_init_nonexistent_parent(tmp_path):
    path = tmp_path / "nonexistent_dir" / "file.parquet"
    with pytest.raises(ValueError, match="does not exist"):
        ParticleFile(path, outputdt=np.timedelta64(1, "s"))


def test_particlefile_init_invalid_mode(tmp_parquet):
    with pytest.raises(ValueError, match="Invalid mode value"):
        ParticleFile(tmp_parquet, outputdt=np.timedelta64(1, "s"), mode="something-else")


@pytest.mark.parametrize("name", ["path", "outputdt"])
def test_particlefile_readonly_attrs(tmp_parquet, name):
    pfile = ParticleFile(tmp_parquet, outputdt=np.timedelta64(1, "s"))
    with pytest.raises(AttributeError, match="property .* of 'ParticleFile' object has no setter"):
        setattr(pfile, name, "something")


def test_particlefile_init_invalid(tmp_path):
    path = tmp_path / "file.not-parquet"
    with pytest.raises(ValueError, match="file extension must be '.parquet'"):
        ParticleFile(path, outputdt=np.timedelta64(1, "s"))


def test_particlefile_readable_after_kernel_error(fieldset, tmp_parquet):
    """Parquet output file must be readable even if the kernel raises an error mid-execution (GH-2713)."""

    def ErrorKernel(particles, fieldset):  # pragma: no cover
        particles.state = StatusCode.Error

    pset = ParticleSet(fieldset, pclass=Particle, x=0, y=0)
    ofile = ParticleFile(tmp_parquet, outputdt=np.timedelta64(1, "s"))

    with pytest.raises(RuntimeError, match="General error occurred at"):
        pset.execute(ErrorKernel, runtime=np.timedelta64(10, "s"), dt=np.timedelta64(1, "s"), output_file=ofile)

    # File must be readable despite the crash
    df = pd.read_parquet(tmp_parquet)
    assert len(df) >= 1  # at least the initial condition was written


@pytest.mark.parametrize(
    "particle",
    [
        Particle,
        parcels.ParticleClass(
            variables=[
                Variable(
                    "lon",
                    dtype=np.float32,
                    attrs={"standard_name": "longitude", "units": "degrees_east", "axis": "X"},
                ),
                Variable(
                    "lat",
                    dtype=np.float32,
                    attrs={"standard_name": "latitude", "units": "degrees_north", "axis": "Y"},
                ),
                Variable(
                    "z",
                    dtype=np.float32,
                    attrs={"standard_name": "vertical coordinate", "units": "m", "positive": "down"},
                ),
            ]
        ),
    ],
)
def test_particle_schema(particle):
    s = get_schema(particle, {}, TimeInterval(datetime(2023, 1, 1, 12, 0), datetime(2023, 1, 2, 12, 0)))

    written_variables = [v for v in particle.variables if v.to_write]

    assert len(s.names) == len(written_variables), (
        "Number of particles in the output schema should be the same as the writable variables in the ParticleClass object."
    )

    for variable, pyarrow_field in zip(
        written_variables,
        s,
        strict=False,
    ):
        assert variable.name == pyarrow_field.name
        if variable.name != "t":
            assert variable.attrs == {k.decode(): v.decode() for k, v in pyarrow_field.metadata.items()}
        else:
            assert b"units" in pyarrow_field.metadata
            assert b"calendar" in pyarrow_field.metadata
        assert pa.from_numpy_dtype(variable.dtype) == pyarrow_field.type
