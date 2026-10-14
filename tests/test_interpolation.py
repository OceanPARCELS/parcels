import numpy as np
import pytest
import xarray as xr

import parcels._sgrid as sgrid
from parcels import (
    FieldSet,
    Particle,
    ParticleFile,
    ParticleSet,
    StatusCode,
    Variable,
    VectorField,
    particlefile_to_v3_zarr,
)
from parcels._core.index_search import _search_time_index
from parcels._core.mesh import get_mesh
from parcels._datasets.structured.generated import simple_UV_dataset
from parcels.interpolators import (
    CGrid_Velocity,
    XFreeslip,
    XLinear,
    XLinearInvdistLandTracer,
    XNearest,
    XPartialslip,
)
from parcels.interpolators._base import VectorInterpolator
from parcels.interpolators._xinterpolators import _get_corner_data_Agrid
from parcels.kernels import AdvectionRK4_3D
from tests.utils import TEST_DATA


@pytest.fixture
def field():
    """Reference data used for testing interpolation."""
    z0 = np.array(  # each x is +1 from the previous, each y is +2 from the previous
        [
            [0.0, 1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0, 7.0],
            [6.0, 7.0, 8.0, 9.0],
        ]
    )
    spatial_data = np.array([z0, z0 + 3, z0 + 6, z0 + 9])  # each z is +3 from the previous
    temporal_data = np.array([spatial_data, spatial_data + 10, spatial_data + 20])  # each t is +10 from the previous

    ds = xr.Dataset(
        {"P": (["time", "depth", "lat", "lon"], temporal_data)},
        coords={
            "time": (["time"], [np.timedelta64(t, "s") for t in [0, 2, 4]], {"axis": "T"}),
            "depth": (["depth"], [0, 1, 2, 3], {"axis": "Z"}),
            "lat": (["lat"], [0, 1, 2, 3], {"axis": "Y"}),
            "lon": (["lon"], [0, 1, 2, 3], {"axis": "X"}),
            "x": (["x"], [0.5, 1.5, 2.5, 3.5], {"axis": "X"}),
            "y": (["y"], [0.5, 1.5, 2.5, 3.5], {"axis": "Y"}),
        },
    ).pipe(
        sgrid._attach_sgrid_metadata,
        sgrid.SGrid2DMetadata(
            cf_role="grid_topology",
            topology_dimension=2,
            node_dimensions=("lon", "lat"),
            face_dimensions=(
                sgrid.FaceNodePadding("x", "lon", sgrid.Padding.LOW),
                sgrid.FaceNodePadding("y", "lat", sgrid.Padding.LOW),
            ),
            node_coordinates=("lon", "lat"),
            vertical_dimensions=(sgrid.FaceNodePadding("ZC", "depth", sgrid.Padding.HIGH),),
        ),
    )
    field = FieldSet.from_sgrid_conventions(ds, mesh="flat").P
    assert isinstance(field.interp_method, XLinear)

    return field


@pytest.mark.parametrize(
    "interpolator, t, z, y, x, expected",
    [
        pytest.param(
            XLinear(),
            [0, 1],
            [0, 0],
            [0.49, 0.49],
            [0.51, 0.51],
            [1.49, 6.49],
            id="Linear-1",
        ),
        pytest.param(XLinear(), 1, 2.5, 0.49, 0.51, 13.99, id="Linear-2"),
        pytest.param(
            XLinear(),
            [0, 1, 1],
            [0, 0, 2.5],
            [0.49, 0.49, 0.49],
            [0.51, 0.51, 0.51],
            [1.49, 6.49, 13.99],
            id="Linear-3",
        ),
        pytest.param(XLinearInvdistLandTracer(), 1, 2.5, 0.49, 0.51, 13.99, id="LinearInvDistLand"),
        pytest.param(
            XNearest(),
            [0, 3],
            [0.2, 0.2],
            [0.2, 0.2],
            [0.51, 0.51],
            [1.0, 16.0],
            id="Nearest",
        ),
    ],
)
def test_raw_2d_interpolation(field, interpolator, t, z, y, x, expected):
    """Test the interpolation functions on the Field."""
    particle_positions = {"time": t, "z": z, "lat": y, "lon": x}
    grid_positions = field.grid.search(z, y, x)
    grid_positions.update(_search_time_index(field, t))

    value = interpolator.interp(particle_positions, grid_positions, field)
    np.testing.assert_equal(value, expected)


@pytest.mark.parametrize("mesh", ["flat", "spherical"])
@pytest.mark.parametrize(
    "func, t, z, y, x, expected",
    [
        (XPartialslip(), 1, 0, 0, 0.0, [[1.0], [1.0]]),
        (XFreeslip(), 1, 0, 0.5, 1.5, [[1.0], [0.5]]),
        (XPartialslip(), 1, 0, 2.5, 1.5, [[0.75], [0.5]]),
        (XFreeslip(), 1, 0, 2.5, 1.5, [[1.0], [0.5]]),
        (XPartialslip(), 1, 0, 1.5, 0.5, [[0.5], [0.75]]),
        (XFreeslip(), 1, 0, 1.5, 0.5, [[0.5], [1.0]]),
        (
            XFreeslip(),
            [1, 0],
            [0, 2],
            [1.5, 1.5],
            [2.5, 0.5],
            [[0.5, 0.5], [1.0, 1.0]],
        ),
    ],
)
def test_spatial_slip_interpolation(field, func, t, z, y, x, expected, mesh):
    field.grid._mesh = get_mesh(mesh)
    field.data[:] = 1.0
    field.data[:, :, 1:3, 1:3] = 0.0  # Set zero land value to test spatial slip
    U = field
    V = field
    UV = VectorField("UV", U, V, interp_method=func)

    velocities = UV[t, z, y, x]
    expected = np.array(expected)
    if mesh == "spherical":
        expected[0] = expected[0] / (1852 * 60.0 * np.cos(np.radians(y)))
        expected[1] = expected[1] / (1852 * 60.0)
    np.testing.assert_array_almost_equal(velocities, expected)


@pytest.mark.parametrize(
    "interpolator, t, z, y, x, expected",
    [
        (XLinearInvdistLandTracer(), 1, 0, 0.5, 0.5, 1.0),
        (XLinearInvdistLandTracer(), 1, 0, 1.5, 1.5, 0.0),
        (
            XLinearInvdistLandTracer(),
            [0, 1],
            [0, 2],
            [0.5, 0.5],
            [0.5, 0.5],
            1.0,
        ),
        (
            XLinearInvdistLandTracer(),
            [0, 1],
            [0, 2],
            [0.5, 1.5],
            [0.5, 1.5],
            [1.0, 0.0],
        ),
    ],
)
def test_invdistland_interpolation(field, interpolator, t, z, y, x, expected):
    field.data[:] = 1.0
    field.data[:, :, 1:3, 1:3] = 0  # Set NaN land value to test inv_dist
    field.interp_method = interpolator

    value = field[t, z, y, x]
    np.testing.assert_array_almost_equal(value, expected)


@pytest.mark.parametrize("mesh", ["spherical", "flat"])
def test_interpolation_mesh_type(mesh, npart=10):
    ds = simple_UV_dataset(mesh=mesh)
    ds["U"].data[:] = 1.0
    fieldset = FieldSet.from_sgrid_conventions(ds, mesh=mesh)

    lat = 30.0
    time = 0.0
    u_expected = 1.0 if mesh == "flat" else 1.0 / (1852 * 60 * np.cos(np.radians(lat)))

    with pytest.warns(RuntimeWarning, match="Sampling of velocities should normally be done"):
        assert fieldset.U.eval(time, 0, lat, 0) == 1.0
        assert fieldset.V[time, 0, lat, 0] == 0.0

    u, v = fieldset.UV[time, 0, lat, 0]
    assert np.isclose(u, u_expected, atol=1e-7)
    assert v == 0.0


@pytest.fixture
def corner_gather_data() -> xr.DataArray:
    rng = np.random.default_rng(0)
    return xr.DataArray(rng.random((6, 5, 7, 8)), dims=("time", "depth", "lat", "lon"))


CORNER_GATHER_AXIS_DIM = {"X": "lon", "Y": "lat", "Z": "depth"}


def _agrid_corners(data, ti, zi, yi, xi, lenT, lenZ, npart):  # noqa: N803
    return _get_corner_data_Agrid(data, ti, zi, yi, xi, lenT, lenZ, npart, CORNER_GATHER_AXIS_DIM)


@pytest.mark.parametrize("lenT", [1, 2])
@pytest.mark.parametrize("lenZ", [1, 2])
@pytest.mark.parametrize("uniform_clock", [True, False])
def test_corner_gather_batch_matches_single_particle(corner_gather_data, lenT, lenZ, uniform_clock):  # noqa: N803
    """Gathering a batch must give each particle what it would get on its own.

    Interpolation is per-particle, so batching cannot change a result. The index
    arrays and the final reshape must therefore agree on where each particle sits
    in the flat gather.
    """
    rng = np.random.default_rng(1)
    npart = 5
    if uniform_clock:
        ti, zi = np.full(npart, 2), np.full(npart, 1)
    else:
        ti, zi = rng.integers(0, 4, npart), rng.integers(0, 3, npart)
    yi, xi = rng.integers(0, 5, npart), rng.integers(0, 6, npart)

    batch = _agrid_corners(corner_gather_data, ti, zi, yi, xi, lenT, lenZ, npart)

    for p in range(npart):
        single = _agrid_corners(
            corner_gather_data, ti[p : p + 1], zi[p : p + 1], yi[p : p + 1], xi[p : p + 1], lenT, lenZ, 1
        )
        np.testing.assert_array_equal(batch[..., p], single[..., 0])


def test_corner_gather_axes_are_ordered_t_z_y_x(corner_gather_data):
    """The returned axes must be (T, Z, Y, X, particle), as the interpolators assume."""
    rng = np.random.default_rng(2)
    npart = 4
    ti, zi = rng.integers(0, 4, npart), rng.integers(0, 3, npart)
    yi, xi = rng.integers(0, 5, npart), rng.integers(0, 6, npart)

    out = _agrid_corners(corner_gather_data, ti, zi, yi, xi, 2, 2, npart)
    raw = corner_gather_data.values

    assert out.shape == (2, 2, 2, 2, npart)
    for p in range(npart):
        for it, iz, iy, ix in np.ndindex(2, 2, 2, 2):
            assert out[it, iz, iy, ix, p] == raw[ti[p] + it, zi[p] + iz, yi[p] + iy, xi[p] + ix]


def test_corner_gather_keeps_axes_missing_from_the_mapping():
    """An axis absent from ``axis_dim`` is not indexed, but still shapes the result."""
    rng = np.random.default_rng(3)
    data = xr.DataArray(rng.random((6, 1, 7, 8)), dims=("time", "depth", "lat", "lon"))
    npart = 3
    ti = np.array([0, 2, 3])
    zi = np.zeros(npart, dtype=int)
    yi, xi = rng.integers(0, 5, npart), rng.integers(0, 6, npart)

    out = _get_corner_data_Agrid(data, ti, zi, yi, xi, 2, 1, npart, {"X": "lon", "Y": "lat"})

    assert out.shape == (2, 1, 2, 2, npart)
    for p in range(npart):
        assert out[0, 0, 0, 0, p] == data.values[ti[p], 0, yi[p], xi[p]]


class XNearest_Velocity(VectorInterpolator):  # noqa:  N801
    """Nearest-Neighbour interpolation on a regular grid for VectorFields of velocity."""

    def interp(
        self,
        particle_positions: dict[str, float | np.ndarray],
        grid_positions,
        vectorfield: VectorField,
    ):
        """Nearest-Neighbour interpolation on a regular grid for VectorFields of velocity."""
        _xnearest = XNearest()
        u = _xnearest.interp(particle_positions, grid_positions, vectorfield.U)
        v = _xnearest.interp(particle_positions, grid_positions, vectorfield.V)
        w = _xnearest.interp(particle_positions, grid_positions, vectorfield.W)
        return u, v, w


@pytest.mark.parametrize(
    ("interp_name", "interp_method"),
    [
        ("linear", XLinear),
        ("freeslip", XFreeslip),
        ("nearest", XNearest_Velocity),
        ("cgrid_velocity", CGrid_Velocity),
    ],
)
def test_interp_regression_v3(interp_name, interp_method, tmp_zarr, tmp_parquet):
    """Test that the v4 versions of the interpolation are the same as the v3 versions."""
    ds_input = xr.open_dataset(str(TEST_DATA / f"test_interpolation_data_random_{interp_name}.nc"))
    ydim = ds_input["U"].shape[2]
    xdim = ds_input["U"].shape[3]
    time = [np.timedelta64(int(t), "s") for t in ds_input["time"].values]

    # Convert the coordinates to float32 to match v3 behavior. This makes a difference for Cgrid velocity interpolation
    for dim in ["lon", "lat", "depth"]:
        ds_input[dim] = ds_input[dim].astype(np.float32)

    ds = xr.Dataset(
        {
            "U": (["time", "depth", "YG", "XG"], ds_input["U"].values),
            "V": (["time", "depth", "YG", "XG"], ds_input["V"].values),
            "W": (["time", "depth", "YG", "XG"], ds_input["W"].values),
        },
        coords={
            "time": (["time"], time, {"axis": "T"}),
            "depth": (["depth"], ds_input["depth"].values, {"axis": "Z"}),
            "YC": (["YC"], np.arange(ydim) + 0.5, {"axis": "Y"}),
            "YG": (["YG"], np.arange(ydim), {"axis": "Y"}),
            "XC": (["XC"], np.arange(xdim) + 0.5, {"axis": "X"}),
            "XG": (["XG"], np.arange(xdim), {"axis": "X"}),
            "lat": (["YG"], ds_input["lat"].values, {"axis": "Y"}),
            "lon": (["XG"], ds_input["lon"].values, {"axis": "X"}),
        },
    ).pipe(
        sgrid._attach_sgrid_metadata,
        sgrid.SGrid2DMetadata(
            cf_role="grid_topology",
            topology_dimension=2,
            node_dimensions=("XG", "YG"),
            face_dimensions=(
                sgrid.FaceNodePadding("XC", "XG", sgrid.Padding.LOW),
                sgrid.FaceNodePadding("YC", "YG", sgrid.Padding.LOW),
            ),
            node_coordinates=("lon", "lat"),
            vertical_dimensions=(sgrid.FaceNodePadding("ZC", "depth", sgrid.Padding.HIGH),),
        ),
    )

    fieldset = FieldSet.from_sgrid_conventions(ds, mesh="flat")
    if interp_name in ["cgrid_velocity", "freeslip", "nearest"]:
        fieldset.UVW.interp_method = interp_method()

    x, y, z = np.meshgrid(np.linspace(0, 1, 7), np.linspace(0, 1, 13), np.linspace(0, 1, 5))

    TestP = Particle.add_variable(Variable("pid", dtype=np.int32, initial=0))
    pset = ParticleSet(fieldset, pclass=TestP, x=x, y=y, z=z, pid=np.arange(x.size))

    def DeleteParticle(particles, fieldset):
        any_error = particles.state >= 50  # This captures all Errors
        particles[any_error].state = StatusCode.Delete

    outfile = ParticleFile(tmp_parquet, outputdt=np.timedelta64(1, "s"), mode="w")
    pset.execute(
        [AdvectionRK4_3D, DeleteParticle],
        runtime=np.timedelta64(4, "s"),
        dt=np.timedelta64(1, "s"),
        output_file=outfile,
    )
    particlefile_to_v3_zarr(tmp_parquet, tmp_zarr)
    ds_v4 = xr.open_zarr(tmp_zarr)

    ds_v3 = xr.open_zarr(str(TEST_DATA / f"test_interpolation_jit_{interp_name}.zarr"))
    # v3 zarr is not sorted by particle_id, so we sort it here to match the v4 output
    ds_v3 = ds_v3.sortby("trajectory")

    # v4 also writes last timestep, so has one more observation than v3. We ignore the last timestep to match v3.
    np.testing.assert_allclose(ds_v3["lon"].values, ds_v4["lon"].values[:, :-1], atol=1e-6, equal_nan=True)
    np.testing.assert_allclose(ds_v3["lat"].values, ds_v4["lat"].values[:, :-1], atol=1e-6, equal_nan=True)
    np.testing.assert_allclose(ds_v3["z"].values, ds_v4["z"].values[:, :-1], atol=1e-6, equal_nan=True)
