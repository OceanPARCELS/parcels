import math

import numpy as np
import xarray as xr

import parcels._sgrid as sgrid
from parcels._core.utils.time import timedelta_to_float


def simple_UV_dataset(n_times=360, nz=2, ny=30, nx=4, maxdepth=1, mesh="spherical"):
    max_lon = 180.0 if mesh == "spherical" else 1e6
    max_lat = 90.0 if mesh == "spherical" else 1e6
    shape = (n_times, nz, ny, nx)

    return xr.Dataset(
        {"U": (["time", "depth", "YG", "XG"], np.zeros(shape)), "V": (["time", "depth", "YG", "XG"], np.zeros(shape))},
        coords={
            "time": (["time"], xr.date_range("2000", "2001", n_times), {"axis": "T"}),
            "depth": (["depth"], np.linspace(0, maxdepth, nz), {"axis": "Z"}),
            "YC": (["YC"], np.arange(ny) + 0.5, {"axis": "Y"}),
            "YG": (["YG"], np.arange(ny), {"axis": "Y"}),
            "XC": (["XC"], np.arange(nx) + 0.5, {"axis": "X"}),
            "XG": (["XG"], np.arange(nx), {"axis": "X"}),
            "lat": (["YG"], np.linspace(-max_lat, max_lat, ny), {"axis": "Y"}),
            "lon": (["XG"], np.linspace(-max_lon, max_lon, nx), {"axis": "X"}),
        },
    ).pipe(
        sgrid._attach_sgrid_metadata,
        sgrid.SGrid2DMetadata(
            cf_role="grid_topology",
            topology_dimension=2,
            node_dimensions=("XG", "YG"),
            node_coordinates=("lon", "lat"),
            face_dimensions=(
                sgrid.FaceNodePadding("XC", "XG", sgrid.Padding.LOW),
                sgrid.FaceNodePadding("YC", "YG", sgrid.Padding.LOW),
            ),
            vertical_dimensions=(sgrid.FaceNodePadding("ZC", "depth", sgrid.Padding.BOTH),),
        ),
    )


def simple_UV_sigma_dataset(
    n_times=360, nz=2, ny=30, nx=4, bottom_depth=None, eta=None, mesh="flat", start_time="2000-01-01"
):
    """simple_UV_dataset with zero U, V and W and a time-varying (4D) depth on the cell centres: nz sigma interfaces from the surface (depth -eta) to bottom_depth ((YC, XC) array, meters; None for a flat bottom at 1 m); eta is None or (time, YC, XC), hourly from start_time."""
    if bottom_depth is None:
        bottom_depth = np.ones((ny, nx))
    if np.shape(bottom_depth) != (ny, nx):
        raise ValueError(
            f"bottom_depth must be a (YC, XC) array of shape {(ny, nx)}, got shape {np.shape(bottom_depth)}"
        )
    if eta is None:
        eta = np.zeros((n_times, ny, nx))
    elif np.shape(eta) != (n_times, ny, nx):
        raise ValueError(f"eta must be a (time, YC, XC) array of shape {(n_times, ny, nx)}, got shape {np.shape(eta)}")

    sigma = np.linspace(0.0, 1.0, nz)
    total_depth = bottom_depth[np.newaxis, :, :] + eta
    zf = (
        sigma[np.newaxis, :, np.newaxis, np.newaxis] * total_depth[:, np.newaxis] - eta[:, np.newaxis]
    )  # (time, depth, YC, XC)

    ds = simple_UV_dataset(n_times=n_times, nz=nz, ny=ny, nx=nx, mesh=mesh)
    ds["W"] = (["time", "depth", "YG", "XG"], np.zeros((n_times, nz, ny, nx)))
    return ds.assign_coords(
        time=(["time"], xr.date_range(start_time, periods=n_times, freq="1h"), {"axis": "T"}),
        depth=(["time", "depth", "YC", "XC"], zf, {"axis": "Z"}),
    )


def radial_rotation_dataset(xdim=200, ydim=200):  # Define 2D flat, square fieldset for testing purposes.
    lon = np.linspace(0, 60, xdim, dtype=np.float32)
    lat = np.linspace(0, 60, ydim, dtype=np.float32)

    x0 = 30.0  # Define the origin to be the centre of the Field.
    y0 = 30.0

    U = np.zeros((2, 1, ydim, xdim), dtype=np.float32)
    V = np.zeros((2, 1, ydim, xdim), dtype=np.float32)

    omega = 2 * np.pi / 86400.0  # Define the rotational period as 1 day.

    for i in range(lon.size):
        for j in range(lat.size):
            r = np.sqrt((lon[i] - x0) ** 2 + (lat[j] - y0) ** 2)
            assert r >= 0.0
            assert r <= np.sqrt(x0**2 + y0**2)

            theta = np.arctan2((lat[j] - y0), (lon[i] - x0))
            assert abs(theta) <= np.pi

            U[:, :, j, i] = r * np.sin(theta) * omega
            V[:, :, j, i] = -r * np.cos(theta) * omega

    return xr.Dataset(
        {"U": (["time", "depth", "YG", "XG"], U), "V": (["time", "depth", "YG", "XG"], V)},
        coords={
            "time": (["time"], [np.timedelta64(0, "s"), np.timedelta64(10, "D")], {"axis": "T"}),
            "depth": (["depth"], np.array([0.0]), {"axis": "Z"}),
            "YC": (["YC"], np.arange(ydim) + 0.5, {"axis": "Y"}),
            "YG": (["YG"], np.arange(ydim), {"axis": "Y"}),
            "XC": (["XC"], np.arange(xdim) + 0.5, {"axis": "X"}),
            "XG": (["XG"], np.arange(xdim), {"axis": "X"}),
            "lat": (["YG"], lat, {"axis": "Y"}),
            "lon": (["XG"], lon, {"axis": "X"}),
        },
    ).pipe(
        sgrid._attach_sgrid_metadata,
        sgrid.SGrid2DMetadata(
            cf_role="grid_topology",
            topology_dimension=2,
            node_dimensions=("XG", "YG"),
            node_coordinates=("lon", "lat"),
            face_dimensions=(
                sgrid.FaceNodePadding("XC", "XG", sgrid.Padding.LOW),
                sgrid.FaceNodePadding("YC", "YG", sgrid.Padding.HIGH),
            ),
            vertical_dimensions=(sgrid.FaceNodePadding("ZC", "depth", sgrid.Padding.BOTH),),
        ),
    )


def moving_eddy_dataset(xdim=2, ydim=2):
    """Create a dataset with an eddy moving in time. Note that there is no spatial variation in the flow."""
    f, u_0, u_g = 1.0e-4, 0.3, 0.04  # Some constants

    lon = np.linspace(0, 25000, xdim, dtype=np.float32)
    lat = np.linspace(0, 25000, ydim, dtype=np.float32)

    time = np.arange(np.timedelta64(0, "s"), np.timedelta64(7, "h"), np.timedelta64(1, "m"))

    U = np.zeros((len(time), 1, ydim, xdim), dtype=np.float32)
    V = np.zeros((len(time), 1, ydim, xdim), dtype=np.float32)

    for t in range(len(time)):
        U[t, :, :, :] = u_g + (u_0 - u_g) * np.cos(f * timedelta_to_float(time[t]))
        V[t, :, :, :] = -(u_0 - u_g) * np.sin(f * timedelta_to_float(time[t]))

    return xr.Dataset(
        {"U": (["time", "depth", "YG", "XG"], U), "V": (["time", "depth", "YG", "XG"], V)},
        coords={
            "time": (["time"], time, {"axis": "T"}),
            "depth": (["depth"], np.array([0.0]), {"axis": "Z"}),
            "YC": (["YC"], np.arange(ydim) + 0.5, {"axis": "Y"}),
            "YG": (["YG"], np.arange(ydim), {"axis": "Y"}),
            "XC": (["XC"], np.arange(xdim) + 0.5, {"axis": "X"}),
            "XG": (["XG"], np.arange(xdim), {"axis": "X"}),
            "lat": (["YG"], lat, {"axis": "Y"}),
            "lon": (["XG"], lon, {"axis": "X"}),
        },
        attrs={
            "u_0": u_0,
            "u_g": u_g,
            "f": f,
        },
    ).pipe(
        sgrid._attach_sgrid_metadata,
        sgrid.SGrid2DMetadata(
            cf_role="grid_topology",
            topology_dimension=2,
            node_dimensions=("XG", "YG"),
            node_coordinates=("lon", "lat"),
            face_dimensions=(
                sgrid.FaceNodePadding("XC", "XG", sgrid.Padding.LOW),
                sgrid.FaceNodePadding("YC", "YG", sgrid.Padding.HIGH),
            ),
            vertical_dimensions=(sgrid.FaceNodePadding("ZC", "depth", sgrid.Padding.BOTH),),
        ),
    )


def decaying_moving_eddy_dataset(xdim=2, ydim=2):
    """Simulate an ocean that accelerates subject to Coriolis force
    and dissipative effects, upon which a geostrophic current is
    superimposed.

    The original test description can be found in: N. Fabbroni, 2009,
    Numerical Simulation of Passive tracers dispersion in the sea,
    Ph.D. dissertation, University of Bologna
    http://amsdottorato.unibo.it/1733/1/Fabbroni_Nicoletta_Tesi.pdf
    """
    u_g = 0.04  # Geostrophic current
    u_0 = 0.3  # Initial speed in x dirrection. v_0 = 0
    gamma = 1.0 / (2.89 * 86400)  # Dissipitave effects due to viscousity.
    gamma_g = 1.0 / (28.9 * 86400)
    f = 1.0e-4  # Coriolis parameter.

    time = np.arange(np.timedelta64(0, "s"), np.timedelta64(1, "D") + np.timedelta64(1, "h"), np.timedelta64(2, "m"))
    lon = np.linspace(0, 20000, xdim, dtype=np.float32)
    lat = np.linspace(5000, 12000, ydim, dtype=np.float32)

    U = np.zeros((time.size, 1, lat.size, lon.size), dtype=np.float32)
    V = np.zeros((time.size, 1, lat.size, lon.size), dtype=np.float32)

    for t in range(time.size):
        t_float = timedelta_to_float(time[t])
        U[t, :, :, :] = u_g * np.exp(-gamma_g * t_float) + (u_0 - u_g) * np.exp(-gamma * t_float) * np.cos(f * t_float)
        V[t, :, :, :] = -(u_0 - u_g) * np.exp(-gamma * t_float) * np.sin(f * t_float)

    return xr.Dataset(
        {"U": (["time", "depth", "YG", "XG"], U), "V": (["time", "depth", "YG", "XG"], V)},
        coords={
            "time": (["time"], time, {"axis": "T"}),
            "depth": (["depth"], np.array([0.0]), {"axis": "Z"}),
            "YC": (["YC"], np.arange(ydim) + 0.5, {"axis": "Y"}),
            "YG": (["YG"], np.arange(ydim), {"axis": "Y"}),
            "XC": (["XC"], np.arange(xdim) + 0.5, {"axis": "X"}),
            "XG": (["XG"], np.arange(xdim), {"axis": "X"}),
            "lat": (["YG"], lat, {"axis": "Y"}),
            "lon": (["XG"], lon, {"axis": "X"}),
        },
        attrs={
            "u_0": u_0,
            "u_g": u_g,
            "f": f,
            "gamma": gamma,
            "gamma_g": gamma_g,
        },
    ).pipe(
        sgrid._attach_sgrid_metadata,
        sgrid.SGrid2DMetadata(
            cf_role="grid_topology",
            topology_dimension=2,
            node_dimensions=("XG", "YG"),
            node_coordinates=("lon", "lat"),
            face_dimensions=(
                sgrid.FaceNodePadding("XC", "XG", sgrid.Padding.LOW),
                sgrid.FaceNodePadding("YC", "YG", sgrid.Padding.HIGH),
            ),
            vertical_dimensions=(sgrid.FaceNodePadding("ZC", "depth", sgrid.Padding.BOTH),),
        ),
    )


def peninsula_dataset(xdim=100, ydim=50, mesh="flat", grid_type="A"):
    """Construct a fieldset encapsulating the flow field around an idealised peninsula.

    Parameters
    ----------
    xdim :
        Horizontal dimension of the generated fieldset
    ydim :
        Vertical dimension of the generated fieldset
    mesh : str
        String indicating the type of mesh coordinates and
        units used during velocity interpolation:

        1. spherical: Lat and lon in degree, with a
           correction for zonal velocity U near the poles.
        2. flat (default): No conversion, lat/lon are assumed to be in m.
    grid_type :
        Option whether grid is either Arakawa A (default) or C

        The original test description can be found in Fig. 2.2.3 in:
        North, E. W., Gallego, A., Petitgas, P. (Eds). 2009. Manual of
        recommended practices for modelling physical - biological
        interactions during fish early life.
        ICES Cooperative Research Report No. 295. 111 pp.
        http://archimer.ifremer.fr/doc/00157/26792/24888.pdf
    """
    domainsizeX, domainsizeY = (1.0e5, 5.0e4)
    La = np.linspace(0, domainsizeX, xdim, dtype=np.float32)
    Wa = np.linspace(0, domainsizeY, ydim, dtype=np.float32)

    u0 = 1
    x0 = domainsizeX / 2
    R = 0.32 * domainsizeX / 2

    # Create the fields
    P = np.zeros((ydim, xdim), dtype=np.float32)
    U = np.zeros_like(P)
    V = np.zeros_like(P)
    x, y = np.meshgrid(La, Wa, sparse=True, indexing="xy")
    P[:, :] = u0 * R**2 * y / ((x - x0) ** 2 + y**2) - u0 * y

    # Set land points to zero
    landpoints = P >= 0.0
    P[landpoints] = 0.0

    if grid_type == "A":
        U[:, :] = u0 - u0 * R**2 * ((x - x0) ** 2 - y**2) / (((x - x0) ** 2 + y**2) ** 2)
        V[:, :] = -2 * u0 * R**2 * ((x - x0) * y) / (((x - x0) ** 2 + y**2) ** 2)
        U[landpoints] = 0.0
        V[landpoints] = 0.0
        Udims = ["YC", "XC"]
        Vdims = ["YC", "XC"]
    elif grid_type == "C":
        U = np.zeros(P.shape)
        V = np.zeros(P.shape)
        U[1:, :] = -(P[1:, :] - P[:-1, :]) / (Wa[1] - Wa[0])
        V[:, 1:] = (P[:, 1:] - P[:, :-1]) / (La[1] - La[0])
        Udims = ["YG", "XC"]
        Vdims = ["YC", "XG"]
    else:
        raise ValueError(f"Grid_type {grid_type} is not a valid option")

    # Convert from m to lat/lon for spherical meshes
    lon = La / 1852.0 / 60.0 if mesh == "spherical" else La
    lat = Wa / 1852.0 / 60.0 if mesh == "spherical" else Wa

    return xr.Dataset(
        {
            "U": (Udims, U),
            "V": (Vdims, V),
            "P": (["YC", "XC"], P),
        },
        coords={
            "YC": (["YC"], np.arange(ydim) - 0.5, {"axis": "Y"}),
            "YG": (["YG"], np.arange(ydim), {"axis": "Y"}),
            "XC": (["XC"], np.arange(xdim) - 0.5, {"axis": "X"}),
            "XG": (["XG"], np.arange(xdim), {"axis": "X"}),
            "lat": (["YG"], lat, {"axis": "Y"}),
            "lon": (["XG"], lon, {"axis": "X"}),
        },
    ).pipe(
        sgrid._attach_sgrid_metadata,
        sgrid.SGrid2DMetadata(
            cf_role="grid_topology",
            topology_dimension=2,
            node_dimensions=("XG", "YG"),
            node_coordinates=("lon", "lat"),
            face_dimensions=(
                sgrid.FaceNodePadding("XC", "XG", sgrid.Padding.LOW),
                sgrid.FaceNodePadding("YC", "YG", sgrid.Padding.LOW),
            ),
        ),
    )


def stommel_gyre_dataset(xdim=200, ydim=200, grid_type="A"):
    """Simulate a periodic current along a western boundary, with significantly
    larger velocities along the western edge than the rest of the region

    The original test description can be found in: N. Fabbroni, 2009,
    Numerical Simulation of Passive tracers dispersion in the sea,
    Ph.D. dissertation, University of Bologna
    http://amsdottorato.unibo.it/1733/1/Fabbroni_Nicoletta_Tesi.pdf
    """
    a = b = 10000 * 1e3
    scalefac = 0.05  # to scale for physically meaningful velocities
    dx, dy = a / xdim, b / ydim

    # Coordinates of the test fieldset (on A-grid in deg)
    lon = np.linspace(0, a, xdim, dtype=np.float32)
    lat = np.linspace(0, b, ydim, dtype=np.float32)

    # Define arrays U (zonal), V (meridional) and P (sea surface height)
    U = np.zeros((lat.size, lon.size), dtype=np.float32)
    V = np.zeros((lat.size, lon.size), dtype=np.float32)
    P = np.zeros((lat.size, lon.size), dtype=np.float32)

    beta = 2e-11
    r = 1 / (11.6 * 86400)
    es = r / (beta * a)

    for j in range(lat.size):
        for i in range(lon.size):
            xi = lon[i] / a
            yi = lat[j] / b
            P[j, i] = (1 - math.exp(-xi / es) - xi) * math.pi * np.sin(math.pi * yi) * scalefac
            if grid_type == "A":
                U[j, i] = -(1 - math.exp(-xi / es) - xi) * math.pi**2 * np.cos(math.pi * yi) * scalefac
                V[j, i] = (math.exp(-xi / es) / es - 1) * math.pi * np.sin(math.pi * yi) * scalefac
    if grid_type == "C":
        U[1:, :] = -(P[1:, :] - P[0:-1, :]) / dy * b
        V[:, 1:] = (P[:, 1:] - P[:, 0:-1]) / dx * a
        Udims = ["YG", "XC"]
        Vdims = ["YC", "XG"]
    else:
        Udims = ["YC", "XC"]
        Vdims = ["YC", "XC"]

    return xr.Dataset(
        {"U": (Udims, U), "V": (Vdims, V), "P": (["YG", "XG"], P)},
        coords={
            "YC": (["YC"], np.arange(ydim) - 0.5, {"axis": "Y"}),
            "YG": (["YG"], np.arange(ydim), {"axis": "Y"}),
            "XC": (["XC"], np.arange(xdim) - 0.5, {"axis": "X"}),
            "XG": (["XG"], np.arange(xdim), {"axis": "X"}),
            "lat": (["YG"], lat, {"axis": "Y"}),
            "lon": (["XG"], lon, {"axis": "X"}),
        },
    ).pipe(
        sgrid._attach_sgrid_metadata,
        sgrid.SGrid2DMetadata(
            cf_role="grid_topology",
            topology_dimension=2,
            node_dimensions=("XG", "YG"),
            node_coordinates=("lon", "lat"),
            face_dimensions=(
                sgrid.FaceNodePadding("XC", "XG", sgrid.Padding.LOW),
                sgrid.FaceNodePadding("YC", "YG", sgrid.Padding.LOW),
            ),
        ),
    )
