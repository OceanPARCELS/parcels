import math

import numpy as np
import uxarray as ux
import xarray as xr

T = 2
vmax = 1.0
delta = 0.1
TIME = xr.date_range("2000", "2001", T)


def simple_small_delaunay(nx=10, ny=10):
    """
    Data on a small Delaunay grid. The naming convention of the dataset and grid is consistent with what is
    provided by UXArray when reading in FESOM2 datasets.
    """
    lon, lat = np.meshgrid(np.linspace(0, 1.0, nx, dtype=np.float32), np.linspace(0, 1.0, ny, dtype=np.float32))
    lon_flat = lon.ravel()
    lat_flat = lat.ravel()
    zf = np.linspace(0.0, 1000.0, 2, endpoint=True, dtype=np.float32)  # Vertical element faces
    zc = 0.5 * (zf[:-1] + zf[1:])  # Vertical element centers

    # mask any point on one of the boundaries
    mask = np.isclose(lon_flat, 0.0) | np.isclose(lon_flat, 1.0) | np.isclose(lat_flat, 0.0) | np.isclose(lat_flat, 1.0)

    boundary_points = np.flatnonzero(mask)

    uxgrid = ux.Grid.from_points(
        (lon_flat, lat_flat),
        method="regional_delaunay",
        boundary_points=boundary_points,
    )
    uxgrid.attrs["Conventions"] = "UGRID-1.0"

    # Define arrays U (zonal), V (meridional), W (vertical), and P (sea surface height)
    U = np.zeros((1, zc.size, uxgrid.n_face), dtype=np.float64)
    V = np.zeros((1, zc.size, uxgrid.n_face), dtype=np.float64)
    W = np.zeros((1, zf.size, uxgrid.n_node), dtype=np.float64)
    P = np.zeros((1, zc.size, uxgrid.n_face), dtype=np.float64)
    # Define Tface, a ficticious tracer field on the face centroids
    Tface = np.zeros((1, zc.size, uxgrid.n_face), dtype=np.float64)

    for i, (x, y) in enumerate(zip(uxgrid.face_lon, uxgrid.face_lat, strict=False)):
        P[0, :, i] = -vmax * delta * (1 - x) * (math.exp(-x / delta) - 1) * np.sin(math.pi * y)
        U[0, :, i] = -vmax * (1 - math.exp(-x / delta) - x) * np.cos(math.pi * y)
        V[0, :, i] = vmax * ((2.0 - x) * math.exp(-x / delta) - 1) * np.sin(math.pi * y)
        Tface[0, :, i] = np.sin(math.pi * y) * np.cos(math.pi * x)

    # Define Tnode, the same ficticious tracer field as above but on the face corner vertices
    Tnode = np.zeros((1, zc.size, uxgrid.n_node), dtype=np.float64)
    for i, (x, y) in enumerate(zip(uxgrid.node_lon, uxgrid.node_lat, strict=False)):
        Tnode[0, :, i] = np.sin(math.pi * y) * np.cos(math.pi * x)

    u = ux.UxDataArray(
        data=U,
        name="U",
        uxgrid=uxgrid,
        dims=["time", "zc", "n_face"],
        coords=dict(
            time=(["time"], [TIME[0]]),
            zc=(["zc"], zc),
        ),
        attrs=dict(
            description="zonal velocity", units="m/s", location="face", mesh="delaunay", Conventions="UGRID-1.0"
        ),
    )
    v = ux.UxDataArray(
        data=V,
        name="V",
        uxgrid=uxgrid,
        dims=["time", "zc", "n_face"],
        coords=dict(
            time=(["time"], [TIME[0]]),
            zc=(["zc"], zc),
        ),
        attrs=dict(
            description="meridional velocity", units="m/s", location="face", mesh="delaunay", Conventions="UGRID-1.0"
        ),
    )
    w = ux.UxDataArray(
        data=W,
        name="W",
        uxgrid=uxgrid,
        dims=["time", "zf", "n_node"],
        coords=dict(
            time=(["time"], [TIME[0]]),
            zf=(["zf"], zf),
        ),
        attrs=dict(
            description="meridional velocity", units="m/s", location="node", mesh="delaunay", Conventions="UGRID-1.0"
        ),
    )
    p = ux.UxDataArray(
        data=P,
        name="p",
        uxgrid=uxgrid,
        dims=["time", "zc", "n_face"],
        coords=dict(
            time=(["time"], [TIME[0]]),
            zc=(["zc"], zc),
        ),
        attrs=dict(description="pressure", units="N/m^2", location="face", mesh="delaunay", Conventions="UGRID-1.0"),
    )

    tface = ux.UxDataArray(
        data=Tface,
        name="T_face",
        uxgrid=uxgrid,
        dims=["time", "zc", "n_face"],
        coords=dict(
            time=(["time"], [TIME[0]]),
            zc=(["zc"], zc),
        ),
        attrs=dict(
            description="Tracer field sampled on face centers",
            units="None",
            location="face",
            mesh="delaunay",
            Conventions="UGRID-1.0",
        ),
    )
    tnode = ux.UxDataArray(
        data=Tnode,
        name="T_node",
        uxgrid=uxgrid,
        dims=["time", "zc", "n_node"],
        coords=dict(
            time=(["time"], [TIME[0]]),
            zc=(["zc"], zc),
        ),
        attrs=dict(
            description="Tracer field sampled on face vertices",
            units="None",
            location="node",
            mesh="delaunay",
            Conventions="UGRID-1.0",
        ),
    )

    return ux.UxDataset({"U": u, "V": v, "W": w, "p": p, "T_face": tface, "T_node": tnode}, uxgrid=uxgrid)


def _build_delaunay_grid(nx, lon_range, lat_range):
    """Build a Delaunay-triangulated UxGrid from a regular nx-by-nx point lattice."""
    lon, lat = np.meshgrid(
        np.linspace(lon_range[0], lon_range[1], nx, dtype=np.float64),
        np.linspace(lat_range[0], lat_range[1], nx, dtype=np.float64),
    )
    lon_flat = lon.ravel()
    lat_flat = lat.ravel()
    mask = (
        np.isclose(lon_flat, lon_range[0])
        | np.isclose(lon_flat, lon_range[1])
        | np.isclose(lat_flat, lat_range[0])
        | np.isclose(lat_flat, lat_range[1])
    )
    boundary_points = np.flatnonzero(mask)
    uxgrid = ux.Grid.from_points(
        (lon_flat, lat_flat),
        method="regional_delaunay",
        boundary_points=boundary_points,
    )
    uxgrid.attrs["Conventions"] = "UGRID-1.0"
    return uxgrid


def _build_triangulated_lattice_grid(nx, x_range, y_range):
    """Build a flat UxGrid by splitting each cell of a regular nx-by-nx node lattice over x_range by y_range (in meters) into two triangles."""
    X, Y = np.meshgrid(np.linspace(x_range[0], x_range[1], nx), np.linspace(y_range[0], y_range[1], nx))
    node_x, node_y = X.ravel(), Y.ravel()
    i_grid, j_grid = np.meshgrid(np.arange(nx - 1), np.arange(nx - 1))
    i, j = i_grid.ravel(), j_grid.ravel()
    sw = (j * nx) + i
    se = sw + 1
    nw = sw + nx
    ne = nw + 1

    tri1 = np.stack((sw, ne, nw), axis=-1)
    tri2 = np.stack((sw, se, ne), axis=-1)
    face_node_connectivity = np.concatenate((tri1, tri2))

    uxgrid = ux.Grid.from_topology(node_lon=node_x, node_lat=node_y, face_node_connectivity=face_node_connectivity)
    uxgrid.node_lon.values[:] = node_x
    uxgrid.node_lat.values[:] = node_y
    uxgrid.attrs["Conventions"] = "UGRID-1.0"
    return uxgrid


def _wrap_uvw_dataset(uxgrid, u, v, w, zc, zf, time, uv_dim, uv_location, description):
    """Wrap (u, v, w) numpy arrays into a UxDataset following Parcels' UGRID conventions.

    u, v are placed on ``uv_dim`` (``"n_face"`` or ``"n_node"``) at vertical centres ``zc``.
    w is always on ``n_node`` at vertical interfaces ``zf``.
    """
    u_da = ux.UxDataArray(
        data=u,
        name="U",
        uxgrid=uxgrid,
        dims=["time", "zc", uv_dim],
        coords=dict(time=(["time"], time), zc=(["zc"], zc)),
        attrs=dict(
            description=f"zonal velocity ({description})",
            units="degrees/s",
            location=uv_location,
            mesh="delaunay",
            Conventions="UGRID-1.0",
        ),
    )
    v_da = ux.UxDataArray(
        data=v,
        name="V",
        uxgrid=uxgrid,
        dims=["time", "zc", uv_dim],
        coords=dict(time=(["time"], time), zc=(["zc"], zc)),
        attrs=dict(
            description=f"meridional velocity ({description})",
            units="degrees/s",
            location=uv_location,
            mesh="delaunay",
            Conventions="UGRID-1.0",
        ),
    )
    w_da = ux.UxDataArray(
        data=w,
        name="W",
        uxgrid=uxgrid,
        dims=["time", "zf", "n_node"],
        coords=dict(time=(["time"], time), zf=(["zf"], zf)),
        attrs=dict(
            description=f"vertical velocity ({description})",
            units="degrees/s",
            location="node",
            mesh="delaunay",
            Conventions="UGRID-1.0",
        ),
    )
    return ux.UxDataset({"U": u_da, "V": v_da, "W": w_da}, uxgrid=uxgrid)


def uniform_translation_face_centered(nx=20, u0=0.001, v0=0.0005):
    """T1-1 uniform translation, face-centered (u, v on ``n_face``).

    Verification field `u = u_0`,`v = v_0` on a Delaunay triangulation of a
    regular ``nx``-by-``nx`` point lattice over ``[0, 10] x [0, 10]``. Single vertical layer
    spans ``z in [0, 1]``. ``w`` is identically zero on the corner nodes.
    """
    uxgrid = _build_delaunay_grid(nx, (0.0, 10.0), (0.0, 10.0))

    zf = np.array([0.0, 1.0], dtype=np.float64)
    zc = np.array([0.5], dtype=np.float64)
    time = xr.date_range("2000-01-01", periods=1, freq="2h")

    u = np.full((1, zc.size, uxgrid.n_face), u0, dtype=np.float64)
    v = np.full((1, zc.size, uxgrid.n_face), v0, dtype=np.float64)
    w = np.zeros((1, zf.size, uxgrid.n_node), dtype=np.float64)

    return _wrap_uvw_dataset(uxgrid, u, v, w, zc, zf, time, "n_face", "face", "uniform translation")


def uniform_translation_node_centered(nx=20, u0=0.001, v0=0.0005):
    """T1-1 uniform translation, node-centered (u, v on ``n_node``).

    Same field as :func:`uniform_translation_face_centered` but with horizontal velocity
    sampled at corner nodes for use with barycentric (linear) interpolators.
    """
    uxgrid = _build_delaunay_grid(nx, (0.0, 10.0), (0.0, 10.0))

    zf = np.array([0.0, 1.0], dtype=np.float64)
    zc = np.array([0.5], dtype=np.float64)
    time = xr.date_range("2000-01-01", periods=1, freq="2h")

    u = np.full((1, zc.size, uxgrid.n_node), u0, dtype=np.float64)
    v = np.full((1, zc.size, uxgrid.n_node), v0, dtype=np.float64)
    w = np.zeros((1, zf.size, uxgrid.n_node), dtype=np.float64)

    return _wrap_uvw_dataset(uxgrid, u, v, w, zc, zf, time, "n_node", "node", "uniform translation")


def solid_body_rotation_face_centered(nx=40, omega=2.0 * math.pi / 3600.0):
    r"""T1-2 2D solid-body rotation, face-centered (u, v on ``n_face``).

    Verification field :math:`u = -\Omega y`, :math:`v = \Omega x` on a Delaunay
    triangulation of a regular ``nx``-by-``nx`` point lattice over ``[-5, 5] x [-5, 5]``
    centred on the rotation axis. Single vertical layer; ``w = 0``.
    """
    uxgrid = _build_delaunay_grid(nx, (-5.0, 5.0), (-5.0, 5.0))

    zf = np.array([0.0, 1.0], dtype=np.float64)
    zc = np.array([0.5], dtype=np.float64)
    time = xr.date_range("2000-01-01", periods=1, freq="2h")

    u_vals = -omega * uxgrid.face_lat.values
    v_vals = omega * uxgrid.face_lon.values
    u = np.broadcast_to(u_vals[np.newaxis, np.newaxis, :], (1, zc.size, uxgrid.n_face)).copy()
    v = np.broadcast_to(v_vals[np.newaxis, np.newaxis, :], (1, zc.size, uxgrid.n_face)).copy()
    w = np.zeros((1, zf.size, uxgrid.n_node), dtype=np.float64)

    return _wrap_uvw_dataset(uxgrid, u, v, w, zc, zf, time, "n_face", "face", "solid-body rotation")


def solid_body_rotation_node_centered(nx=40, omega=2.0 * math.pi / 3600.0):
    """T1-2 2D solid-body rotation, node-centered (u, v on ``n_node``).

    Same field as :func:`solid_body_rotation_face_centered` but sampled at corner
    nodes. Because the field is linear in space, barycentric interpolation reproduces it
    exactly and any error in particle trajectories is attributable to the time integrator.
    """
    uxgrid = _build_delaunay_grid(nx, (-5.0, 5.0), (-5.0, 5.0))

    zf = np.array([0.0, 1.0], dtype=np.float64)
    zc = np.array([0.5], dtype=np.float64)
    time = xr.date_range("2000-01-01", periods=1, freq="2h")

    u_vals = -omega * uxgrid.node_lat.values
    v_vals = omega * uxgrid.node_lon.values
    u = np.broadcast_to(u_vals[np.newaxis, np.newaxis, :], (1, zc.size, uxgrid.n_node)).copy()
    v = np.broadcast_to(v_vals[np.newaxis, np.newaxis, :], (1, zc.size, uxgrid.n_node)).copy()
    w = np.zeros((1, zf.size, uxgrid.n_node), dtype=np.float64)

    return _wrap_uvw_dataset(uxgrid, u, v, w, zc, zf, time, "n_node", "node", "solid-body rotation")


def solid_body_rotation_3d_face_centered(nx=40, nz=10, omega=2.0 * math.pi / 3600.0, w0=0.005):
    r"""T1-3 3D helical motion, face-centered horizontal velocity.

    Verification field :math:`u = -\Omega y`, :math:`v = \Omega x`, :math:`w = w_0` on a
    Delaunay triangulation of a regular ``nx``-by-``nx`` point lattice over
    ``[-5, 5] x [-5, 5]`` with ``nz`` vertical layers spanning ``[0, 100]``. ``u`` and ``v``
    are sampled at face centres; ``w`` is sampled at corner nodes on the layer interfaces.
    """
    uxgrid = _build_delaunay_grid(nx, (-5.0, 5.0), (-5.0, 5.0))

    zf = np.linspace(0.0, 100.0, nz + 1, dtype=np.float64)
    zc = 0.5 * (zf[:-1] + zf[1:])
    time = xr.date_range("2000-01-01", periods=1, freq="2h")

    u_vals = -omega * uxgrid.face_lat.values
    v_vals = omega * uxgrid.face_lon.values
    u = np.broadcast_to(u_vals[np.newaxis, np.newaxis, :], (1, zc.size, uxgrid.n_face)).copy()
    v = np.broadcast_to(v_vals[np.newaxis, np.newaxis, :], (1, zc.size, uxgrid.n_face)).copy()
    w = np.full((1, zf.size, uxgrid.n_node), w0, dtype=np.float64)

    return _wrap_uvw_dataset(uxgrid, u, v, w, zc, zf, time, "n_face", "face", "3D solid-body rotation")


def solid_body_rotation_3d_node_centered(nx=40, nz=10, omega=2.0 * math.pi / 3600.0, w0=0.005):
    """T1-3 3D helical motion, node-centered (u, v, w on nodes).

    Same field as :func:`solid_body_rotation_3d_face_centered` with horizontal velocity
    sampled at corner nodes. The horizontal field is linear and the vertical field is
    constant, so barycentric horizontal + linear vertical interpolation are both exact.
    """
    uxgrid = _build_delaunay_grid(nx, (-5.0, 5.0), (-5.0, 5.0))

    zf = np.linspace(0.0, 100.0, nz + 1, dtype=np.float64)
    zc = 0.5 * (zf[:-1] + zf[1:])
    time = xr.date_range("2000-01-01", periods=1, freq="2h")

    u_vals = -omega * uxgrid.node_lat.values
    v_vals = omega * uxgrid.node_lon.values
    u = np.broadcast_to(u_vals[np.newaxis, np.newaxis, :], (1, zc.size, uxgrid.n_node)).copy()
    v = np.broadcast_to(v_vals[np.newaxis, np.newaxis, :], (1, zc.size, uxgrid.n_node)).copy()
    w = np.full((1, zf.size, uxgrid.n_node), w0, dtype=np.float64)

    return _wrap_uvw_dataset(uxgrid, u, v, w, zc, zf, time, "n_node", "node", "3D solid-body rotation")


def sigma_coordinate_lattice_dataset(nx, x_range, y_range, nz, bottom_depth, eta=None, start_time="2000-01-01"):
    """Zero-velocity UGRID dataset on an nx-by-nx lattice: nz sigma interfaces from the surface (depth -eta) to bottom_depth (scalar or (x, y), meters); eta is None or (t, x, y), hourly from start_time."""
    uxgrid = _build_triangulated_lattice_grid(nx, x_range, y_range)
    sigma = np.linspace(0.0, 1.0, nz)
    if bottom_depth is None or np.ndim(bottom_depth) not in (0, 2):
        raise ValueError(
            f"bottom_depth must be a scalar or an (x, y) array, got {type(bottom_depth).__name__} with ndim {np.ndim(bottom_depth)}"
        )

    if eta is None:
        if np.ndim(bottom_depth) == 0:
            zf = sigma * bottom_depth  # (nz,)
        elif np.ndim(bottom_depth) == 2:
            zf = (sigma * (bottom_depth.T.ravel()[:, np.newaxis])).T  # (nz, n_node)
            zf = zf[np.newaxis, :, :]  # (1, nz, n_node)
    else:
        total_depth = np.swapaxes(bottom_depth + eta, 1, 2).reshape(eta.shape[0], -1)
        eta_nodes = np.swapaxes(eta, 1, 2).reshape(eta.shape[0], -1)
        zf = sigma[np.newaxis, :, np.newaxis] * total_depth[:, np.newaxis, :] - eta_nodes[:, np.newaxis, :]

    n_times = 1 if eta is None else eta.shape[0]
    time = xr.date_range(start_time, periods=n_times, freq="1h")
    zc = 0.5 * (zf[:-1] + zf[1:]) if zf.ndim == 1 else 0.5 * (zf[:, :-1] + zf[:, 1:])

    if zf.ndim == 1:
        vertical_coords: dict[str, tuple] = {"zf": ("zf", zf), "zc": ("zc", zc)}
    else:
        vertical_coords = {"zf": (("time", "zf", "n_node"), zf), "zc": (("time", "zc", "n_node"), zc)}

    velocity_dims = ("time", "zf", "n_node")
    velocity_shape = (n_times, nz, uxgrid.n_node)
    velocity_attrs = dict(units="m/s", location="node", mesh="triangulated_lattice", Conventions="UGRID-1.0")
    velocities = {name: (velocity_dims, np.zeros(velocity_shape), velocity_attrs) for name in ("U", "V", "W")}

    return ux.UxDataset(xr.Dataset(velocities, coords={"time": time, **vertical_coords}), uxgrid=uxgrid)
