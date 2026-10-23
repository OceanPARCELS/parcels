import math

import numpy as np
import uxarray as ux
import xarray as xr

__all__ = ["Nx", "datasets"]

T = 13
Nx = 20
vmax = 1.0
delta = 0.1
TIME = xr.date_range("2000", "2001", T)


def _stommel_gyre_delaunay():
    """
    Stommel gyre on a Delaunay grid. the naming convention of the dataset and grid is consistent with what is
    provided by UXArray when reading in FESOM2 datasets.
    This dataset is a single vertical layer of a barotropic ocean gyre on a square domain with closed boundaries.
    The velocity field provides a slow moving interior circulation and a western boundary current. All fields are placed
    on the vertices of the grid and at the element vertical faces.
    """
    lon, lat = np.meshgrid(np.linspace(0, 60.0, Nx, dtype=np.float32), np.linspace(0, 60.0, Nx, dtype=np.float32))
    lon_flat = lon.ravel()
    lat_flat = lat.ravel()
    zf = np.linspace(0.0, 1000.0, 2, endpoint=True, dtype=np.float32)  # Vertical element faces
    zc = 0.5 * (zf[:-1] + zf[1:])  # Vertical element centers

    # mask any point on one of the boundaries
    mask = (
        np.isclose(lon_flat, 0.0) | np.isclose(lon_flat, 60.0) | np.isclose(lat_flat, 0.0) | np.isclose(lat_flat, 60.0)
    )

    boundary_points = np.flatnonzero(mask)

    uxgrid = ux.Grid.from_points(
        (lon_flat, lat_flat),
        method="regional_delaunay",
        boundary_points=boundary_points,
    )
    uxgrid.attrs["Conventions"] = "UGRID-1.0"

    # Define arrays U (zonal), V (meridional) and P (sea surface height)
    U = np.zeros((1, zc.size, uxgrid.n_face), dtype=np.float64)
    V = np.zeros((1, zc.size, uxgrid.n_face), dtype=np.float64)
    W = np.zeros((1, zf.size, lat.size), dtype=np.float64)
    P = np.zeros((1, zc.size, uxgrid.n_face), dtype=np.float64)

    for i, (x, y) in enumerate(zip(uxgrid.face_lon, uxgrid.face_lat, strict=False)):
        xi = x / 60.0
        yi = y / 60.0

        P[0, 0, i] = -vmax * delta * (1 - xi) * (math.exp(-xi / delta) - 1) * np.sin(math.pi * yi)
        U[0, 0, i] = -vmax * (1 - math.exp(-xi / delta) - xi) * np.cos(math.pi * yi)
        V[0, 0, i] = vmax * ((2.0 - xi) * math.exp(-xi / delta) - 1) * np.sin(math.pi * yi)

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
            description="zonal velocity", units="m/s", location="node", mesh="delaunay", Conventions="UGRID-1.0"
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
            description="meridional velocity", units="m/s", location="node", mesh="delaunay", Conventions="UGRID-1.0"
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
        attrs=dict(description="pressure", units="N/m^2", location="node", mesh="delaunay", Conventions="UGRID-1.0"),
    )

    return ux.UxDataset({"U": u, "V": v, "W": w, "p": p}, uxgrid=uxgrid)


def _fesom2_square_delaunay_uniform_z_coordinate():
    """
    Delaunay grid with uniform z-coordinate, mimicking a FESOM2 dataset.
    This dataset consists of a square domain with closed boundaries, where the grid is generated using Delaunay triangulation.
    The bottom topography is flat and uniform, and the vertical grid spacing is constant with 10 layers spanning [0,1000.0]
    The lateral velocity field components are non-zero constant, and the vertical velocity component is zero.
    The pressure field is constant.
    All fields are placed on location consistent with FESOM2 variable placement conventions
    """
    lon, lat = np.meshgrid(np.linspace(0, 60.0, Nx, dtype=np.float32), np.linspace(0, 60.0, Nx, dtype=np.float32))
    lon_flat = lon.ravel()
    lat_flat = lat.ravel()
    zf = np.linspace(0.0, 1000.0, 10, endpoint=True, dtype=np.float32)  # Vertical element faces
    zc = 0.5 * (zf[:-1] + zf[1:])  # Vertical element centers

    # mask any point on one of the boundaries
    mask = (
        np.isclose(lon_flat, 0.0) | np.isclose(lon_flat, 60.0) | np.isclose(lat_flat, 0.0) | np.isclose(lat_flat, 60.0)
    )

    boundary_points = np.flatnonzero(mask)

    uxgrid = ux.Grid.from_points(
        (lon_flat, lat_flat),
        method="regional_delaunay",
        boundary_points=boundary_points,
    )
    uxgrid.attrs["Conventions"] = "UGRID-1.0"

    # Define arrays U (zonal), V (meridional) and P (sea surface height)
    U = np.ones(
        (T, zc.size, uxgrid.n_face), dtype=np.float64
    )  # Lateral velocity is on the element centers and face centers
    V = np.ones(
        (T, zc.size, uxgrid.n_face), dtype=np.float64
    )  # Lateral velocity is on the element centers and face centers
    W = np.zeros(
        (T, zf.size, uxgrid.n_node), dtype=np.float64
    )  # Vertical velocity is on the element faces and face vertices
    P = np.ones((T, zc.size, uxgrid.n_node), dtype=np.float64)  # Pressure is on the element centers and face vertices

    u = ux.UxDataArray(
        data=U,
        name="U",
        uxgrid=uxgrid,
        dims=["time", "nz1", "n_face"],
        coords=dict(
            time=(["time"], TIME),
            nz1=(["nz1"], zc),
        ),
        attrs=dict(
            description="zonal velocity", units="m/s", location="face", mesh="delaunay", Conventions="UGRID-1.0"
        ),
    )
    v = ux.UxDataArray(
        data=V,
        name="V",
        uxgrid=uxgrid,
        dims=["time", "nz1", "n_face"],
        coords=dict(
            time=(["time"], TIME),
            nz1=(["nz1"], zc),
        ),
        attrs=dict(
            description="meridional velocity", units="m/s", location="face", mesh="delaunay", Conventions="UGRID-1.0"
        ),
    )
    w = ux.UxDataArray(
        data=W,
        name="w",
        uxgrid=uxgrid,
        dims=["time", "nz", "n_node"],
        coords=dict(
            time=(["time"], TIME),
            nz=(["nz"], zf),
        ),
        attrs=dict(
            description="vertical velocity", units="m/s", location="node", mesh="delaunay", Conventions="UGRID-1.0"
        ),
    )
    p = ux.UxDataArray(
        data=P,
        name="p",
        uxgrid=uxgrid,
        dims=["time", "nz1", "n_node"],
        coords=dict(
            time=(["time"], TIME),
            nz1=(["nz1"], zc),
        ),
        attrs=dict(description="pressure", units="N/m^2", location="node", mesh="delaunay", Conventions="UGRID-1.0"),
    )

    return ux.UxDataset({"U": u, "V": v, "W": w, "p": p}, uxgrid=uxgrid)


def _fesom2_square_delaunay_antimeridian():
    """
    Delaunay grid that crosses the antimeridian with uniform z-coordinate, mimicking a FESOM2 dataset.
    This dataset consists of a square domain with closed boundaries, where the grid is generated using Delaunay triangulation.
    The bottom topography is flat and uniform, and the vertical grid spacing is constant with 10 layers spanning [0,1000.0]
    The lateral velocity field components are non-zero constant, and the vertical velocity component is zero.
    The pressure field is constant.
    All fields are placed on location consistent with FESOM2 variable placement conventions
    """
    lon, lat = np.meshgrid(
        np.linspace(-210.0, -150.0, Nx, dtype=np.float32), np.linspace(-40.0, 40.0, Nx, dtype=np.float32)
    )
    # wrap longitude from [-180,180]
    lon_flat = lon.ravel()
    lat_flat = lat.ravel()
    zf = np.linspace(0.0, 1000.0, 10, endpoint=True, dtype=np.float32)  # Vertical element faces
    zc = 0.5 * (zf[:-1] + zf[1:])  # Vertical element centers

    # mask any point on one of the boundaries
    mask = (
        np.isclose(lon_flat, -210.0)
        | np.isclose(lon_flat, -150.0)
        | np.isclose(lat_flat, -40.0)
        | np.isclose(lat_flat, 40.0)
    )

    boundary_points = np.flatnonzero(mask)

    uxgrid = ux.Grid.from_points(
        (lon_flat, lat_flat),
        method="regional_delaunay",
        boundary_points=boundary_points,
    )
    uxgrid.attrs["Conventions"] = "UGRID-1.0"

    # Define arrays U (zonal), V (meridional) and P (sea surface height)
    U = np.ones(
        (T, zc.size, uxgrid.n_face), dtype=np.float64
    )  # Lateral velocity is on the element centers and face centers
    V = np.ones(
        (T, zc.size, uxgrid.n_face), dtype=np.float64
    )  # Lateral velocity is on the element centers and face centers
    W = np.zeros(
        (T, zf.size, uxgrid.n_node), dtype=np.float64
    )  # Vertical velocity is on the element faces and face vertices
    P = np.ones((T, zc.size, uxgrid.n_node), dtype=np.float64)  # Pressure is on the element centers and face vertices

    u = ux.UxDataArray(
        data=U,
        name="U",
        uxgrid=uxgrid,
        dims=["time", "nz1", "n_face"],
        coords=dict(
            time=(["time"], TIME),
            nz1=(["nz1"], zc),
        ),
        attrs=dict(
            description="zonal velocity", units="m/s", location="face", mesh="delaunay", Conventions="UGRID-1.0"
        ),
    )
    v = ux.UxDataArray(
        data=V,
        name="V",
        uxgrid=uxgrid,
        dims=["time", "nz1", "n_face"],
        coords=dict(
            time=(["time"], TIME),
            nz1=(["nz1"], zc),
        ),
        attrs=dict(
            description="meridional velocity", units="m/s", location="face", mesh="delaunay", Conventions="UGRID-1.0"
        ),
    )
    w = ux.UxDataArray(
        data=W,
        name="w",
        uxgrid=uxgrid,
        dims=["time", "nz", "n_node"],
        coords=dict(
            time=(["time"], TIME),
            nz=(["nz"], zf),
        ),
        attrs=dict(
            description="vertical velocity", units="m/s", location="node", mesh="delaunay", Conventions="UGRID-1.0"
        ),
    )
    p = ux.UxDataArray(
        data=P,
        name="p",
        uxgrid=uxgrid,
        dims=["time", "nz1", "n_node"],
        coords=dict(
            time=(["time"], TIME),
            nz1=(["nz1"], zc),
        ),
        attrs=dict(description="pressure", units="N/m^2", location="node", mesh="delaunay", Conventions="UGRID-1.0"),
    )

    return ux.UxDataset({"U": u, "V": v, "W": w, "p": p}, uxgrid=uxgrid)


def _icon_square_delaunay_uniform_z_coordinate():
    """
    Delaunay grid with uniform z-coordinate, mimicking an ICON dataset.
    This dataset consists of a square domain with closed boundaries, where the grid is generated using Delaunay triangulation.
    The bottom topography is flat and uniform, and the vertical grid spacing is constant with 10 layers spanning [0,1000.0]
    The lateral velocity field components are non-zero constant, and the vertical velocity component is zero.
    The pressure field is constant.
    All fields are face registered and at vertical layer centers, except for the vertical velocity component, which is
    at vertical layer interfaces.
    """
    lon, lat = np.meshgrid(np.linspace(0, 60.0, Nx, dtype=np.float64), np.linspace(0, 60.0, Nx, dtype=np.float64))
    lon_flat = lon.ravel()
    lat_flat = lat.ravel()
    zf = np.linspace(0.0, 1000.0, 10, endpoint=True, dtype=np.float64)  # Vertical element faces
    zc = 0.5 * (zf[:-1] + zf[1:])  # Vertical element centers

    # mask any point on one of the boundaries
    mask = (
        np.isclose(lon_flat, 0.0) | np.isclose(lon_flat, 60.0) | np.isclose(lat_flat, 0.0) | np.isclose(lat_flat, 60.0)
    )

    boundary_points = np.flatnonzero(mask)

    uxgrid = ux.Grid.from_points(
        (lon_flat, lat_flat),
        method="regional_delaunay",
        boundary_points=boundary_points,
    )

    # Define arrays U (zonal), V (meridional) and P (sea surface height)
    U = np.ones(
        (T, zc.size, uxgrid.n_face), dtype=np.float64
    )  # Lateral velocity is on the element centers and face centers
    V = np.ones(
        (T, zc.size, uxgrid.n_face), dtype=np.float64
    )  # Lateral velocity is on the element centers and face centers
    W = np.zeros(
        (T, zf.size, uxgrid.n_face), dtype=np.float64
    )  # Vertical velocity is on the element faces and face vertices
    P = np.ones((T, zc.size, uxgrid.n_node), dtype=np.float64)  # Pressure is on the element centers and face vertices

    U[0, :, :] = zc[:, None] * uxgrid.face_lon.values[None, :]
    V[0, :, :] = zc[:, None] * uxgrid.face_lat.values[None, :]
    W[0, :, :] = zf[:, None] * uxgrid.face_lon.values[None, :] * uxgrid.face_lat.values[None, :]
    P[0, :, :] = zc[:, None] * (uxgrid.node_lon.values[None, :] + uxgrid.node_lat.values[None, :])

    u = ux.UxDataArray(
        data=U,
        name="U",
        uxgrid=uxgrid,
        dims=["time", "depth", "n_face"],
        coords=dict(
            time=(["time"], TIME),
            depth=(["depth"], zc),
        ),
        attrs=dict(
            description="zonal velocity", units="m/s", location="face", mesh="delaunay", Conventions="UGRID-1.0"
        ),
    )
    v = ux.UxDataArray(
        data=V,
        name="V",
        uxgrid=uxgrid,
        dims=["time", "depth", "n_face"],
        coords=dict(
            time=(["time"], TIME),
            depth=(["depth"], zc),
        ),
        attrs=dict(
            description="meridional velocity", units="m/s", location="face", mesh="delaunay", Conventions="UGRID-1.0"
        ),
    )
    w = ux.UxDataArray(
        data=W,
        name="W",
        uxgrid=uxgrid,
        dims=["time", "depth_2", "n_face"],
        coords=dict(
            time=(["time"], TIME),
            depth_2=(["depth_2"], zf),
        ),
        attrs=dict(
            description="vertical velocity", units="m/s", location="node", mesh="delaunay", Conventions="UGRID-1.0"
        ),
    )
    p = ux.UxDataArray(
        data=P,
        name="p",
        uxgrid=uxgrid,
        dims=["time", "depth", "n_node"],
        coords=dict(
            time=(["time"], TIME),
            depth=(["depth"], zc),
        ),
        attrs=dict(description="pressure", units="N/m^2", location="node", mesh="delaunay", Conventions="UGRID-1.0"),
    )

    return ux.UxDataset({"U": u, "V": v, "W": w, "p": p}, uxgrid=uxgrid)


def _ux_constant_flow_face_centered_2D():
    NX = 10
    NT = 2
    lon, lat = np.meshgrid(
        np.linspace(0, 20, NX, dtype=np.float64),
        np.linspace(0, 20, NX, dtype=np.float64),
    )
    lon_flat, lat_flat = lon.ravel(), lat.ravel()
    mask = np.isclose(lon_flat, 0) | np.isclose(lon_flat, 20) | np.isclose(lat_flat, 0) | np.isclose(lat_flat, 20)
    uxgrid = ux.Grid.from_points(
        (lon_flat, lat_flat),
        method="regional_delaunay",
        boundary_points=np.flatnonzero(mask),
    )
    uxgrid.attrs["Conventions"] = "UGRID-1.0"

    # --- Uniform velocity field on face centers ---
    U0 = 0.001  # degrees/s
    V0 = 0.0
    TIME = xr.date_range("2000-01-01", periods=NT, freq="1h")
    zf = np.array([0.0, 1.0])
    zc = np.array([0.5])

    U = np.full((NT, 1, uxgrid.n_face), U0)
    V = np.full((NT, 1, uxgrid.n_face), V0)
    W = np.zeros((NT, 2, uxgrid.n_node))

    ds = ux.UxDataset(
        {
            "U": ux.UxDataArray(
                U,
                uxgrid=uxgrid,
                dims=["time", "zc", "n_face"],
                coords=dict(time=(["time"], TIME), zc=(["zc"], zc)),
                attrs=dict(location="face", mesh="delaunay", Conventions="UGRID-1.0"),
            ),
            "V": ux.UxDataArray(
                V,
                uxgrid=uxgrid,
                dims=["time", "zc", "n_face"],
                coords=dict(time=(["time"], TIME), zc=(["zc"], zc)),
                attrs=dict(location="face", mesh="delaunay", Conventions="UGRID-1.0"),
            ),
            "W": ux.UxDataArray(
                W,
                uxgrid=uxgrid,
                dims=["time", "zf", "n_node"],
                coords=dict(time=(["time"], TIME), nz=(["zf"], zf)),
                attrs=dict(location="node", mesh="delaunay", Conventions="UGRID-1.0"),
            ),
        },
        uxgrid=uxgrid,
    )
    ds.uxgrid.node_lon.attrs["units"] = "m"
    ds.uxgrid.node_lat.attrs["units"] = "m"
    return ds


datasets = {
    "stommel_gyre_delaunay": _stommel_gyre_delaunay(),
    "fesom2_square_delaunay_uniform_z_coordinate": _fesom2_square_delaunay_uniform_z_coordinate(),
    "fesom2_square_delaunay_antimeridian": _fesom2_square_delaunay_antimeridian(),
    "icon_square_delaunay_uniform_z_coordinate": _icon_square_delaunay_uniform_z_coordinate(),
    "ux_constant_flow_face_centered_2D": _ux_constant_flow_face_centered_2D(),
}
