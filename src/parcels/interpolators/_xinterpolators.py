"""Collection of pre-built interpolation kernels for structured grids."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import xarray as xr
from dask import is_dask_collection

import parcels._core.utils.interpolation as i_u
import parcels._sgrid as sgrid
import parcels._typing as ptyping
from parcels.interpolators._base import ScalarInterpolator, VectorInterpolator

if TYPE_CHECKING:
    from parcels._core._windowed_array import WindowedArray
    from parcels._core.field import Field, VectorField
    from parcels._core.xgrid import XGrid


_CORNER_AXES: tuple[ptyping.XgcmAxisDirection, ...] = ("T", "Z", "Y", "X")


def _gather_corners(
    data: xr.DataArray | WindowedArray,
    axis_dim: dict[ptyping.XgridAxis, str],
    levels: dict[ptyping.XgcmAxisDirection, tuple[np.ndarray, ...]],
    npart: int,
) -> np.ndarray:
    """Gather field data at the corners bracketing each particle.

    The gather is the outer product over ``_CORNER_AXES``, with each axis
    contributing one or two levels and the particle index innermost. The local
    variable ``shape`` states that layout, and is used both to build the index
    arrays and to reshape the result.

    Parameters
    ----------
    data : xr.DataArray or WindowedArray
        Field data, with dimensions ordered ``(time, Z, Y, X)``.
    axis_dim : dict
        Maps ``"X"``, ``"Y"`` and ``"Z"`` to dimension names. The ``"T"``
        dimension is always named ``"time"``.
    levels : dict
        Maps every axis in ``_CORNER_AXES`` to the per-particle index arrays of
        its corner levels, e.g. ``{"T": (ti,), "Z": (zi,), "Y": (yi, yi + 1),
        "X": (xi, xi + 1)}``. An axis bracketed by a single level is given a
        one-tuple. If the field has no dimension for an axis, that axis still
        contributes to the output shape, but is not indexed, so its corners
        repeat.
    npart : int
        Number of particles.

    Returns
    -------
    np.ndarray
        Array of shape ``(*counts, npart)``, where ``counts[i]`` is the number
        of levels gathered on ``_CORNER_AXES[i]``.
    """
    dims = {axis: ("time" if axis == "T" else axis_dim.get(axis)) for axis in _CORNER_AXES}
    counts = tuple(len(levels[axis]) for axis in _CORNER_AXES)
    shape = (*counts, npart)

    selection_dict = {}
    for i, axis in enumerate(_CORNER_AXES):
        if dims[axis] is not None and dims[axis] in data.dims:
            stacked = np.stack(np.broadcast_arrays(*levels[axis]))  # (n_levels, npart)
            # Put the level axis in slot i of the corner grid, spread it over the
            # other slots, and flatten. This is the inverse of the reshape below.
            other_slots = tuple(j for j in range(len(_CORNER_AXES)) if j != i)
            in_slot_i = np.expand_dims(stacked, other_slots)
            selection_dict[dims[axis]] = xr.DataArray(np.broadcast_to(in_slot_i, shape).reshape(-1), dims="points")

    return data.isel(selection_dict).data.reshape(shape)


def _get_corner_data_Agrid(
    data: xr.DataArray | WindowedArray,
    ti: np.ndarray,
    zi: np.ndarray,
    yi: np.ndarray,
    xi: np.ndarray,
    lenT: int,  # noqa: N803
    lenZ: int,  # noqa: N803
    npart: int,
    axis_dim: dict[ptyping.XgridAxis, str],
) -> np.ndarray:
    """Helper function to get the corner data for a given A-grid field and position."""
    levels: dict[ptyping.XgcmAxisDirection, tuple[np.ndarray, ...]] = {
        "T": (ti,) if lenT == 1 else (ti, np.clip(ti + 1, 0, data.shape[0] - 1)),
        "Z": (zi,) if lenZ == 1 else (zi, np.clip(zi + 1, 0, data.shape[1] - 1)),
        "Y": (yi, np.clip(yi + 1, 0, data.shape[2] - 1)),
        "X": (xi, np.clip(xi + 1, 0, data.shape[3] - 1)),
    }
    return _gather_corners(data, axis_dim, levels, npart)


def _get_offsets_dictionary(grid: XGrid) -> dict[ptyping.CfAxisSpatial, Literal[1, 0]]:
    offsets = {}
    metadata = grid.sgrid_metadata
    for fnp, axis in zip(metadata.face_dimensions, ["X", "Y"], strict=False):
        offsets[axis] = 1 if (fnp.padding == sgrid.Padding.LOW or fnp.padding == sgrid.Padding.BOTH) else 0
    if metadata.vertical_dimensions is not None:
        fnp_z = metadata.vertical_dimensions[0]
        offsets["Z"] = 1 if (fnp_z.padding == sgrid.Padding.LOW or fnp_z.padding == sgrid.Padding.BOTH) else 0
    else:
        offsets["Z"] = 0
    return offsets


class XLinear(ScalarInterpolator):
    """Trilinear interpolation on a regular grid."""

    def interp(
        self,
        particle_positions: dict[str, float | np.ndarray],
        grid_positions: dict[ptyping.XgridAxis, dict[str, int | float | np.ndarray]],
        field: Field,
    ):
        """Trilinear interpolation on a regular grid."""
        xi, xsi = grid_positions["X"]["index"], grid_positions["X"]["bcoord"]
        yi, eta = grid_positions["Y"]["index"], grid_positions["Y"]["bcoord"]
        zi, zeta = grid_positions["Z"]["index"], grid_positions["Z"]["bcoord"]
        ti, tau = grid_positions["T"]["index"], grid_positions["T"]["bcoord"]

        axis_dim = field.grid.get_axis_dim_mapping(field.data.dims)
        data = field.data

        lenT = 2 if np.any(tau > 0) else 1
        lenZ = 2 if np.any(zeta > 0) else 1

        corner_data = _get_corner_data_Agrid(data, ti, zi, yi, xi, lenT, lenZ, len(xsi), axis_dim)

        if lenT == 2:
            tau = tau[np.newaxis, :]
            corner_data = corner_data[0, :] * (1 - tau) + corner_data[1, :] * tau
        else:
            corner_data = corner_data[0, :]

        if lenZ == 2:
            zeta = zeta[np.newaxis, :]
            corner_data = corner_data[0, :] * (1 - zeta) + corner_data[1, :] * zeta
        else:
            corner_data = corner_data[0, :]

        value = (
            (1 - xsi) * (1 - eta) * corner_data[0, 0, :]
            + xsi * (1 - eta) * corner_data[0, 1, :]
            + (1 - xsi) * eta * corner_data[1, 0, :]
            + xsi * eta * corner_data[1, 1, :]
        )
        return value.compute() if is_dask_collection(value) else value


class XConstantField(ScalarInterpolator):
    """Returns the single value of a Constant Field (with a size=(1,1,1,1) array)."""

    def interp(
        self,
        particle_positions: dict[str, float | np.ndarray],
        grid_positions: dict[ptyping.XgridAxis, dict[str, int | float | np.ndarray]],
        field: Field,
    ):
        """Returning the single value of a Constant Field (with a size=(1,1,1,1) array)"""
        return field.data[0, 0, 0, 0].values * np.ones_like(particle_positions["x"])


class XLinear_Velocity(VectorInterpolator):  # noqa:  N801
    """Trilinear interpolation on a regular grid for VectorFields of velocity."""

    def interp(
        self,
        particle_positions: dict[str, float | np.ndarray],
        grid_positions: dict[ptyping.XgridAxis, dict[str, int | float | np.ndarray]],
        vectorfield: VectorField,
    ):
        """Trilinear interpolation on a regular grid for VectorFields of velocity."""
        _xlinear = XLinear()
        u = _xlinear.interp(particle_positions, grid_positions, vectorfield.U)
        v = _xlinear.interp(particle_positions, grid_positions, vectorfield.V)
        if vectorfield.grid.mesh.is_spherical():
            u /= vectorfield.grid.deg2m * np.cos(np.deg2rad(particle_positions["y"]))
            v /= vectorfield.grid.deg2m

        if vectorfield.W:
            w = _xlinear.interp(particle_positions, grid_positions, vectorfield.W)
        else:
            w = np.zeros_like(u)
        return u, v, w


class CGrid_Velocity(VectorInterpolator):  # noqa:  N801
    """
    Interpolation kernel for velocity fields on a C-Grid.
    Following Delandmeter and Van Sebille (2019), velocity fields should be interpolated
    only in the direction of the grid cell faces.
    """

    def interp(
        self,
        particle_positions: dict[str, float | np.ndarray],
        grid_positions: dict[ptyping.XgridAxis, dict[str, int | float | np.ndarray]],
        vectorfield: VectorField,
    ):
        """
        Interpolation kernel for velocity fields on a C-Grid.
        Following Delandmeter and Van Sebille (2019), velocity fields should be interpolated
        only in the direction of the grid cell faces.
        """
        xi, xsi = grid_positions["X"]["index"], grid_positions["X"]["bcoord"]
        yi, eta = grid_positions["Y"]["index"], grid_positions["Y"]["bcoord"]
        zi, zeta = grid_positions["Z"]["index"], grid_positions["Z"]["bcoord"]
        ti, tau = grid_positions["T"]["index"], grid_positions["T"]["bcoord"]

        U = vectorfield.U.data
        V = vectorfield.V.data
        grid = vectorfield.grid
        offsets = _get_offsets_dictionary(grid)
        tdim, zdim, ydim, xdim = U.shape[0], U.shape[1], U.shape[2], U.shape[3]
        lenT = 2 if np.any(tau > 0) else 1

        if grid.lon.ndim == 1:
            px = np.array([grid.lon[xi], grid.lon[xi + 1], grid.lon[xi + 1], grid.lon[xi]])
            py = np.array([grid.lat[yi], grid.lat[yi], grid.lat[yi + 1], grid.lat[yi + 1]])
        else:
            px = np.array([grid.lon[yi, xi], grid.lon[yi, xi + 1], grid.lon[yi + 1, xi + 1], grid.lon[yi + 1, xi]])
            py = np.array([grid.lat[yi, xi], grid.lat[yi, xi + 1], grid.lat[yi + 1, xi + 1], grid.lat[yi + 1, xi]])

        if grid.mesh.is_spherical():
            px = ((px + 180.0) % 360.0) - 180.0
            px[1:] = np.where(px[1:] - px[0] > 180, px[1:] - 360, px[1:])
            px[1:] = np.where(-px[1:] + px[0] > 180, px[1:] + 360, px[1:])
        c1 = i_u._geodetic_distance(
            py[0], py[1], px[0], px[1], grid.mesh, np.einsum("ij,ji->i", i_u.phi2D_lin(0.0, xsi), py), grid.deg2m
        )
        c2 = i_u._geodetic_distance(
            py[1], py[2], px[1], px[2], grid.mesh, np.einsum("ij,ji->i", i_u.phi2D_lin(eta, 1.0), py), grid.deg2m
        )
        c3 = i_u._geodetic_distance(
            py[2], py[3], px[2], px[3], grid.mesh, np.einsum("ij,ji->i", i_u.phi2D_lin(1.0, xsi), py), grid.deg2m
        )
        c4 = i_u._geodetic_distance(
            py[3], py[0], px[3], px[0], grid.mesh, np.einsum("ij,ji->i", i_u.phi2D_lin(eta, 0.0), py), grid.deg2m
        )

        npart = len(xsi)
        t_levels = (ti,) if lenT == 1 else (ti, np.clip(ti + 1, 0, tdim - 1))

        def _compute_corner_data(data, y_levels, x_levels, z_levels=None) -> np.ndarray:
            """Gather the two bracketing face values and reduce over time if needed.

            Exactly one of the Z, Y and X axes contributes the two corners. The
            other two contribute a single level each.
            """
            levels = {
                "T": t_levels,
                "Z": z_levels if z_levels is not None else (zi,),
                "Y": y_levels,
                "X": x_levels,
            }
            axis_dim = grid.get_axis_dim_mapping(data.dims)
            corner_data = _gather_corners(data, axis_dim, levels, npart).reshape(lenT, 2, npart)

            if lenT == 2:
                tau_full = tau[np.newaxis, :]
                corner_data = corner_data[0, :] * (1 - tau_full) + corner_data[1, :] * tau_full
            else:
                corner_data = corner_data[0, :]
            return corner_data

        # Compute U velocity: the two corners are the X faces
        yi_o = np.clip(yi + offsets["Y"], 0, ydim - 1)
        xi_1 = np.clip(xi + 1, 0, xdim - 1)
        corner_data = _compute_corner_data(U, y_levels=(yi_o,), x_levels=(xi, xi_1))

        U0 = corner_data[0, :] * c4
        U1 = corner_data[1, :] * c2
        Uvel = (1 - xsi) * U0 + xsi * U1

        # Compute V velocity: the two corners are the Y faces
        yi_1 = np.clip(yi + 1, 0, ydim - 1)
        xi_o = np.clip(xi + offsets["X"], 0, xdim - 1)
        corner_data = _compute_corner_data(V, y_levels=(yi, yi_1), x_levels=(xi_o,))

        V0 = corner_data[0, :] * c1
        V1 = corner_data[1, :] * c3
        Vvel = (1 - eta) * V0 + eta * V1

        if grid.mesh.is_spherical():
            jac = i_u._compute_jacobian_determinant(py, px, eta, xsi) * grid.deg2m
        else:
            jac = i_u._compute_jacobian_determinant(py, px, eta, xsi)

        u = (
            (-(1 - eta) * Uvel - (1 - xsi) * Vvel) * px[0]
            + ((1 - eta) * Uvel - xsi * Vvel) * px[1]
            + (eta * Uvel + xsi * Vvel) * px[2]
            + (-eta * Uvel + (1 - xsi) * Vvel) * px[3]
        ) / jac
        v = (
            (-(1 - eta) * Uvel - (1 - xsi) * Vvel) * py[0]
            + ((1 - eta) * Uvel - xsi * Vvel) * py[1]
            + (eta * Uvel + xsi * Vvel) * py[2]
            + (-eta * Uvel + (1 - xsi) * Vvel) * py[3]
        ) / jac
        if is_dask_collection(u):
            u = u.compute()
            v = v.compute()

        if grid.mesh.is_spherical():
            conversion = grid.deg2m * np.cos(np.deg2rad(particle_positions["y"]))
            u /= conversion
            v /= conversion

        if vectorfield.W:
            W = vectorfield.W.data

            # Compute W velocity: the two corners are the Z faces
            yi_o = np.clip(yi + offsets["Y"], 0, ydim - 1)
            xi_o = np.clip(xi + offsets["X"], 0, xdim - 1)
            zi_0 = np.clip(zi + offsets["Z"], 0, zdim - 1)
            zi_1 = np.clip(zi + offsets["Z"] + 1, 0, zdim - 1)
            corner_data = _compute_corner_data(W, y_levels=(yi_o,), x_levels=(xi_o,), z_levels=(zi_0, zi_1))

            w = corner_data[0, :] * (1 - zeta) + corner_data[1, :] * zeta
            if is_dask_collection(w):
                w = w.compute()
        else:
            w = np.zeros_like(u)

        return (u, v, w)


class CGrid_Tracer(ScalarInterpolator):  # noqa:  N801
    """
    Interpolation kernel for tracer fields on a C-Grid.
    Following Delandmeter and Van Sebille (2019), tracer fields should be interpolated
    constant over the grid cell.
    """

    def interp(
        self,
        particle_positions: dict[str, float | np.ndarray],
        grid_positions: dict[ptyping.XgridAxis, dict[str, int | float | np.ndarray]],
        field: Field,
    ):
        """Interpolation kernel for tracer fields on a C-Grid.

        Following Delandmeter and Van Sebille (2019), tracer fields should be interpolated
        constant over the grid cell
        """
        xi = grid_positions["X"]["index"]
        yi = grid_positions["Y"]["index"]
        zi = grid_positions["Z"]["index"]
        ti = grid_positions["T"]["index"]
        tau = grid_positions["T"]["bcoord"]

        axis_dim = field.grid.get_axis_dim_mapping(field.data.dims)
        data = field.data

        offsets = _get_offsets_dictionary(field.grid)
        zi = np.clip(zi + offsets["Z"], 0, data.shape[1] - 1)
        yi = np.clip(yi + offsets["Y"], 0, data.shape[2] - 1)
        xi = np.clip(xi + offsets["X"], 0, data.shape[3] - 1)

        lenT = 2 if np.any(tau > 0) else 1
        npart = len(xi)

        levels = {
            "T": (ti,) if lenT == 1 else (ti, np.clip(ti + 1, 0, data.shape[0] - 1)),
            "Z": (zi,),
            "Y": (yi,),
            "X": (xi,),
        }
        value = _gather_corners(data, axis_dim, levels, npart).reshape(lenT, npart)

        if lenT == 2:
            value = value[0, :] * (1 - tau) + value[1, :] * tau
        else:
            value = value[0, :]

        return value.compute() if is_dask_collection(value) else value


def _Spatialslip(
    particle_positions: dict[str, float | np.ndarray],
    grid_positions: dict[ptyping.XgridAxis, dict[str, int | float | np.ndarray]],
    vectorfield: VectorField,
    a: np.float32,
    b: np.float32,
):
    """Helper function for spatial boundary condition interpolation for velocity fields."""
    xi, xsi = grid_positions["X"]["index"], grid_positions["X"]["bcoord"]
    yi, eta = grid_positions["Y"]["index"], grid_positions["Y"]["bcoord"]
    zi, zeta = grid_positions["Z"]["index"], grid_positions["Z"]["bcoord"]
    ti, tau = grid_positions["T"]["index"], grid_positions["T"]["bcoord"]

    axis_dim = vectorfield.U.grid.get_axis_dim_mapping(vectorfield.U.data.dims)
    lenT = 2 if np.any(tau > 0) else 1
    lenZ = 2 if np.any(zeta > 0) else 1
    npart = len(xsi)

    _xlinear = XLinear()
    u = _xlinear.interp(particle_positions, grid_positions, vectorfield.U)
    v = _xlinear.interp(particle_positions, grid_positions, vectorfield.V)
    if vectorfield.W:
        w = _xlinear.interp(particle_positions, grid_positions, vectorfield.W)

    corner_dataU = _get_corner_data_Agrid(vectorfield.U.data, ti, zi, yi, xi, lenT, lenZ, npart, axis_dim)
    corner_dataV = _get_corner_data_Agrid(vectorfield.V.data, ti, zi, yi, xi, lenT, lenZ, npart, axis_dim)

    def is_land(ti: int, zi: int, yi: int, xi: int):
        uval = corner_dataU[ti, zi, yi, xi, :]
        vval = corner_dataV[ti, zi, yi, xi, :]
        return np.where(np.isclose(uval, 0.0) & np.isclose(vval, 0.0), True, False)

    if is_dask_collection(u):
        u = u.compute()
        v = v.compute()
        if vectorfield.W:
            w = w.compute()

    f_u = np.ones_like(xsi)
    if lenZ == 1:
        land = is_land(0, 0, 0, 0) & is_land(0, 0, 0, 1) & (eta > 0)
    else:
        land = is_land(0, 0, 0, 0) & is_land(0, 0, 0, 1) & is_land(0, 1, 0, 0) & is_land(0, 1, 0, 1) & (eta > 0)
    f_u[land] = f_u[land] * (a + b * eta[land]) / eta[land]

    if lenZ == 1:
        land = is_land(0, 0, 1, 0) & is_land(0, 0, 1, 1) & (eta < 1)
    else:
        land = is_land(0, 0, 1, 0) & is_land(0, 0, 1, 1) & is_land(0, 1, 1, 0) & is_land(0, 1, 1, 1) & (eta < 1)
    f_u[land] = f_u[land] * (1 - b * eta[land]) / (1 - eta[land])

    u = u * f_u
    if vectorfield.grid.mesh.is_spherical():
        u /= 1852 * 60 * np.cos(np.deg2rad(particle_positions["y"]))

    f_v = np.ones_like(eta)
    if lenZ == 1:
        land = is_land(0, 0, 0, 0) & is_land(0, 0, 1, 0) & (xsi > 0)
    else:
        land = is_land(0, 0, 0, 0) & is_land(0, 0, 1, 0) & is_land(0, 1, 0, 0) & is_land(0, 1, 1, 0) & (xsi > 0)
    f_v[land] = f_v[land] * (a + b * xsi[land]) / xsi[land]

    if lenZ == 1:
        land = is_land(0, 0, 0, 1) & is_land(0, 0, 1, 1) & (xsi < 1)
    else:
        land = is_land(0, 0, 0, 1) & is_land(0, 0, 1, 1) & is_land(0, 1, 0, 1) & is_land(0, 1, 1, 1) & (xsi < 1)
    f_v[land] = f_v[land] * (1 - b * xsi[land]) / (1 - xsi[land])

    v = v * f_v
    if vectorfield.grid.mesh.is_spherical():
        v /= 1852 * 60

    if vectorfield.W:
        f_w = np.ones_like(zeta)
        land = is_land(0, 0, 0, 0) & is_land(0, 0, 0, 1) & is_land(0, 1, 0, 0) & is_land(0, 1, 0, 1) & (eta > 0)
        f_w[land] = f_w[land] * (a + b * eta[land]) / eta[land]

        land = is_land(0, 0, 1, 0) & is_land(0, 0, 1, 1) & is_land(0, 1, 1, 0) & is_land(0, 1, 1, 1) & (eta < 1)
        f_w[land] = f_w[land] * (1 - b * eta[land]) / (1 - eta[land])

        land = is_land(0, 0, 0, 0) & is_land(0, 0, 1, 0) & is_land(0, 1, 0, 0) & is_land(0, 1, 1, 0) & (xsi > 0)
        f_w[land] = f_w[land] * (a + b * xsi[land]) / xsi[land]

        land = is_land(0, 0, 0, 1) & is_land(0, 0, 1, 1) & is_land(0, 1, 0, 1) & is_land(0, 1, 1, 1) & (xsi < 1)
        f_w[land] = f_w[land] * (1 - b * xsi[land]) / (1 - xsi[land])

        w = w * f_w
    else:
        w = np.zeros_like(u)

    return u, v, w


class XFreeslip(VectorInterpolator):
    """Free-slip boundary condition interpolation for velocity fields."""

    def interp(
        self,
        particle_positions: dict[str, float | np.ndarray],
        grid_positions: dict[ptyping.XgridAxis, dict[str, int | float | np.ndarray]],
        vectorfield: VectorField,
    ):
        """Free-slip boundary condition interpolation for velocity fields."""
        return _Spatialslip(particle_positions, grid_positions, vectorfield, a=1.0, b=0.0)


class XPartialslip(VectorInterpolator):
    """Partial-slip boundary condition interpolation for velocity fields."""

    def interp(
        self,
        particle_positions: dict[str, float | np.ndarray],
        grid_positions: dict[ptyping.XgridAxis, dict[str, int | float | np.ndarray]],
        vectorfield: VectorField,
    ):
        """Partial-slip boundary condition interpolation for velocity fields."""
        return _Spatialslip(particle_positions, grid_positions, vectorfield, a=0.5, b=0.5)


class XNearest(ScalarInterpolator):
    """
    Nearest-Neighbour spatial interpolation on a regular grid.
    Note that this still uses linear interpolation in time.
    """

    def interp(
        self,
        particle_positions: dict[str, float | np.ndarray],
        grid_positions: dict[ptyping.XgridAxis, dict[str, int | float | np.ndarray]],
        field: Field,
    ):
        """
        Nearest-Neighbour spatial interpolation on a regular grid.
        Note that this still uses linear interpolation in time.
        """
        xi, xsi = grid_positions["X"]["index"], grid_positions["X"]["bcoord"]
        yi, eta = grid_positions["Y"]["index"], grid_positions["Y"]["bcoord"]
        zi, zeta = grid_positions["Z"]["index"], grid_positions["Z"]["bcoord"]
        ti, tau = grid_positions["T"]["index"], grid_positions["T"]["bcoord"]

        axis_dim = field.grid.get_axis_dim_mapping(field.data.dims)
        data = field.data

        lenT = 2 if np.any(tau > 0) else 1

        # Spatial coordinates: left if barycentric < 0.5, otherwise right
        zi_1 = np.clip(zi + 1, 0, data.shape[1] - 1)
        zi_full = np.where(zeta <= 0.5, zi, zi_1)

        yi_1 = np.clip(yi + 1, 0, data.shape[2] - 1)
        yi_full = np.where(eta <= 0.5, yi, yi_1)

        xi_1 = np.clip(xi + 1, 0, data.shape[3] - 1)
        xi_full = np.where(xsi <= 0.5, xi, xi_1)

        levels: dict[ptyping.XgcmAxisDirection, tuple[np.ndarray, ...]] = {
            "T": (ti,) if lenT == 1 else (ti, np.clip(ti + 1, 0, data.shape[0] - 1)),
            "Z": (zi_full,),
            "Y": (yi_full,),
            "X": (xi_full,),
        }
        corner_data = _gather_corners(data, axis_dim, levels, len(xsi))[:, 0, 0, 0]

        if lenT == 2:
            value = corner_data[0] * (1 - tau) + corner_data[1] * tau
        else:
            value = corner_data[0]
        return value.compute() if is_dask_collection(value) else value


class XLinearInvdistLandTracer(ScalarInterpolator):
    """Linear spatial interpolation on a regular grid, where points on land are not used."""

    def interp(
        self,
        particle_positions: dict[str, float | np.ndarray],
        grid_positions: dict[ptyping.XgridAxis, dict[str, int | float | np.ndarray]],
        field: Field,
    ):
        """Linear spatial interpolation on a regular grid, where points on land are not used."""
        values = XLinear().interp(particle_positions, grid_positions, field)

        xi, xsi = grid_positions["X"]["index"], grid_positions["X"]["bcoord"]
        yi, eta = grid_positions["Y"]["index"], grid_positions["Y"]["bcoord"]
        zi, zeta = grid_positions["Z"]["index"], grid_positions["Z"]["bcoord"]
        ti, tau = grid_positions["T"]["index"], grid_positions["T"]["bcoord"]

        axis_dim = field.grid.get_axis_dim_mapping(field.data.dims)
        lenT = 2 if np.any(tau > 0) else 1
        lenZ = 2 if np.any(zeta > 0) else 1

        corner_data = _get_corner_data_Agrid(field.data, ti, zi, yi, xi, lenT, lenZ, len(xsi), axis_dim)

        land_mask = np.isclose(corner_data, 0.0)
        nb_land = np.sum(land_mask, axis=(0, 1, 2, 3))

        if np.any(nb_land):
            all_land_mask = nb_land == 4 * lenZ * lenT
            values[all_land_mask] = 0.0

            some_land = np.logical_and(nb_land > 0, nb_land < 4 * lenZ * lenT)
            if np.any(some_land):
                i_grid = np.arange(2)[None, None, None, :, None]
                j_grid = np.arange(2)[None, None, :, None, None]
                eta_b = eta[None, None, None, None, :]
                xsi_b = xsi[None, None, None, None, :]

                dist2 = (eta_b - j_grid) ** 2 + (xsi_b - i_grid) ** 2

                valid_mask = ~land_mask
                # Normal inverse-distance weighting
                inv_dist = 1.0 / dist2
                weighted = np.where(valid_mask, corner_data * inv_dist, 0.0)

                val = np.sum(weighted, axis=(0, 1, 2, 3))
                w_sum = np.sum(np.where(valid_mask, inv_dist, 0.0), axis=(0, 1, 2, 3))

                values[some_land] = val[some_land] / w_sum[some_land]

                # If a particle hits exactly one of the 8 corner points, extract it
                exact_mask = (dist2 == 0) & valid_mask
                exact_vals = np.sum(np.where(exact_mask, corner_data, 0.0), axis=(0, 1, 2, 3))
                has_exact = np.any(exact_mask, axis=(0, 1, 2, 3))

                exact_particles = some_land & has_exact
                values[exact_particles] = exact_vals[exact_particles]

        return values.compute() if is_dask_collection(values) else values
