"""Collection of pre-built interpolation kernels for unstructured grids."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from dask import is_dask_collection

from parcels.interpolators._base import ScalarInterpolator, VectorInterpolator

if TYPE_CHECKING:
    from parcels._core.field import Field, VectorField
    from parcels._core.uxgrid import _UXGRID_AXES


class UxConstantFaceConstantZC(ScalarInterpolator):
    """Piecewise constant interpolation kernel for face registered data that is vertically centered (on zc points)"""

    def interp(
        self,
        particle_positions: dict[str, float | np.ndarray],
        grid_positions: dict[_UXGRID_AXES, dict[str, int | float | np.ndarray]],
        field: Field,
    ):
        # Broadcast the per-axis indices to a common (npart,) shape (``ti`` may be scalar for time-constant fields)
        ti, zi, fi = np.broadcast_arrays(
            grid_positions["T"]["index"], grid_positions["Z"]["index"], grid_positions["FACE"]["index"]
        )

        tdim, zdim, fdim = field.data.dims
        selection_dict = {
            tdim: xr.DataArray(ti, dims="points"),
            zdim: xr.DataArray(zi, dims="points"),
            fdim: xr.DataArray(fi, dims="points"),
        }
        value = field.data.isel(selection_dict, ignore_grid=True).data
        return value.compute() if is_dask_collection(value) else value


class UxConstantFaceLinearZF(ScalarInterpolator):
    """
    Piecewise constant interpolation (lateral) with linear vertical interpolation kernel for face registered data
    that is located at vertical interface levels (on zf points)
    """

    def interp(
        self,
        particle_positions: dict[str, float | np.ndarray],
        grid_positions: dict[_UXGRID_AXES, dict[str, int | float | np.ndarray]],
        field: Field,
    ):
        ti, zi, fi = np.broadcast_arrays(
            grid_positions["T"]["index"], grid_positions["Z"]["index"], grid_positions["FACE"]["index"]
        )
        zeta = grid_positions["Z"]["bcoord"]

        tdim, zdim, fdim = field.data.dims

        def _zsample(z_index):
            """Pointwise ``isel`` of the face values at a single vertical interface level."""
            selection_dict = {
                tdim: xr.DataArray(ti, dims="points"),
                zdim: xr.DataArray(z_index, dims="points"),
                fdim: xr.DataArray(fi, dims="points"),
            }
            value = field.data.isel(selection_dict, ignore_grid=True).data
            return value.compute() if is_dask_collection(value) else value

        # The zi refers to the vertical layer index. The field in this routine are assumed to be defined at the vertical interface levels.
        # For interface zi, the interface indices are [zi, zi+1], so we need to use the values at zi and zi+1.
        # First, do barycentric interpolation in the lateral direction for each interface level
        fzk = _zsample(zi)
        fzkp1 = _zsample(zi + 1)

        # Then, do piecewise linear interpolation in the vertical direction
        return fzk * (1 - zeta) + fzkp1 * zeta


class UxLinearNodeConstantZC(ScalarInterpolator):
    """Piecewise linear interpolation kernel for node registered data that is vertically centered (zc points)."""

    def interp(
        self,
        particle_positions: dict[str, float | np.ndarray],
        grid_positions: dict[_UXGRID_AXES, dict[str, int | float | np.ndarray]],
        field: Field,
    ):
        """
        Piecewise linear interpolation kernel for node registered data that is vertically centered (zc points).
        Effectively, it applies barycentric interpolation in the lateral direction
        and piecewise constant interpolation in the vertical direction.
        """
        ti, zi, fi = np.broadcast_arrays(
            grid_positions["T"]["index"], grid_positions["Z"]["index"], grid_positions["FACE"]["index"]
        )
        bcoords = xr.DataArray(grid_positions["FACE"]["bcoord"], dims=("points", "nodes"))
        node_ids = field.grid.uxgrid.face_node_connectivity[fi, :].values

        tdim, zdim, ndim = field.data.dims
        selection_dict = {
            tdim: xr.DataArray(ti, dims="points"),
            zdim: xr.DataArray(zi, dims="points"),
            ndim: xr.DataArray(node_ids, dims=("points", "nodes")),
        }

        node_data = field.data.isel(selection_dict, ignore_grid=True)
        value = (node_data * bcoords).sum("nodes").data  # Barycentric interpolation in the lateral direction
        return value.compute() if is_dask_collection(value) else value


class UxLinearNodeLinearZF(ScalarInterpolator):
    """Piecewise linear interpolation kernel for node registered data located at vertical interface levels (zf points)."""

    def interp(
        self,
        particle_positions: dict[str, float | np.ndarray],
        grid_positions: dict[_UXGRID_AXES, dict[str, int | float | np.ndarray]],
        field: Field,
    ):
        """
        Piecewise linear interpolation kernel for node registered data located at vertical interface levels (zf points).
        Effectively, it applies barycentric interpolation in the lateral direction
        and piecewise linear interpolation in the vertical direction.
        """
        ti, zi, fi = np.broadcast_arrays(
            grid_positions["T"]["index"], grid_positions["Z"]["index"], grid_positions["FACE"]["index"]
        )
        zeta = grid_positions["Z"]["bcoord"]
        bcoords = xr.DataArray(grid_positions["FACE"]["bcoord"], dims=("points", "nodes"))
        node_ids = field.grid.uxgrid.face_node_connectivity[fi, :].values

        tdim, zdim, ndim = field.data.dims

        def _zsample(z_index):
            """Barycentric (lateral) interpolation of the node values at a single vertical interface level."""
            selection_dict = {
                tdim: xr.DataArray(ti, dims="points"),
                zdim: xr.DataArray(z_index, dims="points"),
                ndim: xr.DataArray(node_ids, dims=("points", "nodes")),
            }
            # Reduce over the "nodes" dimension by name so the result is independent of ``isel`` dim order.
            node_data = field.data.isel(selection_dict, ignore_grid=True)
            return (node_data * bcoords).sum("nodes").data

        # The zi refers to the vertical layer index. The field in this routine are assumed to be defined at the vertical interface levels.
        # For interface zi, the interface indices are [zi, zi+1], so we need to use the values at zi and zi+1.
        # First, do barycentric interpolation in the lateral direction for each interface level
        fzk = _zsample(zi)
        fzkp1 = _zsample(zi + 1)

        # Then, do piecewise linear interpolation in the vertical direction
        value = fzk * (1 - zeta) + fzkp1 * zeta
        return value.compute() if is_dask_collection(value) else value


class Ux_Velocity(VectorInterpolator):  # noqa: N801
    """Interpolation kernel for Vectorfields of velocity on a UxGrid."""

    def interp(
        self,
        particle_positions: dict[str, float | np.ndarray],
        grid_positions: dict[_UXGRID_AXES, dict[str, int | float | np.ndarray]],
        vectorfield: VectorField,
    ):
        u = vectorfield.U.interp_method.interp(particle_positions, grid_positions, vectorfield.U)
        v = vectorfield.V.interp_method.interp(particle_positions, grid_positions, vectorfield.V)
        if vectorfield.grid._mesh.is_spherical():
            u /= vectorfield.grid.deg2m * np.cos(np.deg2rad(particle_positions["y"]))
            v /= vectorfield.grid.deg2m

        if "3D" in vectorfield.vector_type:
            w = vectorfield.W.interp_method.interp(particle_positions, grid_positions, vectorfield.W)
        else:
            w = np.zeros_like(u)
        return u, v, w
