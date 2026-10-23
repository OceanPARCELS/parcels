from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np
import xarray as xr
from dask import is_dask_collection

from parcels._core.basegrid import BaseGrid
from parcels._core.index_search import GRID_SEARCH_ERROR, _search_1d_array, _search_1d_columns, uxgrid_point_in_cell
from parcels._core.mesh import BaseMesh, TMesh
from parcels._core.warnings import FieldSetWarning

if TYPE_CHECKING:
    import uxarray as ux

_UXGRID_AXES = Literal["Z", "FACE"]


class UxGrid(BaseGrid):
    """
    Extension of uxarray's Grid class that supports point-location search
    for interpolation on unstructured grids.
    """

    def __init__(self, grid: ux.grid.Grid, z: ux.UxDataArray, mesh: TMesh) -> None:
        """
        Initializes the UxGrid with a uxarray grid and vertical coordinate array.

        Parameters
        ----------
        grid : ux.grid.Grid
            The uxarray grid object containing the unstructured grid data.
        z : ux.UxDataArray
            Vertical coordinates (depths) of the layer interface heights (not the mid-layer depths). Either a 1D array,
            constant in time and space (flat bottom topography, no moving vertical grid), or a 3D array with dims
            ("time", "zf", "n_node"), varying in time and across the mesh nodes.
        mesh : FlatMesh or SphericalMesh
            The mesh of the grid, e.g. ``FlatMesh()`` or ``SphericalMesh(radius=...)``.
        """
        if not isinstance(mesh, BaseMesh):
            raise TypeError(f"mesh must be a FlatMesh or SphericalMesh object. Got {mesh=!r}")
        import uxarray as ux

        if grid.n_max_face_nodes > 3:
            raise ValueError("Provided ux.grid.Grid must contain only triangular cells (n_max_face_nodes=3)")
        self.uxgrid = grid
        if not isinstance(z, ux.UxDataArray):
            raise TypeError("z must be an instance of ux.UxDataArray")
        if z.ndim not in (1, 3):
            raise ValueError(f"z must be a 1D or 3D array of vertical coordinates, got {z.ndim}D")
        if z.ndim == 3 and z.dims != ("time", "zf", "n_node"):
            raise ValueError(f"A 3D z must have dims ('time', 'zf', 'n_node'), got {z.dims}")
        if z.ndim == 3:
            warnings.warn(
                "Time-varying (3D) z coordinates are experimental and may cause significant memory overhead that "
                f"leads to OOM errors. This z coordinate has sizes {dict(z.sizes)} ({z.nbytes / 1e9:.3g} GB). "
                "Assumptions: z is defined at the layer interfaces ('zf') on the mesh nodes ('n_node') and is strictly "
                "increasing along 'zf'; each particle's z column is interpolated barycentrically from its face's "
                "nodes and linearly in time between z snapshots.",
                FieldSetWarning,
                stacklevel=4,
            )
        self.z = z
        self.mesh = mesh
        self._spatialhash = None
        if is_dask_collection(self.uxgrid.face_node_connectivity.data):
            self.uxgrid.face_node_connectivity.load()

    @property
    def depth(self):
        """
        Notes
        -----
        Included for compatibility with v3 codebase. May be removed in future.
        TODO v4: Evaluate
        """
        try:
            _ = self.z.values
        except KeyError:
            return np.zeros(1)
        return self.z.values

    @property
    def axes(self) -> list[_UXGRID_AXES]:
        return ["Z", "FACE"]

    def get_axis_dim(self, axis: _UXGRID_AXES) -> int:
        if axis not in self.axes:
            raise ValueError(f"Axis {axis!r} is not part of this grid. Available axes: {self.axes}")

        if axis == "Z":
            if self.z.ndim == 3:
                return self.z.sizes["zf"]
            return len(self.z.values)
        elif axis == "FACE":
            return self.uxgrid.n_face

    @property
    def deg2m(self) -> float:
        """Metres per arcdegree for this grid's mesh."""
        if self.mesh.is_spherical():
            return self.mesh.deg2m
        return 1.0

    def search(self, z, y, x, ei=None, ti=None, tau=None, tol=1e-6):
        """
        Search for the grid cell (face) and vertical layer that contains the given points.

        Parameters
        ----------
        z : float or np.ndarray
            The vertical coordinate(s) (depth) of the point(s).
        y : float or np.ndarray
            The latitude(s) of the point(s).
        x : float or np.ndarray
            The longitude(s) of the point(s).
        ei : np.ndarray, optional
            Precomputed horizontal indices (face indices) for the points.

            TO BE IMPLEMENTED : If provided, we'll check
            if the points are within the faces specified by these indices. For cells where the particles
            are not found, a nearest neighbor search will be performed. As a last resort, the spatial hash will be used.
        ti : np.ndarray, optional
            Time index of each point, as returned by ``_search_time_index``. Required when z is 3D; selects the
            earlier of the two z snapshots (``ti`` and ``ti + 1``) that are linearly interpolated with ``tau``.
        tau : np.ndarray, optional
            Barycentric time coordinate of each point, as returned by ``_search_time_index``. Required when z is 3D;
            used for linear interpolation of the z coordinate for the construction of a particle's column.
        tol : float, optional
            Tolerance for barycentric coordinate checks. Default is 1e-6.
        """
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        z = np.asarray(z, dtype=np.float32)

        if np.any(ei):
            indices = self.unravel_index(ei)
            fi = indices.get("FACE")
            is_in_cell, coords = uxgrid_point_in_cell(self, y, x, fi, fi)
            y_check = y[is_in_cell == 0]
            x_check = x[is_in_cell == 0]
            zero_indices = np.where(is_in_cell == 0)[0]
        else:
            # Otherwise, we need to check all points
            fi = np.full(len(y), GRID_SEARCH_ERROR, dtype=np.int32)
            y_check = y
            x_check = x
            coords = -1.0 * np.ones((len(y), 3), dtype=np.float32)
            zero_indices = np.arange(len(y))

        if len(zero_indices) > 0:
            _, face_ids_q, coords_q = self.get_spatial_hash().query(y_check, x_check)
            coords[zero_indices, :] = coords_q
            fi[zero_indices] = face_ids_q

        if self.z.ndim == 3:
            zi, zeta = self._search_time_varying_z(z, fi, coords, ti, tau)
        else:
            zi, zeta = _search_1d_array(self.z.values, z)

        return {"Z": {"index": zi, "bcoord": zeta}, "FACE": {"index": fi, "bcoord": coords}}

    def _search_time_varying_z(self, z, fi, coords, ti, tau):
        """Vertical search of each particle in the z column interpolated barycentrically from the nodes of its face,
        using temporal linear interpolation between the snapshots ti and ti + 1.
        """
        found = fi >= 0
        cols_ti = self.z.isel(
            time=xr.DataArray(np.broadcast_to(ti, fi.shape)[found], dims="points"),
            n_node=xr.DataArray(self.uxgrid.face_node_connectivity[fi[found], :].values, dims=("points", "nodes")),
            ignore_grid=True,
        ).compute()

        if self.z.shape[0] == 1:
            node_columns = cols_ti
        else:
            cols_tnext = self.z.isel(
                time=xr.DataArray(np.broadcast_to(ti + 1, fi.shape)[found], dims="points"),
                n_node=xr.DataArray(self.uxgrid.face_node_connectivity[fi[found], :].values, dims=("points", "nodes")),
                ignore_grid=True,
            ).compute()

            tau_points = xr.DataArray(np.broadcast_to(tau, fi.shape)[found], dims="points")
            node_columns = (1 - tau_points) * cols_ti + tau_points * cols_tnext

        bcoords = xr.DataArray(coords[found], dims=("points", "nodes"))
        # Particles outside the mesh are given a NaN z column for vertical searching
        columns = np.full((fi.size, self.z.sizes["zf"]), np.nan)
        columns[found] = xr.dot(node_columns, bcoords, dim="nodes").values
        zi, zeta = _search_1d_columns(columns, z)

        # Particles outside the mesh are given a 0 vertical index and a NaN vertical barycentric coordinate
        return np.where(found, zi, 0), np.where(found, zeta, np.nan)
