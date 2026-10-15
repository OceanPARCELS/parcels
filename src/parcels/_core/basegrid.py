"""Module defining the BaseGrid class for XGrid and UXGrid"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

from parcels._core.mesh import FlatMesh, SphericalMesh
from parcels._core.spatialhash import SpatialHash

if TYPE_CHECKING:
    import numpy as np


class GridType(IntEnum):
    RectilinearZGrid = 0
    RectilinearSGrid = 1
    CurvilinearZGrid = 2
    CurvilinearSGrid = 3


class BaseGrid(ABC):
    """Base class for parcels.XGrid and parcels.UxGrid defining common methods and properties"""

    _spatialhash: SpatialHash | None
    _mesh: FlatMesh | SphericalMesh

    @abstractmethod
    def search(
        self, z: float, y: float, x: float, ei=None, ti=None, tau=None
    ) -> dict[str, tuple[int, float | np.ndarray]]:
        """
        Perform a spatial (and optionally vertical) search to locate the grid element
        that contains a given point (x, y, z).

        This method delegates to grid-type-specific logic (e.g., structured or unstructured)
        to determine the appropriate indices and barycentric coordinates for evaluating a field.

        Parameters
        ----------
        z : float
            Vertical coordinate of the query point. If `search2D=True`, this may be ignored.
        y : float
            Latitude or vertical index, depending on grid type and projection.
        x : float
            Longitude or horizontal index, depending on grid type and projection.
        ei : int, optional
            A previously computed encoded index (e.g., raveled face or cell index). If provided,
            the search will first attempt to validate and reuse it before falling back to
            a global or local search strategy.
        ti : np.ndarray, optional
            Time index of each query point, as returned by ``_search_time_index``.
        tau : np.ndarray, optional
            Barycentric time coordinate of each query point, as returned by ``_search_time_index``.
        search2D : bool, default=False
            If True, perform only a 2D search (x, y), ignoring the vertical component z.

        Returns
        -------
        dict
            A dictionary mapping spatial axis names to tuples of (index, barycentric_coordinates).
            The returned axes depend on the grid dimensionality and type:

            - 3D structured grid: {"Z": (zi, zeta), "Y": (yi, eta), "X": (xi, xsi)}
            - 2D structured grid: {"Y": (yi, eta), "X": (xi, xsi)}
            - 1D structured grid (depth): {"Z": (zi, zeta)}
            - Unstructured grid: {"Z": (zi, zeta), "FACE": (fi, bcoords)}

            Where:

            - index (int): The cell position of the particles along the given axis
            - barycentric_coordinates (float or np.ndarray): The coordinates defining
              the particles positions within the grid cell. For structured grids, this
              is a single coordinate per axis; for unstructured grids, this can be
              an array of coordinates for the face polygon.

        Raises
        ------
        FieldOutOfBoundError
            Raised when the queried point lies outside the bounds of the grid.
        NotImplementedError
            Raised if the search method is not implemented for the current grid type.
        """
        ...

    def ravel_index(self, axis_indices: dict[str, np.ndarray]) -> np.ndarray:
        """
        Convert a dictionary of axis indices to a single encoded index (ei).

        This method takes the individual indices for each spatial axis and combines them
        into a single integer that uniquely identifies a grid cell. This encoded
        index can be used for efficient caching and lookup operations.

        Parameters
        ----------
        axis_indices : dict[str, np.ndarray(int)]
            A dictionary mapping axis names to their corresponding indices.
            The expected keys depend on the grid dimensionality and type:

            - 3D structured grid: {"Z": zi, "Y": yi, "X": xi}
            - 2D structured grid: {"Y": yi, "X": xi}
            - 1D structured grid: {"Z": zi}
            - Unstructured grid: {"Z": zi, "FACE": fi}

        Returns
        -------
        np.ndarray(int)
            The encoded indices (ei) representing the unique grid cells or faces.

        Raises
        ------
        KeyError
            Raised when required axis keys are missing from axis_indices.
        ValueError
            Raised when index values are out of bounds for the grid.
        NotImplementedError
            Raised if the method is not implemented for the current grid type.
        """
        dims = np.array([self.get_axis_dim(axis) for axis in self.axes], dtype=int)
        indices = np.array([axis_indices[axis] for axis in self.axes], dtype=int)
        return _ravel(dims, indices)

    def unravel_index(self, ei: int) -> dict[str, np.ndarray]:
        """
        Convert a single encoded index (ei) back to a dictionary of axis indices.

        This method is the inverse of ravel_index, taking an encoded index and
        decomposing it back into the individual indices for each spatial axis.

        Parameters
        ----------
        ei : int
            The encoded index representing a unique grid cell or face.

        Returns
        -------
        dict[str, int]
            A dictionary mapping axis names to their corresponding indices.
            The returned keys depend on the grid dimensionality and type:

            - 3D structured grid: {"Z": zi, "Y": yi, "X": xi}
            - 2D structured grid: {"Y": yi, "X": xi}
            - 1D structured grid: {"Z": zi}
            - Unstructured grid: {"Z": zi, "FACE": fi}

        Raises
        ------
        ValueError
            Raised when the encoded index is out of bounds or invalid for the grid.
        NotImplementedError
            Raised if the method is not implemented for the current grid type.
        """
        dims = np.array([self.get_axis_dim(axis) for axis in self.axes], dtype=int)
        indices = _unravel(dims, ei)
        return dict(zip(self.axes, indices, strict=True))

    @property
    @abstractmethod
    def axes(self) -> list[str]:
        """
        Return a list of axis names that are part of this grid.

        This list must at least be of length 1, and `get_axis_dim` should
        return a valid integer for each axis name in the list.

        Returns
        -------
        list[str]
            List of axis names, e.g. ["Z", "Y", "X"] for a 3D structured grid or ["Z", "FACE"] for an unstructured grid.
        """
        ...

    @abstractmethod
    def get_axis_dim(self, axis: str) -> int:
        """
        Return the dimensionality (number of cells/faces) along a specific axis.

        Parameters
        ----------
        axis : str
            The name of the axis to get the dimensionality for. Must be one of the values returned by self.axes.

        Returns
        -------
        int
            The number of cells/edges along the specified axis.

        Raises
        ------
        ValueError
            If the specified axis is not part of this grid.
        """
        ...

    def get_spatial_hash(
        self,
        reconstruct=False,
    ):
        """Get the SpatialHash data structure of this Grid that allows for
        fast face search queries. Face searches are used to find the faces that
        a list of points, in spherical coordinates, are contained within.

        Parameters
        ----------
        global_grid : bool, default=False
            If true, the hash grid is constructed using the domain [-pi,pi] x [-pi,pi]
        reconstruct : bool, default=False
            If true, reconstructs the spatial hash

        Returns
        -------
        self._spatialhash : parcels.spatialhash.SpatialHash
            SpatialHash instance

        """
        if self._spatialhash is None or reconstruct:
            self._spatialhash = SpatialHash(self)

        return self._spatialhash


def _unravel(dims, ei):
    """
    Converts a flattened (raveled) index back to multi-dimensional indices.

    Args:
        dims (1d-array-like): The dimensions along each axis
        ei (int): The flattened index to convert

    Returns
    -------
        array-like: Indices along each axis corresponding to the given flattened index

    Example:
        >>> dims = [2, 3, 4]
        >>> ei = 9
        >>> unravel(dims, ei)
        array([0, 2, 1])
        # Calculation:
        # i0 = 9 // (3*4) = 9 // 12 = 0
        # remainder = 9 % 12 = 9
        # i1 = 9 // 4 = 2
        # i2 = 9 % 4 = 1
    """
    strides = np.cumprod(dims[::-1])[::-1]

    indices = np.empty((len(dims), len(ei)), dtype=int)

    for i in range(len(dims) - 1):
        indices[i, :] = ei // strides[i + 1]
        ei = ei % strides[i + 1]

    indices[-1, :] = ei
    return indices


def _ravel(dims, indices):
    """
    Converts indices to a flattened (raveled) index.

    Args:
        dims (1d-array-like): The dimensions along each axis
        indices (array-like): Indices along each axis to convert

    Returns
    -------
        int: The flattened index corresponding to the given indices

    Example:
        >>> dims = [2, 3, 4]
        >>> indices = [0, 2, 1]
        >>> ravel(dims, indices)
        9
        # Calculation: 0 * (3 * 4) + 2 * (4) + 1 = 0 + 8 + 1 = 9
    """
    strides = np.cumprod(dims[::-1])[::-1]
    ei = 0
    for i in range(len(dims) - 1):
        ei += indices[i] * strides[i + 1]

    return ei + indices[-1]
