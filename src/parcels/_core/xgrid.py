import warnings
from collections.abc import Hashable, Sequence
from functools import cached_property
from typing import Literal, cast

import numpy as np
import numpy.typing as npt
import xarray as xr
from dask import is_dask_collection

import parcels._sgrid as sgrid
import parcels._typing as ptyping
from parcels._core.basegrid import BaseGrid
from parcels._core.index_search import _search_1d_array, _search_1d_columns, _search_indices_curvilinear_2d
from parcels._core.mesh import SphericalMesh, get_mesh
from parcels._core.warnings import FieldSetWarning
from parcels._sgrid.accessor import _get_dim_to_axis_mapping, get_dim_position
from parcels.interpolators._xinterpolators import _get_offsets_dictionary

_FIELD_DATA_ORDERING: Sequence[ptyping.XgcmAxisDirection] = "TZYX"
_XGRID_AXES_ORDERING: Sequence[ptyping.XgridAxis] = "ZYX"


def get_cell_count_along_dim(ds: xr.Dataset, fnp: sgrid.FaceNodePadding) -> int:
    if fnp.face in ds.dims:
        return ds.sizes[fnp.face] - 1
    return sgrid.get_n_faces(ds.sizes[fnp.node], fnp.padding) - 1


def get_time(ds: xr.Dataset, time_dim: str) -> npt.NDArray:
    return ds[time_dim].values


def _get_xgrid_axes(metadata: sgrid.SGrid2DMetadata, ds_dims: set[str]) -> list[ptyping.XgridAxis]:
    dim_to_axis = _get_dim_to_axis_mapping(metadata)
    present = {axis for dim, axis in dim_to_axis.items() if dim in ds_dims}
    return sorted(present, key=_XGRID_AXES_ORDERING.index)


def _drop_field_data(ds: xr.Dataset) -> xr.Dataset:
    """
    Removes DataArrays from the dataset that are associated with field data so that
    when passed to the XGCM grid, the object only functions as an in memory representation
    of the grid.
    """
    return ds.drop_vars(set(ds.data_vars) - {"grid"})  # don't drop sgrid metadata


def assert_all_field_dims_have_axis(da: xr.DataArray, metadata: sgrid.SGrid2DMetadata) -> None:
    dim_to_axis = _get_dim_to_axis_mapping(metadata) | {"time": "T"}
    ax_dims = [(dim_to_axis.get(str(dim)), str(dim)) for dim in da.dims]

    for ax, dim_name in ax_dims:
        if ax is None:
            raise ValueError(
                f'Dimension "{dim_name}" has no axis attribute. '
                f'HINT: You may want to add an {{"axis": A}} to your DataSet["{dim_name}"], where A is one of "X", "Y", "Z" or "T"'
            )

    seen_axes: dict[str, str] = {}
    for ax, dim_name in ax_dims:
        if ax in seen_axes:
            raise ValueError(
                f"Two dimensions ({dim_name!r} and {seen_axes[ax]!r}) provide values in the axis direction {ax!r}. "
                "This is not possible, a field cannot have two dimensions on a single axis."
            )
        seen_axes[ax] = dim_name
    assert len(ax_dims) <= 4, (
        "The input dataset appears to have more than 4 dimensions after conversion. Execution should never reach this point. Please file an issue sharing more about your input dataset."
    )
    return


def _transpose_xfield_data_to_tzyx(da: xr.DataArray, sgrid_metadata: sgrid.SGrid2DMetadata) -> xr.DataArray:
    """
    Transpose a DataArray of any shape into a 4D array of order TZYX. Uses SGRID metadata to determine
    the axes, and inserts mock dimensions of size 1 for any axes not present in the DataArray.
    """
    dim_to_axis = _get_dim_to_axis_mapping(sgrid_metadata) | {"time": "T"}

    # filter to only dims in da
    dim_to_axis = {dim: axis for dim, axis in dim_to_axis.items() if dim in da.dims}

    if dim_to_axis == {}:
        # Assuming its a 1D constant field (hence has no axes)
        assert da.shape == (1, 1, 1, 1)
        return da.rename({old_dim: f"mock{axis}" for old_dim, axis in zip(da.dims, _FIELD_DATA_ORDERING, strict=True)})

    # All dimensions must be associated with an axis in the grid
    if set(dim_to_axis) != set(da.dims):
        dims_not_on_grid = set(da.dims) - set(dim_to_axis)
        raise ValueError(
            f"DataArray {da.name!r} with dims {da.dims} has dimensions {dims_not_on_grid} that are not associated with a direction on the provided grid."
        )

    axes_not_in_field = set(_FIELD_DATA_ORDERING).difference(set(dim_to_axis.values()))

    mock_dims_to_create = {}
    for ax in axes_not_in_field:
        mock_dims_to_create[f"mock{ax}"] = 1
        dim_to_axis[f"mock{ax}"] = ax

    if mock_dims_to_create:
        da = da.expand_dims(mock_dims_to_create, create_index_for_new_dim=False)

    ax_dims = sorted(dim_to_axis.items(), key=lambda x: _FIELD_DATA_ORDERING.index(x[1]))

    return da.transpose(*[ax_dim[0] for ax_dim in ax_dims])


class XGrid(BaseGrid):
    """
    Class to represent a structured grid in Parcels.

    This class provides methods and properties required for indexing and interpolating on the grid.
    Grid topology is derived directly from SGRID metadata attached to the dataset.

    Assumptions:
    - If using Parcels in the context of a spatially periodic simulation, the provided grid already has a halo

    """

    def __init__(self, model_data: xr.Dataset, mesh: Literal["flat", "spherical"] | SphericalMesh):
        self.sgrid_metadata = model_data.sgrid.metadata
        self._ds = model_data
        self._mesh = get_mesh(mesh)
        self._spatialhash = None
        ds = model_data

        # Set the coordinates for the dataset (needed to be done explicitly for curvilinear grids)
        if "lon" in ds:
            ds.set_coords("lon")
        if "lat" in ds:
            ds.set_coords("lat")

        axes = self.axes
        if len(set(axes) & {"X", "Y"}) > 0:  # Only if spatial grid is >0D (see #2054 for further development)
            assert_valid_lat_lon(ds["lat"], ds["lon"], self.sgrid_metadata)

        if "Z" in axes:
            depth = ds["depth"]
            if depth.ndim not in (1, 4):
                raise ValueError(f"depth must be a 1D or 4D array of vertical coordinates, got {depth.ndim}D")
            if depth.ndim == 4:
                assert self.sgrid_metadata.vertical_dimensions is not None
                vertical_dim = self.sgrid_metadata.vertical_dimensions[0].node
                fnp_x, fnp_y = self.sgrid_metadata.face_dimensions
                expected_dims = ("time", vertical_dim, fnp_y.face, fnp_x.face)
                if depth.dims != expected_dims:
                    raise ValueError(f"A 4D depth must have dims {expected_dims}, got {depth.dims}")
                warnings.warn(
                    "Time-varying (4D) z coordinates are experimental and may cause significant memory overhead that "
                    f"leads to OOM errors. This z coordinate has sizes {dict(depth.sizes)} ({depth.nbytes / 1e9:.3g} GB). "
                    f"Assumptions: z has dims {expected_dims}, i.e. it is "
                    f"defined at the layer interfaces on the cell centres, and is strictly increasing along "
                    f"{vertical_dim!r}; each particle uses the z column at the centre of its grid cell, so z is constant "
                    "within a cell and jumps between cells, it is linearly interpolated in time between z snapshots.",
                    FieldSetWarning,
                    stacklevel=5,
                )
            else:
                assert_valid_depth(depth)

        self._ds = ds

    # def __repr__(self):
    #     return xgrid_repr(self)

    @property
    def axes(self) -> list[ptyping.XgridAxis]:
        return _get_xgrid_axes(self.sgrid_metadata, set(self._ds.dims))

    @property
    def lon(self):
        """
        Notes
        -----
        Included for compatibility with v3 codebase. May be removed in future.
        TODO v4: Evaluate
        """
        if "X" not in self.axes:
            return np.zeros(1)
        # ensure lon is loaded into memory for dask-backed datasets, as it is used in the search method
        if is_dask_collection(self._ds["lon"].data):
            self._ds["lon"].load()
        return self._ds["lon"].values

    @property
    def lat(self):
        """
        Notes
        -----
        Included for compatibility with v3 codebase. May be removed in future.
        TODO v4: Evaluate
        """
        if "Y" not in self.axes:
            return np.zeros(1)
        # ensure lat is loaded into memory for dask-backed datasets, as it is used in the search method
        if is_dask_collection(self._ds["lat"].data):
            self._ds["lat"].load()
        return self._ds["lat"].values

    @property
    def depth(self):
        """
        Notes
        -----
        Included for compatibility with v3 codebase. May be removed in future.
        TODO v4: Evaluate
        """
        if "Z" not in self.axes:
            return np.zeros(1)
        return self._ds["depth"].values

    @property
    def _datetimes(self):
        if "time" not in self._ds.dims:
            return np.zeros(1)
        return get_time(self._ds, "time")

    @property
    def time(self):
        return self._datetimes.astype(np.float64) / 1e9

    @property
    def deg2m(self) -> float:
        """Metres per degree of arc for this grid's mesh."""
        if self._mesh.is_spherical():
            return self._mesh.deg2m
        return 1.0

    @cached_property
    def xdim(self) -> int:
        return self.get_axis_dim("X")

    @cached_property
    def ydim(self) -> int:
        return self.get_axis_dim("Y")

    @cached_property
    def zdim(self) -> int:
        return self.get_axis_dim("Z")

    def get_axis_dim(self, axis: ptyping.XgridAxis) -> int:
        if axis not in self.axes:
            raise ValueError(f"Axis {axis!r} is not part of this grid. Available axes: {self.axes}")

        fnp_x, fnp_y = self.sgrid_metadata.face_dimensions
        if axis == "X":
            return get_cell_count_along_dim(self._ds, fnp_x)
        if axis == "Y":
            return get_cell_count_along_dim(self._ds, fnp_y)
        # axis == "Z"
        assert self.sgrid_metadata.vertical_dimensions is not None
        return get_cell_count_along_dim(self._ds, self.sgrid_metadata.vertical_dimensions[0])

    def localize(
        self, position: dict[ptyping.XgridAxis, tuple[int, float]], dims: list[str]
    ) -> dict[str, tuple[int, float]]:
        """
        Uses the grid context (i.e., the staggering of the grid) to convert a position relative
        to the f-points in the grid to a position relative to the staggered grid the array
        of interest is defined on.

        Uses dimensions of the DataArray to determine the staggered grid.

        WARNING: This API is unstable and subject to change in future versions.

        Parameters
        ----------
        position : dict
            A mapping of the axis to a tuple of (index, barycentric coordinate) for the
            f-points in the grid.
        dims : list[str]
            A list of dimension names that the DataArray is defined on. This is used to determine
            the staggering of the grid and which axis each dimension corresponds to.

        Returns
        -------
        dict[str, tuple[int, float]]
            A mapping of the dimension names to a tuple of (index, barycentric coordinate) for
            the staggered grid the DataArray is defined on.

        Example
        -------
        >>> position = {'X': (5, 0.51), 'Y': (
            10, 0.25), 'Z': (3, 0.75)}
        >>> dims = ['time', 'depth', 'YC', 'XC']
        >>> grid.localize(position, dims)
        {'depth': (3, 0.75), 'YC': (9, 0.75), 'XC': (5, 0.01)}
        """
        dim_to_axis = _get_dim_to_axis_mapping(self.sgrid_metadata)
        axis_to_var = {dim_to_axis[dim]: dim for dim in dims if dim in dim_to_axis}
        var_positions = {
            axis: get_dim_position(self.sgrid_metadata, dim) for axis, dim in axis_to_var.items() if axis != "T"
        }
        return {
            axis_to_var[axis]: _convert_center_pos_to_fpoint(
                index=index,
                bcoord=bcoord,
                position=var_positions[axis],
                f_point_position=self._fpoint_info[axis],
            )
            for axis, (index, bcoord) in position.items()
        }

    @property
    def _z4d(self) -> Literal[0, 1]:
        """
        Notes
        -----
        Included for compatibility with v3 codebase. May be removed in future.
        TODO v4: Evaluate
        """
        return 1 if self.depth.shape == 4 else 0

    @property
    def zonal_periodic(self): ...  # ? hmmm, from v3, do we still need this?

    @property
    def _gtype(self):
        """This class is created *purely* for compatibility with v3 code and will be removed
        or changed in future.

        TODO: Remove
        """
        from parcels._core.basegrid import GridType

        if len(self.lon.shape) <= 1:
            if self.depth is None or len(self.depth.shape) <= 1:
                return GridType.RectilinearZGrid
            else:
                return GridType.RectilinearSGrid
        else:
            if self.depth is None or len(self.depth.shape) <= 1:
                return GridType.CurvilinearZGrid
            else:
                return GridType.CurvilinearSGrid

    def search(self, z, y, x, ei=None, ti=None, tau=None):
        ds = self._ds

        if "X" not in self.axes or ds.lon.ndim == 1:
            if "Y" in self.axes:
                yi, eta = _search_1d_array(ds.lat.values, y)
            else:
                yi, eta = np.zeros(y.shape, dtype=int), np.zeros(y.shape, dtype=float)

            if "X" in self.axes:
                xi, xsi = _search_1d_array(ds.lon.values, x)
            else:
                xi, xsi = np.zeros(x.shape, dtype=int), np.zeros(x.shape, dtype=float)
        elif ds.lon.ndim == 2:
            yi, xi = None, None
            if ei is not None:
                axis_indices = self.unravel_index(ei)
                xi = axis_indices.get("X")
                yi = axis_indices.get("Y")

            yi, eta, xi, xsi = _search_indices_curvilinear_2d(self, y, x, yi, xi)
        else:
            raise NotImplementedError("Searching in >2D lon/lat arrays is not implemented yet.")

        if "Z" in self.axes and ds.depth.ndim == 4:
            zi, zeta = self._search_time_varying_depth(z, yi, xi, ti, tau)
        elif "Z" in self.axes:
            zi, zeta = _search_1d_array(ds.depth.values, z)
        else:
            zi, zeta = np.zeros(z.shape, dtype=int), np.zeros(z.shape, dtype=float)

        return {
            "Z": {"index": zi, "bcoord": zeta},
            "Y": {"index": yi, "bcoord": eta},
            "X": {"index": xi, "bcoord": xsi},
        }

    def _search_time_varying_depth(self, z, yi, xi, ti, tau):
        """Vertical search of each particle in the z column at the centre of its cell using temporal linear
        interpolation between the snapshots ti and ti + 1.
        """
        found = (yi >= 0) & (xi >= 0)
        offsets = _get_offsets_dictionary(self)
        yi_centre = yi[found] + offsets["Y"]
        xi_centre = xi[found] + offsets["X"]

        depth = self._ds["depth"]
        fnp_x, fnp_y = self.sgrid_metadata.face_dimensions
        cell_centres = {
            fnp_y.face: xr.DataArray(yi_centre, dims="points"),
            fnp_x.face: xr.DataArray(xi_centre, dims="points"),
        }

        ti_found = np.broadcast_to(ti, yi.shape)[found]
        columns_ti = depth.isel({"time": xr.DataArray(ti_found, dims="points"), **cell_centres})

        if depth.sizes["time"] == 1:
            cell_columns = columns_ti
        else:
            columns_tnext = depth.isel({"time": xr.DataArray(ti_found + 1, dims="points"), **cell_centres})
            tau_found = xr.DataArray(np.broadcast_to(tau, yi.shape)[found], dims="points")
            cell_columns = (1 - tau_found) * columns_ti + tau_found * columns_tnext

        cell_columns = cell_columns.transpose("points", ...)

        # Particles outside the grid are given a NaN z column for vertical searching
        columns = np.full((yi.size, cell_columns.shape[1]), np.nan)
        columns[found] = cell_columns.values
        zi, zeta = _search_1d_columns(columns, z)

        # Particles outside the grid are given a 0 vertical index and a NaN vertical barycentric coordinate
        return np.where(found, zi, 0), np.where(found, zeta, np.nan)

    @cached_property
    def _fpoint_info(self) -> dict[ptyping.XgridAxis, sgrid.Padding]:
        """Returns a mapping of the spatial axes in the Grid to their Padding values (node positions)."""
        metadata = self.sgrid_metadata
        fnp_x, fnp_y = metadata.face_dimensions
        result: dict[ptyping.XgridAxis, sgrid.Padding] = {}
        axes = self.axes
        if "X" in axes:
            result["X"] = fnp_x.padding
        if "Y" in axes:
            result["Y"] = fnp_y.padding
        if "Z" in axes and metadata.vertical_dimensions:
            result["Z"] = metadata.vertical_dimensions[0].padding
        return result

    def get_axis_dim_mapping(self, dims: Sequence[Hashable]) -> dict[ptyping.XgridAxis, str]:
        """
        Maps xarray dimension names to their corresponding axis (X, Y, Z).

        WARNING: This API is unstable and subject to change in future versions.

        Parameters
        ----------
        dims : Sequence[Hashable]
            Sequence of xarray dimension names

        Returns
        -------
        dict[_XGRID_AXES, str]
            Dictionary mapping axes (X, Y, Z) to their corresponding dimension names

        Examples
        --------
        >>> grid.get_axis_dim_mapping(['time', 'lat', 'lon'])
        {'Y': 'lat', 'X': 'lon'}

        Notes
        -----
        Only returns mappings for spatial axes (X, Y, Z) that are present in the grid.
        """
        dim_to_axis = _get_dim_to_axis_mapping(self.sgrid_metadata)
        result = {}
        for dim in dims:
            axis = dim_to_axis.get(str(dim))
            if axis in self.axes:
                result[cast(ptyping.XgridAxis, axis)] = str(dim)
        return result


def get_axis_from_dim_name(metadata: sgrid.SGrid2DMetadata, dim: Hashable) -> ptyping.XgcmAxisDirection | None:
    """For a given dimension name in a grid, returns the direction axis it is on."""
    dim_to_axis = _get_dim_to_axis_mapping(metadata) | {"time": "T"}
    return dim_to_axis.get(str(dim))


def get_xgcm_position_from_dim_name(metadata: sgrid.SGrid2DMetadata, dim: str) -> ptyping.GridPosition | None:
    """For a given dimension, returns the GridPosition of the variable in the grid."""
    try:
        return get_dim_position(metadata, dim)
    except ValueError:
        return None


def assert_all_dimensions_correspond_with_axis(da: xr.DataArray, metadata: sgrid.SGrid2DMetadata) -> None:
    dim_to_axis = _get_dim_to_axis_mapping(metadata)
    for dim in da.dims:
        if dim not in dim_to_axis:
            raise ValueError(
                f"Dimension {dim!r} for DataArray {da.name!r} with dims {da.dims} is not associated with a direction on the provided grid."
            )


def assert_valid_field_array(da: xr.DataArray, metadata: sgrid.SGrid2DMetadata):
    """
    Asserts that for a data array:
    - All dimensions are associated with a direction on the grid
    - These directions are T, Z, Y, X and the array is ordered as T, Z, Y, X
    """
    dim_to_axis = _get_dim_to_axis_mapping(metadata) | {"time": "T"}

    for dim in da.dims:
        if dim not in dim_to_axis:
            raise ValueError(
                f"Dimension {dim!r} for DataArray {da.name!r} with dims {da.dims} is not associated with a direction on the provided grid."
            )

    dim_to_axis_for_da = {dim: dim_to_axis[dim] for dim in da.dims}
    dim_to_axis_for_da = cast(dict[Hashable, ptyping.XgcmAxisDirection], dim_to_axis_for_da)

    # Assert all dimensions are present
    if set(dim_to_axis_for_da.values()) != {"T", "Z", "Y", "X"}:
        raise ValueError(
            f"DataArray {da.name!r} with dims {da.dims} has directions {tuple(dim_to_axis_for_da.values())}."
            "Expected directions of 'T', 'Z', 'Y', and 'X'."
        )

    # Assert order is t, z, y, x
    if list(dim_to_axis_for_da.values()) != ["T", "Z", "Y", "X"]:
        raise ValueError(
            f"Dimension order for array {da.name!r} is not valid. Got {tuple(dim_to_axis_for_da.keys())} with associated directions of {tuple(dim_to_axis_for_da.values())}.  Expected directions of ('T', 'Z', 'Y', 'X'). Transpose your array accordingly."
        )


def assert_valid_lat_lon(da_lat, da_lon, metadata: sgrid.SGrid2DMetadata):
    """
    Asserts that the provided longitude and latitude DataArrays are defined appropriately
    on the F points to match the internal representation in Parcels.

    - Longitude and latitude must be 1D or 2D (both must have the same dimensionality)
    - Both are defined on the node points (i.e., not the face/center)
    - If 1D:
      - Longitude is associated with the X axis
      - Latitude is associated with the Y axis
    - If 2D:
      - Lon and lat are defined on the same dimensions
      - Lon and lat are transposed such they're Y, X
    """
    assert_all_dimensions_correspond_with_axis(da_lon, metadata)
    assert_all_dimensions_correspond_with_axis(da_lat, metadata)

    for dim in da_lon.dims:
        if get_dim_position(metadata, dim) == "face":
            raise ValueError(
                f"Longitude DataArray {da_lon.name!r} with dims {da_lon.dims} is defined on the faces of the grid, but must be defined on the F nodes."
            )
    for dim in da_lat.dims:
        if get_dim_position(metadata, dim) == "face":
            raise ValueError(
                f"Latitude DataArray {da_lat.name!r} with dims {da_lat.dims} is defined on the faces of the grid, but must be defined on the F nodes."
            )

    if da_lon.ndim != da_lat.ndim:
        raise ValueError(
            f"Longitude DataArray {da_lon.name!r} with dims {da_lon.dims} and Latitude DataArray {da_lat.name!r} with dims {da_lat.dims} have different dimensionalities."
        )
    if da_lon.ndim not in (1, 2):
        raise ValueError(
            f"Longitude DataArray {da_lon.name!r} with dims {da_lon.dims} and Latitude DataArray {da_lat.name!r} with dims {da_lat.dims} must be 1D or 2D."
        )

    dim_to_axis = _get_dim_to_axis_mapping(metadata)

    if da_lon.ndim == 1:
        if dim_to_axis.get(da_lon.dims[0]) != "X":
            raise ValueError(
                f"Longitude DataArray {da_lon.name!r} with dims {da_lon.dims} is not associated with the X axis."
            )
        if dim_to_axis.get(da_lat.dims[0]) != "Y":
            raise ValueError(
                f"Latitude DataArray {da_lat.name!r} with dims {da_lat.dims} is not associated with the Y axis."
            )

        if not np.all(np.diff(da_lon.values) > 0):
            raise ValueError(
                f"Longitude DataArray {da_lon.name!r} with dims {da_lon.dims} must be strictly increasing."
            )
        if not np.all(np.diff(da_lat.values) > 0):
            raise ValueError(f"Latitude DataArray {da_lat.name!r} with dims {da_lat.dims} must be strictly increasing.")

    if da_lon.ndim == 2:
        if da_lon.dims != da_lat.dims:
            raise ValueError(
                f"Longitude DataArray {da_lon.name!r} with dims {da_lon.dims} and Latitude DataArray {da_lat.name!r} with dims {da_lat.dims} must be defined on the same dimensions."
            )

        lon_axes = [dim_to_axis.get(dim) for dim in da_lon.dims]
        if lon_axes != ["Y", "X"]:
            raise ValueError(
                f"Longitude DataArray {da_lon.name!r} with dims {da_lon.dims} and Latitude DataArray {da_lat.name!r} with dims {da_lat.dims} must be defined on the X and Y axes and transposed to have dimensions in order of Y, X."
            )


def assert_valid_depth(da_depth):
    if not np.all(np.diff(da_depth.values) > 0):
        raise ValueError(
            f"Depth DataArray {da_depth.name!r} with dims {da_depth.dims} must be strictly increasing. "
            f'HINT: you may be able to use ds.reindex to flip depth - e.g., ds = ds.reindex({da_depth.name}=ds["{da_depth.name}"][::-1])'
        )


def _convert_center_pos_to_fpoint(
    *,
    index: int,
    bcoord: float,
    position: ptyping.GridPosition,
    f_point_position: sgrid.Padding,
) -> tuple[int, float]:
    """Converts a physical position relative to the cell edges defined in the grid to be relative to the face center.

    This is used to "localize" a position to be relative to the staggered grid at which the field is defined, so that
    it can be easily interpolated.

    This also handles different model input cell edges and centers are staggered in different directions (e.g., with NEMO and MITgcm).
    """
    if position != "face":  # Data is already defined on the F nodes
        return index, bcoord

    bcoord = bcoord - 0.5
    if bcoord < 0:
        bcoord += 1.0
        index -= 1

    # Correct relative to the f-point position
    # Padding.BOTH was "inner", Padding.LOW was "right" in xgcm vocabulary
    if f_point_position in (sgrid.Padding.BOTH, sgrid.Padding.LOW):
        index += 1

    return index, bcoord
