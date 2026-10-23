from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable, Sequence
from typing import TYPE_CHECKING, Any, Self

import cf_xarray  # noqa: F401
import xarray as xr
from dask import is_dask_collection

import parcels._sgrid as sgrid
import parcels._typing as ptyping
from parcels._chunk_cached_array import wrap_dataset
from parcels._core._windowed_array import maybe_windowed
from parcels._core.basegrid import BaseGrid
from parcels._core.field import Field, VectorField
from parcels._core.mesh import FlatMesh, SphericalMesh, _infer_mesh_from_coordinates
from parcels._core.utils.time import TimeInterval
from parcels._core.uxgrid import UxGrid
from parcels._core.xgrid import (
    XGrid,
    _transpose_xfield_data_to_tzyx,
    assert_all_field_dims_have_axis,  # noqa: F401, leave import for now until decision is made # TODO v4: Make decision
)
from parcels._logger import logger
from parcels._python import NOTSET, NotSetType
from parcels._sgrid.accessor import assert_metadata_ds_consistency
from parcels.interpolators import (
    CGrid_Velocity,
    Ux_Velocity,
    UxConstantFaceConstantZC,
    UxConstantFaceLinearZF,
    UxLinearNodeConstantZC,
    UxLinearNodeLinearZF,
    XLinear,
    XLinear_Velocity,
)
from parcels.interpolators._base import ScalarInterpolator, VectorInterpolator

if TYPE_CHECKING:
    import uxarray as ux


class ModelData(ABC):
    data: Any
    grid: BaseGrid
    field_to_interpolator: dict[str, ScalarInterpolator | VectorInterpolator]
    vector_field_components: ptyping.VectorFields

    @property
    def mesh(self) -> FlatMesh | SphericalMesh:
        return self.grid.mesh

    @abstractmethod
    def construct_fields(self) -> list[Field | VectorField]: ...

    @property
    @abstractmethod
    def scalar_field_names(self) -> list[str]: ...

    @abstractmethod
    def assert_valid_field_data(self, field_data: Any) -> None: ...

    def assert_valid_model_data(self) -> None:
        for field_name in self.scalar_field_names:
            field_data = self.data[field_name]
            try:
                self.assert_valid_field_data(field_data)
            except Exception as e:
                e.add_note(f"Error validating field {field_name!r}.")
                raise e
        return

    def field_data(self, name: str) -> Any:
        """Return the array backing field ``name``.

        Normally this is the ``xr.DataArray`` held in the dataset. After
        :meth:`to_windowed_arrays`, dask-backed fields are served through a
        cached :class:`~parcels._core._windowed_array.WindowedArray` instead.
        """
        windowed = self.__dict__.get("_windowed")
        if windowed is not None and name in windowed:
            return windowed[name]
        return self.data[name]

    def to_windowed_arrays(self, *, max_levels: int | None = None) -> Self:
        """Wrap dask-backed field data in rolling time-window caches.

        Opt-in optimization for forward-marching simulations where all particles
        share a single clock. For each dask-backed, time-leading field, ``isel``
        then samples a resident NumPy window (each time level loaded once and
        evicted as the clock advances) instead of re-reading chunks and paying the
        dask scheduling overhead on every kernel step. NumPy-backed (eager) fields
        and non-time-leading fields are left unchanged.

        Idempotent: re-invoking reuses the existing wrapper (and its warm cache)
        rather than rebuilding it.

        Parameters
        ----------
        max_levels : int, optional
            Hard cap on the number of time levels kept resident per field.
            With the default ``None``, each interpolation call decides what
            stays resident: the cache keeps exactly the span of time indices
            that call requests and evicts every level outside it. During time
            integration particles bracket the current time between two
            adjacent levels, so the default keeps at most two levels resident.
            Only when a single call requests a wider time span (e.g. particles
            spread across many time levels) does the window grow beyond that,
            and ``max_levels`` then bounds its size.
        """
        windowed = self.__dict__.setdefault("_windowed", {})
        for dim in ["lon", "lat", "depth"]:
            # ensure lon, lat, depth are loaded into memory for dask-backed datasets
            if dim in self.data and is_dask_collection(self.data[dim]):
                self.data[dim].load()
        for name in self.scalar_field_names:
            current = windowed.get(name, self.data[name])
            windowed[name] = maybe_windowed(current, max_levels=max_levels)
        return self

    def to_chunk_cached_arrays(self, *, max_cache_bytes: int = 600_000_000) -> Self:
        """Wrap dask-backed field data in chunk-level LRU caches.

        Opt-in optimization that replaces each dask-backed data variable's
        internal storage with a :class:`~parcels._chunk_cached_array.ChunkCachedArray`.
        Repeated vectorized ``.isel()`` calls then hit an in-memory LRU cache
        keyed by chunk coordinates instead of recomputing dask task graphs.

        Coordinate variables are loaded eagerly into memory (they are small 1D
        arrays) to avoid dask task-graph construction overhead on every
        ``.isel()`` call.

        Idempotent: re-invoking is safe — ``wrap_dataset`` skips variables
        whose storage is already a ``ChunkCachedArray``.

        Parameters
        ----------
        max_cache_bytes : int, optional
            Maximum cache size in bytes, per variable. Defaults to 600 MB.
        """
        self.data = wrap_dataset(self.data, max_cache_bytes=max_cache_bytes)
        return self

    @property
    def time_interval(self) -> TimeInterval | None:
        try:
            time_interval = _get_time_interval(self.data)
        except ValueError as e:
            e.add_note(
                f"Error getting time interval for:\n {self.data!r}\n\nAre you sure that the time dimension on the xarray dataset is stored as timedelta, datetime or cftime datetime objects?"
            )
            raise e
        return time_interval


def preprocess_sgrid_model_data(ds: xr.Dataset) -> xr.Dataset:
    metadata: sgrid.SGrid2DMetadata = ds.sgrid.metadata

    for field_name in set(ds.data_vars) - {ds.sgrid._get_grid_topology().name}:
        ds[field_name] = _transpose_xfield_data_to_tzyx(ds[field_name], metadata)
    return ds


def validate_field_data(ds: xr.Dataset) -> xr.Dataset:
    return ds.fillna(0)


class StructuredModelData(ModelData):
    def __init__(
        self,
        data: xr.Dataset,
        mesh: ptyping.TMesh,
        vector_field_components: ptyping.VectorFields,
        skip_field_data_validation: bool = False,
    ):
        if not isinstance(data, xr.Dataset):
            raise ValueError(f"Expected `data` to be an xarray.Dataset . Got {type(data)}")

        data = preprocess_sgrid_model_data(data)
        if not skip_field_data_validation:
            data = validate_field_data(data)
        grid = XGrid(data, mesh)

        self.data = data
        self.grid = grid
        self.vector_field_components = vector_field_components
        self.field_to_interpolator = {}
        self._fields: list[Field | VectorField] | None = None
        self.assert_valid_model_data()

    def assert_valid_field_data(self, field_data: xr.DataArray) -> None:
        # assert_all_field_dims_have_axis(field_data, self.grid.xgcm_grid) #! TODO v4: These checks should be revisited
        _assert_has_time_coordinate(field_data)

    @property
    def scalar_field_names(self) -> list[str]:
        # Create fields from data variables, skipping grid metadata variables
        # Skip variables that are SGRID metadata (have cf_role='grid_topology')
        skip_vars = set()
        for var in self.data.data_vars:
            if self.data[var].attrs.get("cf_role") == "grid_topology":
                skip_vars.add(var)
        return list(set(self.data.data_vars) - skip_vars)

    def construct_fields(self) -> list[Field | VectorField]:
        single_fields: dict[str, Field] = {}
        vector_fields: dict[str, VectorField] = {}
        scalar_field_names = self.scalar_field_names

        for varname in set(scalar_field_names):
            single_fields[varname] = Field(str(varname), self)

        for vfield_name, components in self.vector_field_components.items():
            interp_method = (
                XLinear_Velocity() if _is_agrid(self.data, u=components[0], v=components[1]) else CGrid_Velocity()
            )

            component_fields = [single_fields[name] for name in components]
            vector_fields[vfield_name] = VectorField(vfield_name, *component_fields, interp_method=interp_method)  # type:ignore[misc,arg-type]

        fields: dict[str, Field | VectorField] = {**single_fields, **vector_fields}
        return list(fields.values())

    @classmethod
    def from_sgrid_conventions(
        cls,
        ds: xr.Dataset,
        *,
        mesh: ptyping.TMesh | None = None,
        vector_fields: ptyping.VectorFields | NotSetType,
        skip_field_data_validation: bool = False,
    ) -> Self:
        ds = ds.copy()
        if mesh is None:
            mesh = _infer_mesh_from_sgrid_data(ds)

        # Ensure time dimension has axis attribute if present
        if "time" in ds.dims and "time" in ds.coords:
            if "axis" not in ds["time"].attrs:
                logger.debug(
                    "Dataset contains 'time' dimension but no 'axis' attribute. Setting 'axis' attribute to 'T'."
                )
                ds["time"].attrs["axis"] = "T"

        # Find time dimension based on axis attribute and rename to `time`
        if (time_dims := ds.cf.axes.get("T")) is not None:
            if len(time_dims) > 1:
                raise ValueError("Multiple time coordinates found in dataset. This is not supported by Parcels.")
            (time_dim,) = time_dims
            if time_dim != "time":
                logger.debug(f"Renaming time axis coordinate from {time_dim} to 'time'.")
                ds = ds.rename({time_dim: "time"})

        # if "lon" not in ds.coords or "lat" not in ds.coords:
        #     node_dimensions = sgrid.load_mappings(ds.grid.node_dimensions)
        #     ds["lon"] = ds[node_dimensions[0]]
        #     ds["lat"] = ds[node_dimensions[1]]

        assert_metadata_ds_consistency(ds, ds.sgrid.metadata)

        vector_fields = resolve_vector_fields(ds, vector_fields)
        assert_valid_vector_fields(ds, vector_fields)

        model = cls(
            ds,
            mesh=mesh,
            vector_field_components=vector_fields,
            skip_field_data_validation=skip_field_data_validation,
        )
        model._fields = model.construct_fields()
        for f in model._fields:
            if isinstance(f, Field):
                f.interp_method = XLinear()
        return model


def resolve_vector_fields(ds: xr.Dataset, vector_fields: ptyping.VectorFields | NotSetType) -> ptyping.VectorFields:
    if vector_fields is NOTSET:  # i.e., the default vectorfield discovery behaviour
        return _default_vector_field_components(list(ds.data_vars))
    return vector_fields


def assert_valid_vector_fields(ds: xr.Dataset, vector_fields: ptyping.VectorFields) -> None:
    if not isinstance(vector_fields, dict):
        raise ValueError(f"vector_fields must be a dictionary. Got {type(vector_fields)=!r}.")

    for vfield_name, components in vector_fields.items():
        if not isinstance(vfield_name, str):
            raise ValueError(
                f"Invalid `vector_fields` argument. Vector field name in `vector_fields` should be a string. Got field name {vfield_name!r}."
            )
        if not (2 <= len(components) <= 3):
            raise ValueError(
                f"Invalid `vector_fields` argument. Vector fields must have either 2 or 3 components. Vector field {vfield_name} has {len(components)} components."
            )
        for c in components:
            if not isinstance(c, str):
                raise ValueError(
                    f"Invalid `vector_fields` argument. Component names must be strings. Got component name of value {c!r}."
                )

    assert_vector_field_components_in_dataset(ds, vector_fields)
    return


def assert_vector_field_components_in_dataset(ds: xr.Dataset, vector_fields: ptyping.VectorFields) -> None:
    for components in vector_fields.values():
        for c in components:
            if c not in ds.data_vars:
                raise ValueError(
                    f"Field component '{c}' not present in the source dataset, but is listed in {vector_fields=!r}. This component cannot be used in this mapping."
                )
    return


def create_empty_constant_field_model(mesh: SphericalMesh | FlatMesh) -> StructuredModelData:
    """Create a empty model for constant fields with the given mesh type."""
    unit = "m" if isinstance(mesh, FlatMesh) else "degrees"
    return StructuredModelData.from_sgrid_conventions(
        xr.Dataset(
            {},
            coords={
                "lat": (["lat"], [0], {"axis": "Y", "units": unit}),
                "lon": (["lon"], [0], {"axis": "X", "units": unit}),
                "depth": (["depth"], [0], {"axis": "Z"}),
                "time": (["time"], [0], {"axis": "T"}),
            },
        ).pipe(
            sgrid._attach_sgrid_metadata,
            sgrid.SGrid2DMetadata(
                cf_role="grid_topology",
                topology_dimension=2,
                node_dimensions=("lon", "lat"),
                face_dimensions=(
                    sgrid.FaceNodePadding("XC", "lon", sgrid.Padding.LOW),
                    sgrid.FaceNodePadding("YC", "lat", sgrid.Padding.LOW),
                ),
            ),
        ),
        mesh=mesh,
        vector_fields={},
    )


class UnstructuredModelData(ModelData):
    def __init__(self, data: ux.UxDataset, grid: UxGrid, vector_field_components: ptyping.VectorFields):
        import uxarray as ux

        if not isinstance(data, ux.UxDataset):
            raise ValueError(f"Expected `data` to be an uxarray.UxDataset . Got {type(data)}")

        if not isinstance(grid, UxGrid):
            raise ValueError(f"Expected `grid` to be a Parcels UxGrid object. Got {type(grid)}.")

        if grid.z.ndim == 3 and not grid.z["time"].equals(data["time"]):
            raise ValueError("A time-varying (3D) z must have the same time coordinate as `data`.")

        self.data = data
        self.grid = grid
        self.vector_field_components = vector_field_components
        self.field_to_interpolator = {}
        self._fields: list[Field | VectorField] | None = None

    def construct_fields(self) -> list[Field | VectorField]:
        single_fields: dict[str, Field] = {}
        vector_fields: dict[str, VectorField] = {}
        scalar_field_names = self.scalar_field_names

        for varname in set(scalar_field_names):
            single_fields[varname] = Field(str(varname), self)

        for vfield_name, components in self.vector_field_components.items():
            interp_method = Ux_Velocity()

            component_fields = [single_fields[name] for name in components]
            vector_fields[vfield_name] = VectorField(vfield_name, *component_fields, interp_method=interp_method)  # type:ignore[misc, arg-type]

        fields: dict[str, Field | VectorField] = {**single_fields, **vector_fields}
        return list(fields.values())

    def assert_valid_field_data(self, field_data: ux.UxDataArray) -> None:
        _assert_valid_uxdataarray(field_data)
        _assert_has_time_coordinate(field_data)

    @property
    def scalar_field_names(self) -> list[str]:
        return list(self.data.data_vars)

    @classmethod
    def from_ugrid_conventions(
        cls, ds: ux.UxDataset, *, mesh: ptyping.TMesh | None = None, vector_fields: ptyping.VectorFields | NotSetType
    ):
        ds_dims = list(ds.dims)
        if not all(dim in ds_dims for dim in ["time", "zf", "zc"]):
            raise ValueError(
                f"Dataset missing one of the required dimensions 'time', 'zf', or 'zc' for uxDataset. Found dimensions {ds_dims}"
            )

        if mesh is None:
            mesh = _infer_mesh_from_ugrid_data(ds)

        grid = UxGrid(ds.uxgrid, z=ds.coords["zf"], mesh=mesh)
        ds = _discover_ux_U_and_V(ds)

        vector_fields = resolve_vector_fields(ds, vector_fields)
        assert_valid_vector_fields(ds, vector_fields)

        model = cls(ds, grid, vector_fields)
        model._fields = model.construct_fields()
        for f in model._fields:
            if isinstance(f, Field):
                interp_cls = _select_uxinterpolator(model.data[f.name])
                if interp_cls is not None:
                    f.interp_method = interp_cls()
        return model


def _infer_mesh_from_ugrid_data(ds_ugrid: ux.UxDataset) -> ptyping.TMesh | None:
    """Small helper to inspect UxGRID metadata and dataset metadata to determine mesh type."""
    return _infer_mesh_from_coordinates(ds_ugrid.uxgrid.node_lon, ds_ugrid.uxgrid.node_lat)


def _infer_mesh_from_sgrid_data(ds_sgrid: xr.Dataset) -> ptyping.TMesh:
    """Small helper to inspect SGRID metadata and dataset metadata to determine mesh type."""
    sgrid_metadata = ds_sgrid.sgrid.metadata

    if sgrid_metadata.node_coordinates is None:
        horizontal_dims = [d for d in sgrid_metadata.node_dimensions[:2] if d in ds_sgrid.dims]
        if horizontal_dims:
            raise ValueError(
                f"SGRID metadata has no 'node_coordinates', but the dataset has horizontal node dimensions "
                f"{horizontal_dims}. Add 'node_coordinates' to the SGRID metadata, or pass `mesh` explicitly."
            )
        # No horizontal dimensions (e.g. 0D or Z-only grids), so there is no geometry to infer from
        return FlatMesh()

    fpoint_x, fpoint_y = sgrid_metadata.node_coordinates

    return _infer_mesh_from_coordinates(ds_sgrid[fpoint_x], ds_sgrid[fpoint_y])


def _default_vector_field_components(data_vars: Sequence[Hashable]) -> ptyping.VectorFields:
    vars = set(data_vars)
    ret: ptyping.VectorFields = {}

    if {"U", "V"}.issubset(vars):
        ret["UV"] = ("U", "V")
    if {"U", "V", "W"}.issubset(vars):
        ret["UVW"] = ("U", "V", "W")
    return ret


def _discover_ux_U_and_V(ds: ux.UxDataset) -> ux.UxDataset:
    # Common variable names for U and V found in UxDatasets
    common_ux_UV = [("unod", "vnod"), ("u", "v")]
    common_ux_W = ["w"]

    if "W" not in ds:
        for common_W in common_ux_W:
            if common_W in ds:
                ds = ds.rename({common_W: "W"})
                break

    if "U" in ds and "V" in ds:
        return ds  # U and V already present
    elif "U" in ds or "V" in ds:
        raise ValueError(
            "Dataset has only one of the two variables 'U' and 'V'. Please rename the appropriate variable in your dataset to have both 'U' and 'V' for Parcels simulation."
        )

    for common_U, common_V in common_ux_UV:
        if common_U in ds:
            if common_V not in ds:
                raise ValueError(
                    f"Dataset has variable with standard name {common_U!r}, "
                    f"but not the matching variable with standard name {common_V!r}. "
                    "Please rename the appropriate variables in your dataset to have both 'U' and 'V' for Parcels simulation."
                )
            else:
                ds = ds.rename({common_U: "U", common_V: "V"})
                break

        else:
            if common_V in ds:
                raise ValueError(
                    f"Dataset has variable with standard name {common_V!r}, "
                    f"but not the matching variable with standard name {common_U!r}. "
                    "Please rename the appropriate variables in your dataset to have both 'U' and 'V' for Parcels simulation."
                )
            continue

    return ds


def _select_uxinterpolator(da: ux.UxDataArray):
    """Selects the appropriate uxarray interpolator for a given uxarray UxDataArray"""
    supported_uxinterp_mapping = {
        # (zc,n_face): face-center laterally, layer centers vertically — piecewise constant
        "zc,n_face": UxConstantFaceConstantZC,
        # (zc,n_node): node/corner laterally, layer centers vertically — barycentric lateral & piecewise constant vertical
        "zc,n_node": UxLinearNodeConstantZC,
        # (zf,n_node): node/corner laterally, layer interfaces vertically — barycentric lateral & linear vertical
        "zf,n_node": UxLinearNodeLinearZF,
        # (zf,n_face): face-center laterally, layer interfaces vertically — piecewise constant lateral & linear vertical
        "zf,n_face": UxConstantFaceLinearZF,
    }
    # Extract only spatial dimensions, neglecting time
    da_spatial_dims = tuple(d for d in da.dims if d not in ("time",))
    if len(da_spatial_dims) != 2:
        raise ValueError(
            "Fields on unstructured grids must have two spatial dimensions, one vertical (zf or zc) and one lateral (n_face, n_edge, or n_node)"
        )

    # Construct key (string) for mapping to interpolator
    # Find vertical and lateral tokens
    vdim = None
    ldim = None
    for d in da_spatial_dims:
        if d in ("zf", "zc"):
            vdim = d
        if d in ("n_face", "n_node"):
            ldim = d
    # Map to supported interpolators
    if vdim and ldim:
        key = f"{vdim},{ldim}"
        if key in supported_uxinterp_mapping.keys():
            return supported_uxinterp_mapping[key]

    return None


def _is_agrid(ds: xr.Dataset, u: str, v: str) -> bool:
    # check if U and V are defined on the same dimensions
    # if yes, interpret as A grid
    return set(ds[u].dims) == set(ds[v].dims)


def _get_time_interval(data: xr.DataArray | ux.UxDataArray) -> TimeInterval | None:
    if "time" not in data or data["time"].size == 1:
        return None

    return TimeInterval(data.time.values[0], data.time.values[-1])


def _assert_valid_uxdataarray(data: ux.UxDataArray):
    """Verifies that all the required attributes are present in the xarray.DataArray or
    uxarray.UxDataArray object.
    """
    # Validate dimensions
    if not ("zf" in data.dims or "zc" in data.dims):
        raise ValueError(
            "Field is missing a 'zf' or 'zc' dimension in the field's metadata. "
            "This attribute is required for xarray.DataArray objects."
        )

    if "time" not in data.dims:
        raise ValueError(
            "Field is missing a 'time' dimension in the field's metadata. "
            "This attribute is required for xarray.DataArray objects."
        )


def _assert_has_time_coordinate(da: xr.DataArray) -> None:
    if da.shape[0] > 1:
        if "time" not in da.coords:
            raise ValueError("Field data is missing a 'time' coordinate.")
    return
