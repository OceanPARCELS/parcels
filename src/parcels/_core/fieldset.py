from __future__ import annotations

import sys
from collections.abc import Iterable
from typing import IO, TYPE_CHECKING

import cf_xarray  # noqa: F401
import numpy as np
import xarray as xr

import parcels._typing as ptyping
from parcels._core.field import Field, VectorField
from parcels._core.mesh import FlatMesh, SphericalMesh
from parcels._core.model import (
    ModelData,
    StructuredModelData,
    UnstructuredModelData,
    create_empty_constant_field_model,
)
from parcels._core.utils.string import _assert_str_and_python_varname
from parcels._core.utils.time import get_datetime_type_calendar
from parcels._core.utils.time import is_compatible as datetime_is_compatible
from parcels._python import NOTSET, NotSetType
from parcels._repr_utils import fieldset_describe
from parcels.interpolators import (
    XConstantField,
)

if TYPE_CHECKING:
    import uxarray as ux

    from parcels._core.basegrid import BaseGrid
    from parcels._typing import TimeLike
__all__ = ["FieldSet"]


class FieldSet:
    """FieldSet class that holds hydrodynamic data needed to execute particles.

    Parameters
    ----------
    ds : xarray.Dataset | uxarray.UxDataset)
        xarray.Dataset and/or uxarray.UxDataset objects containing the field data.

    Notes
    -----
    The `ds` object is a xarray.Dataset or uxarray.UxDataset object.
    In XArray terminology, the (Ux)Dataset holds multiple (Ux)DataArray objects.
    Each (Ux)DataArray object is a single "field" that is associated with their own
    dimensions and coordinates within the (Ux)Dataset.

    A (Ux)Dataset object is associated with a single mesh, which can have multiple
    types of "points" (multiple "grids") (e.g. for UxDataSets, these are "face_lon",
    "face_lat", "node_lon", "node_lat", "edge_lon", "edge_lat"). Each (Ux)DataArray is
    registered to a specific set of points on the mesh.

    For UxDataset objects, each `UXDataArray.attributes` field dictionary contains
    the necessary metadata to help determine which set of points a field is registered
    to and what parent model the field is associated with. Parcels uses this metadata
    during execution for interpolation.  Each `UXDataArray.attributes` field dictionary
    must have:
    * "location" key set to "face", "node", or "edge" to define which pairing of points a field is associated with.
    * "mesh" key to define which parent model the fields are associated with (e.g. "fesom_mesh", "icon_mesh")

    """

    def __init__(self, models: list[ModelData]):
        if models == []:
            raise ValueError("List of models can't be empty.")
        for model in models:
            if not isinstance(model, ModelData):
                raise ValueError(f"Expected `model` to be a ModelData object. Got {model}")
        # assert_compatible_calendars(fields)

        self.models = models
        self.constant_model: StructuredModelData | None = None
        self._fields: dict[str, Field | VectorField] | None = None
        self.reconstruct_fields()
        self.context: dict[str, float] = {}
        assert_models_have_same_mesh(self.models)

    @property
    def mesh(self) -> FlatMesh | SphericalMesh:
        return self.models[0].mesh

    def __setattr__(self, name, value):
        """Set field attribute by name. If context exists and name in context, raise error to prevent overwriting context variable."""
        context = self.__dict__.get("context")

        if context is not None and name in context:
            raise AttributeError(f"Cannot assign '{name}' directly. Use fieldset.context['{name}'] instead.")
        # Handle setting of attributes not in context per default
        super().__setattr__(name, value)

    @property
    def fields(self):
        if self._fields is None:
            self.reconstruct_fields()
        assert self._fields is not None
        return self._fields

    def reconstruct_fields(self):
        fields = []
        for model in self.models:
            fields += model.construct_fields()
        if self.constant_model is not None:
            fields += self.constant_model.construct_fields()
        self._fields = {f.name: f for f in fields}

    def __getattr__(self, name):
        """Get the field by name. If the field is not found, check if it's a context variable."""
        if name in self._fields:
            return self._fields[name]
        elif name in self.context:
            return self.context[name]
        else:
            raise AttributeError(f"FieldSet has no attribute '{name}'")

    def __add__(self, other: FieldSet) -> FieldSet:
        if not isinstance(other, FieldSet):
            return NotImplemented
        assert_compatible_fieldsets(self, other)
        combined = FieldSet(self.models + other.models)
        combined.context = {**self.context, **other.context}
        # Carry over constant model
        if self.constant_model is not None and other.constant_model is not None:
            # Merge data variables from both constant models into one
            combined.constant_model = self.constant_model
            for name in other.constant_model.scalar_field_names:
                combined.constant_model.data[name] = other.constant_model.data[name]
        elif self.constant_model is not None or other.constant_model is not None:
            combined.constant_model = self.constant_model or other.constant_model

        combined.reconstruct_fields()
        return combined

    # def __repr__(self):
    #     return fieldset_repr(self)

    @property
    def time_interval(self):
        """Returns the valid executable time interval of the FieldSet,
        which is the overlap of the time intervals of all fields
        in the FieldSet.
        """
        time_intervals = (m.time_interval for m in self.models)

        # Filter out Nones from constant Fields
        time_intervals = [t for t in time_intervals if t is not None]
        if len(time_intervals) == 0:  # All fields are constant fields
            return None

        overlap = time_intervals[0]
        for time_interval in time_intervals[1:]:
            if overlap is None:
                return None
            overlap = overlap.intersection(time_interval)

        return overlap

    def to_windowed_arrays(self, *, max_levels: int | None = None):
        """Wrap dask-backed field data in rolling time-window caches.

        Opt-in optimization for forward-marching simulations where all particles
        share a single clock. Delegates to each underlying model; dask-backed,
        time-leading fields are served through a resident NumPy window (each time
        level loaded once and evicted as the clock advances) instead of re-reading
        chunks on every kernel step. NumPy-backed (eager) and non-time-leading
        fields are left unchanged, and re-invoking is idempotent, so this is safe
        to call more than once.

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

        Returns
        -------
        FieldSet
            ``self``, to allow chaining.
        """
        for model in self.models:
            model.to_windowed_arrays(max_levels=max_levels)
        return self

    def to_chunk_cached_arrays(self, *, max_cache_bytes: int = 600_000_000):
        """Wrap dask-backed field data in chunk-level LRU caches.

        Opt-in optimization that replaces each dask-backed data variable's
        internal storage with a :class:`~parcels._chunk_cached_array.ChunkCachedArray`.
        Delegates to each underlying model; repeated vectorized ``.isel()``
        calls then hit an in-memory LRU cache instead of recomputing dask task
        graphs. NumPy-backed (eager) fields are left unchanged, and re-invoking
        is idempotent.

        Parameters
        ----------
        max_cache_bytes : int, optional
            Maximum cache size in bytes, per variable. Defaults to 600 MB.

        Returns
        -------
        FieldSet
            ``self``, to allow chaining.
        """
        for model in self.models:
            model.to_chunk_cached_arrays(max_cache_bytes=max_cache_bytes)
        return self

    def add_constant_field(self, name: str, value):
        """Wrapper function to add a Field that is constant in space,
           useful e.g. when using constant horizontal diffusivity

        Parameters
        ----------
        name : str
            Name of the :class:`parcels.field.Field` object to be added
        value :
            Value of the constant field
        """
        if self.constant_model is None:
            self.constant_model = create_empty_constant_field_model(self.mesh)

        self.constant_model.data[name] = (["time", "depth", "lat", "lon"], np.full((1, 1, 1, 1), value))

        self.reconstruct_fields()
        field = getattr(self, name)
        field.interp_method = XConstantField()
        assert_models_have_same_mesh(self.models)

    def add_context(self, name, value):
        """Add context variable to the FieldSet.

        Parameters
        ----------
        name : str
            Name of the context variable
        value :
            Value of the context variable

        """
        _assert_str_and_python_varname(name)

        if name in self.context:
            raise ValueError(f"FieldSet already has a context with name '{name}'")
        self.context[name] = value

    @property
    def gridset(self) -> list[BaseGrid]:
        grids = []
        for field in self.fields.values():
            if field.grid not in grids:
                grids.append(field.grid)
        return grids

    @classmethod
    def from_ugrid_conventions(
        cls,
        ds: ux.UxDataset,
        mesh: str = "spherical",
        vector_fields: ptyping.VectorFields | NotSetType = NOTSET,
    ):
        """Create a FieldSet from a Parcels compliant uxarray.UxDataset.

        This is the primary ingestion method in Parcels for structured grid datasets.

        The main requirements for a uxDataset are naming conventions for vertical grid dimensions & coordinates

          zf - Name for coordinate and dimension for vertical positions at layer interfaces
          zc - Name for coordinate and dimension for vertical positions at layer centers

        Parameters
        ----------
        ds : uxarray.UxDataset
            uxarray.UxDataset as obtained from the uxarray package but with appropriate named vertical dimensions
        vector_fields : Mapping[str, tuple[str, ...]], optional
            Mapping of vector field names to tuples of component variable names in the dataset.
            For example, ``{"UV": ("U", "V"), "UVW": ("U", "V", "W")}``.
            If omitted (default), vector fields are auto-discovered from standard variable names (``U``/``V``/``W``).

        Returns
        -------
        FieldSet
            FieldSet object containing the fields from the dataset that can be used for a Parcels simulation.

        Notes
        -----
        See https://ugrid-conventions.github.io/ugrid-conventions/ for more information on the UGRID conventions.
        """
        model = UnstructuredModelData.from_ugrid_conventions(ds, mesh=mesh, vector_fields=vector_fields)
        return cls([model])

    @classmethod
    def from_sgrid_conventions(
        cls,
        ds: xr.Dataset,
        mesh: ptyping.TMesh | None = None,
        vector_fields: ptyping.VectorFields | NotSetType = NOTSET,
        skip_field_data_validation: bool = False,
    ):
        """Create a FieldSet from a dataset using SGRID convention metadata.

        This is the primary ingestion method in Parcels for structured grid datasets.

        Assumes that U, V, (and optionally W) variables are named 'U', 'V', and 'W' in the dataset.

        Parameters
        ----------
        ds : xarray.Dataset
            xarray.Dataset with SGRID convention metadata.
        mesh : MeshType
            MeshType indicating the type of mesh coordinates used during
            velocity interpolation. Default is None, in which case the mesh type will be inferred
            from the units of the dataset coordinates.
        vector_fields : Mapping[str, tuple[str, ...]], optional
            Mapping of vector field names to tuples of component variable names in the dataset.
            For example, ``{"UV": ("U", "V"), "UVW": ("U", "V", "W")}``.
            If omitted (default), vector fields are auto-discovered from standard variable names (``U``/``V``/``W``).
        skip_field_data_validation : bool, optional
            If True, skip validation of the field data. This can be useful for performance reasons, but may lead to unexpected behavior if the field data is invalid. Default is False.

        Returns
        -------
        FieldSet
            FieldSet object containing the fields from the dataset that can be used for a Parcels simulation.

        Notes
        -----
        This method uses the SGRID convention metadata to parse the grid structure
        and create appropriate Fields for a Parcels simulation. The dataset should
        contain a variable with 'cf_role' attribute set to 'grid_topology'.

        See https://sgrid.github.io/sgrid/ for more information on the SGRID conventions.
        """
        model = StructuredModelData.from_sgrid_conventions(
            ds, mesh=mesh, vector_fields=vector_fields, skip_field_data_validation=skip_field_data_validation
        )
        return cls([model])

    def describe(self, buf: IO | None = None) -> None:
        """
        Summary of a FieldSet including available Fields, associated
        interpolators, and context values.

        Parameters
        ----------
        buf : file-like, default: sys.stdout
            writable buffer
        """
        if buf is None:
            buf = sys.stdout
        assert buf is not None

        buf.write(fieldset_describe(self))


def assert_compatible_fieldsets(left: FieldSet, right: FieldSet) -> None:
    """Assert that two FieldSets can be combined without name conflicts.

    Parameters
    ----------
    left, right : FieldSet
        The two FieldSets to check.

    Raises
    ------
    ValueError
        If the FieldSets share field names or constant names.
    """
    common_fields = set(left.fields) & set(right.fields)
    if common_fields:
        raise ValueError(
            f"Cannot add FieldSets that have field names in common. Duplicate field names are: {sorted(common_fields)}"
        )

    common_context = set(left.context) & set(right.context)
    if common_context:
        raise ValueError(
            f"Cannot add FieldSets that have context value names in common. Duplicate context value names are: {sorted(common_context)}"
        )


class IncompatibleMeshesException(Exception): ...


def assert_models_have_same_mesh(models: list[ModelData]):
    if models == []:
        return

    first_mesh = None
    for i, model in enumerate(models):
        if first_mesh is None:
            first_mesh = model.mesh
            continue

        if model.mesh != first_mesh:
            raise IncompatibleMeshesException(
                f"All ModelData objects must have the same meshes. ModelData at index 0 has a mesh of {first_mesh!r} while ModelData at index {i} has mesh {model.mesh!r} "
            )


class CalendarError(Exception):  # TODO: Move to a Parcels errors module
    """Exception raised when the calendar of a field is not compatible with the rest of the Fields. The user should ensure that they only add fields to a FieldSet that have compatible CFtime calendars."""


def assert_compatible_calendars(fields: Iterable[Field | VectorField]):
    time_intervals = [f.time_interval for f in fields if f.time_interval is not None]

    if len(time_intervals) == 0:  # All time intervals are none
        return

    reference_datetime_object = time_intervals[0].left

    for field in fields:
        if field.time_interval is None:
            continue

        if not datetime_is_compatible(reference_datetime_object, field.time_interval.left):
            msg = _format_calendar_error_message(field, reference_datetime_object)
            raise CalendarError(msg)


def _datetime_to_msg(example_datetime: TimeLike) -> str:
    datetime_type, calendar = get_datetime_type_calendar(example_datetime)
    msg = str(datetime_type)
    if calendar is not None:
        msg += f" with cftime calendar {calendar}'"
    return msg


def _format_calendar_error_message(field: Field | VectorField, reference_datetime: TimeLike) -> str:
    return f"Expected field {field.name!r} to have calendar compatible with datetime object {_datetime_to_msg(reference_datetime)}. Got field with calendar {_datetime_to_msg(field.time_interval.left)}. Have you considered using xarray to update the time dimension of the dataset to have a compatible calendar?"


_COPERNICUS_MARINE_AXIS_VARNAMES = {
    "T": "time",
    "Z": "depth",
    "Y": "lat",
    "X": "lon",
}


_COPERNICUS_MARINE_CF_STANDARD_NAME_FALLBACKS = {
    "UV": [
        (
            "eastward_sea_water_velocity",
            "northward_sea_water_velocity",
        ),  # GLOBAL_ANALYSISFORECAST_PHY_001_024, MEDSEA_ANALYSISFORECAST_PHY_006_013, BALTICSEA_ANALYSISFORECAST_PHY_003_006, BLKSEA_ANALYSISFORECAST_PHY_007_001, IBI_ANALYSISFORECAST_PHY_005_001, NWSHELF_ANALYSISFORECAST_PHY_004_013, MULTIOBS_GLO_PHY_MYNRT_015_003, MULTIOBS_GLO_PHY_W_3D_REP_015_007
        (
            "surface_geostrophic_eastward_sea_water_velocity",
            "surface_geostrophic_northward_sea_water_velocity",
        ),  # SEALEVEL_GLO_PHY_L4_MY_008_047, SEALEVEL_EUR_PHY_L4_NRT_008_060
        (
            "geostrophic_eastward_sea_water_velocity",
            "geostrophic_northward_sea_water_velocity",
        ),  # MULTIOBS_GLO_PHY_TSUV_3D_MYNRT_015_012
        (
            "sea_surface_wave_stokes_drift_x_velocity",
            "sea_surface_wave_stokes_drift_y_velocity",
        ),  # GLOBAL_ANALYSISFORECAST_WAV_001_027, MEDSEA_MULTIYEAR_WAV_006_012, ARCTIC_ANALYSIS_FORECAST_WAV_002_014, BLKSEA_ANALYSISFORECAST_WAV_007_003, IBI_ANALYSISFORECAST_WAV_005_005, NWSHELF_ANALYSISFORECAST_WAV_004_014
        ("sea_water_x_velocity", "sea_water_y_velocity"),  # ARCTIC_ANALYSISFORECAST_PHY_002_001
        (
            "eastward_sea_water_velocity_vertical_mean_over_pelagic_layer",
            "northward_sea_water_velocity_vertical_mean_over_pelagic_layer",
        ),  # GLOBAL_MULTIYEAR_BGC_001_033
    ],
    "W": ["upward_sea_water_velocity", "vertical_sea_water_velocity"],
}
