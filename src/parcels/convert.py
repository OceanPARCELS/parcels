"""
This module provides a series of functions to convert model outputs (which might be following
their own conventions) to metadata rich data (i.e., data following SGRID/UGRID as well as CF
conventions). Parcels needs this metadata rich data to discover grid geometries among other
things.

These functions use knowledge about the model to attach any missing metadata. The functions
emit verbose messaging so that the user is kept in the loop. The returned output is an
Xarray dataset so that users can further provide any missing metadata that was unable to
be determined before they pass it to the FieldSet constructor.
"""

from __future__ import annotations

import dataclasses
import enum
import typing
import warnings
from typing import cast

import numpy as np
import xarray as xr

import parcels._sgrid as sgrid
from parcels._logger import logger

if typing.TYPE_CHECKING:
    import uxarray as ux

    from parcels._typing import XgcmAxisDirection


class _Status(enum.Enum):
    REQUIRED = enum.auto()
    OPTIONAL = enum.auto()


@dataclasses.dataclass(frozen=True)
class _ModelConventions:
    """Names used in a model's output, and how they map onto the names Parcels expects."""

    expected_coords: list[tuple[str, _Status]] = dataclasses.field(default_factory=list)
    """Coordinates to pick out of the user-provided ``coords`` dataset."""
    varnames_mapping: dict[str, str] = dataclasses.field(default_factory=dict)
    """Renames for variables and coordinates."""
    axis_varnames: dict[str, XgcmAxisDirection] = dataclasses.field(default_factory=dict)
    """CF axis of each dimension coordinate (set as its 'axis' attribute)."""


_NEMO = _ModelConventions(
    expected_coords=[
        ("glamf", _Status.REQUIRED),
        ("gphif", _Status.REQUIRED),
        ("depthw", _Status.OPTIONAL),
    ],
    varnames_mapping={
        "time_counter": "time",
        "depthw": "depth",
        "uo": "U",
        "vo": "V",
        "wo": "W",
    },
    axis_varnames={
        "x": "X",
        "x_center": "X",
        "y": "Y",
        "y_center": "Y",
        "depth": "Z",
        "depth_center": "Z",
        "time": "T",
    },
)

# Dimensions and coordinates kept in the NEMO output (others are dropped)
_NEMO_DIMENSION_COORD_NAMES: list[str] = [
    "time",
    "x",
    "x_center",
    "y",
    "y_center",
    "depth",
    "depth_center",
    "glamf",
    "gphif",
]

# NEMO uses a separate depth dimension per grid (T, U, V and W points)
_NEMO_DEPTH_DIM_RENAMES: dict[str, str] = {
    "depthu": "depth_center",
    "depthv": "depth_center",
    "deptht": "depth_center",
    "depthw": "depth",
}

_MOM6 = _ModelConventions(
    expected_coords=[
        ("geolon_c", _Status.REQUIRED),
        ("geolat_c", _Status.REQUIRED),
    ],
    axis_varnames={
        "xq": "X",
        "xh": "X",
        "yq": "Y",
        "yh": "Y",
        "time": "T",
    },
)

# Dimensions and coordinates kept in the MOM6 output (others are dropped)
_MOM6_DIMENSION_COORD_NAMES: list[str] = [
    "xq",
    "yq",
    "time",
    "xh",
    "yh",
    "geolon_c",
    "geolat_c",
]

_MITGCM = _ModelConventions(
    expected_coords=[(name, _Status.REQUIRED) for name in ["XG", "YG", "Zl"]],
    varnames_mapping={
        "XG": "lon",
        "YG": "lat",
        "Zl": "depth",
    },
    axis_varnames={
        "XC": "X",
        "XG": "X",
        "Xp1": "X",
        "lon": "X",
        "YC": "Y",
        "YG": "Y",
        "Yp1": "Y",
        "lat": "Y",
        "Zu": "Z",
        "Zl": "Z",
        "Zp1": "Z",
        "time": "T",
    },
)

_CROCO = _ModelConventions(
    expected_coords=[(name, _Status.REQUIRED) for name in ["x_rho", "y_rho", "s_w", "time"]],
    varnames_mapping={
        "x_rho": "lon",
        "y_rho": "lat",
        "s_w": "depth",
    },
)

# Copernicus Marine data is CF compliant, so coordinates are found by their CF axis and renamed
_COPERNICUS_MARINE_AXIS_VARNAMES: dict[XgcmAxisDirection, str] = {
    "X": "lon",
    "Y": "lat",
    "Z": "depth",
    "T": "time",
}

_DELFT3D_X = _ModelConventions(
    expected_coords=[(name, _Status.REQUIRED) for name in ["XZETA", "YZETA"]],
    varnames_mapping={
        "XZETA": "lon",
        "YZETA": "lat",
        "SIGMA_C": "depth",
        "TIME": "time",
    },
    axis_varnames={
        "M": "X",
        "N": "Y",
        "LAYER": "Z",
        "time": "T",
    },
)

# Known vertical dimension names of UGRID models, as {"interface": ..., "center": ...}
_FESOM2_VERTICAL_DIMS = {"interface": "nz", "center": "nz1"}
_ICON_VERTICAL_DIMS = {"interface": "depth_2", "center": "depth"}


def _pick_expected_coords(coords: xr.Dataset, expected_coord_names: list[tuple[str, _Status]]) -> xr.Dataset:
    coords_to_use = {}
    for name, status in expected_coord_names:
        if name in coords:
            coords_to_use[name] = coords[name]
        else:
            if status == _Status.REQUIRED:
                raise ValueError(f"Expected coordinate '{name}' not found in provided coords dataset.")
    return xr.Dataset(coords_to_use)


def _rename_nemo_depth_dims(ds: xr.Dataset) -> xr.Dataset:
    for var in ds.data_vars:
        for old_depth, target in _NEMO_DEPTH_DIM_RENAMES.items():
            if old_depth in ds[var].dims:
                ds[var] = ds[var].rename({old_depth: target})
    return ds


def _maybe_rename_coords(ds: xr.Dataset, axis_varnames: dict[XgcmAxisDirection, str]):
    try:
        for axis, [coord] in ds.cf.axes.items():
            ds = ds.rename({coord: axis_varnames[axis]})
    except ValueError as e:
        raise ValueError(f"Multiple coordinates found on axis '{axis}'. Check your DataSet.") from e
    return ds


def _maybe_rename_variables(ds: xr.Dataset, varnames_mapping: dict[str, str]):
    rename_dict = {old: new for old, new in varnames_mapping.items() if (old in ds.data_vars) or (old in ds.coords)}
    if rename_dict:
        ds = ds.rename(rename_dict)
    return ds


def _assign_dims_as_coords(ds: xr.Dataset, dimension_names: list[str]):
    for axis in dimension_names:
        if axis in ds.dims and axis not in ds.coords:
            ds = ds.assign_coords({axis: np.arange(ds.sizes[axis])})
    return ds


def _drop_unused_dimensions_and_coords(ds: xr.Dataset, dimension_and_coord_names: list[str]):
    for dim in ds.dims:
        if dim not in dimension_and_coord_names:
            dim = cast(str, dim)
            ds = ds.drop_dims(dim, errors="ignore")
    for coord in ds.coords:
        coord = cast(str, coord)
        if coord not in dimension_and_coord_names:
            ds = ds.drop_vars(coord, errors="ignore")
    return ds


def _set_coords(ds: xr.Dataset, dimension_names):
    for varname in dimension_names:
        if varname in ds and varname not in ds.coords:
            ds = ds.set_coords([varname])
    return ds


def _maybe_remove_depth_from_lonlat(ds):
    for coord in ["glamf", "gphif"]:
        if coord in ds.coords and "depth" in ds[coord].dims:
            ds[coord] = ds[coord].squeeze("depth", drop=True)
    return ds


def _set_axis_attrs(ds: xr.Dataset, dim_axis: dict[str, XgcmAxisDirection]):
    for dim, axis in dim_axis.items():
        if dim in ds:
            ds[dim].attrs["axis"] = axis
    return ds


def _maybe_convert_time_from_float_to_timedelta(ds: xr.Dataset) -> xr.Dataset:
    if "time" in ds.coords:
        if np.issubdtype(ds["time"].data.dtype, np.floating):
            time_units = ds["time"].attrs.get("units", "").lower()
            if "hour" in time_units:
                factor = 3600.0 * 1e9
            elif "day" in time_units:
                factor = 86400.0 * 1e9
            elif "minute" in time_units:
                factor = 60.0 * 1e9
            else:
                # default to seconds if unspecified
                factor = 1.0 * 1e9

            ns_int = np.rint(ds["time"].values * factor).astype("int64")
            try:
                ds["time"] = ns_int.astype("timedelta64[ns]")
                logger.info("Converted time coordinate from float to timedelta based on units.")
            except Exception as e:
                logger.warning(f"Failed to convert time coordinate to timedelta: {e}")
    return ds


def _maybe_swap_depth_direction(ds: xr.Dataset) -> xr.Dataset:
    if ds["depth"].size > 1:
        if ds["depth"][0] > ds["depth"][-1]:
            logger.info(
                "Depth dimension appears to be decreasing upward (i.e., from positive to negative values). Swapping depth dimension to be increasing upward for Parcels simulation."
            )
            ds = ds.reindex(depth=ds["depth"][::-1])
    return ds


def _maybe_expand_depth_dimension(ds: xr.Dataset) -> xr.Dataset:
    if "depth" not in ds.dims:
        logger.info(
            "No depth dimension found in dataset. Assuming surface data and adding a singleton depth dimension."
        )
        ds = ds.expand_dims(depth=[0]).transpose("time", "depth", ..., missing_dims="ignore")
    return ds


def _normalize_fields(fields: dict[str, xr.Dataset | xr.DataArray]) -> dict[str, xr.DataArray]:
    """Return each field as a DataArray named after its key in ``fields``."""
    normalized = {}
    for name, field in fields.items():
        if isinstance(field, xr.Dataset):
            # Convenience for model output with one dataset per grid (e.g., NEMO's U and V files) where the
            # variable already has the Parcels name: {"U": ds_u} selects ds_u["U"]. Other variables are dropped.
            if name not in field.data_vars:
                raise ValueError(
                    f"fields[{name!r}] is a Dataset without a variable named {name!r} (it has variables "
                    f"{list(field.data_vars)}). Either rename the variable in the Dataset to {name!r}, or pass "
                    f"the DataArray instead (e.g., fields={{{name!r}: ds['<variable name>']}})."
                )
            normalized[name] = field[name]
            # TODO: logging message, warn if multiple fields are in this dataset
        else:
            normalized[name] = field.rename(name)
    return normalized


def _flip_w_to_positive_down(ds: xr.Dataset) -> xr.Dataset:
    """Negate W to convert from up positive to down positive (as that's the direction of positive z)."""
    if "W" in ds.data_vars:
        ds["W"] = ds["W"].copy(data=-ds["W"].data)  # new array, so that the caller's data is left untouched
    return ds


def nemo_to_sgrid(*, fields: dict[str, xr.Dataset | xr.DataArray], coords: xr.Dataset):
    """Create an sgrid-compliant xarray.Dataset from a dataset of NEMO netcdf files.

    Parameters
    ----------
    fields : dict[str, xr.Dataset | xr.DataArray]
        Dictionary mapping Parcels field names (e.g., "U", "V") to xarray.DataArray objects as obtained from a
        set of NEMO netcdf files, e.g. ``{"U": ds["uo"]}``. An xarray.Dataset can be given instead of a
        DataArray if it contains a variable with the field name, which is then used (e.g., ``{"U": ds_u}`` uses
        ``ds_u["U"]``).
    coords : xarray.Dataset
        xarray.Dataset containing coordinate variables (e.g., the NEMO mesh mask). Must contain
        ``glamf`` and ``gphif``, and can optionally contain ``depthw``.

    Returns
    -------
    xarray.Dataset
        Dataset object following SGRID conventions to be (optionally) modified and passed to a FieldSet constructor.

    Notes
    -----
    The NEMO model (https://nemo-ocean.eu/) is used by a variety of oceanographic institutions around the world.
    Output from these models may differ subtly in terms of variable names and metadata conventions.
    This function attempts to standardize these differences to create a Parcels FieldSet.
    If you encounter issues with your specific NEMO dataset, please open an issue on the Parcels GitHub repository with details about your dataset.

    """
    coords = _pick_expected_coords(coords, _NEMO.expected_coords)
    field_das = _normalize_fields(fields)

    for name, field_da in field_das.items():
        match name:
            case "U":
                field_da = field_da.rename({"y": "y_center"})
            case "V":
                field_da = field_da.rename({"x": "x_center"})
            case _:
                pass
        field_da = field_da.reset_coords(drop=True)
        field_das[name] = field_da

    if "time" in coords.dims:
        if coords.sizes["time"] != 1:
            raise ValueError("Time dimension in coords must be length 1 (i.e., no time-varying grid).")
        coords = coords.isel(time=0).drop_vars("time")

    if (
        len(coords.dims) == 3
    ):  #! This should really be looking at the dimensionality of the lons and lats arrays. Currently having 2D lon lat and 1D depth triggers this `if` clause
        for dim, len_ in coords.sizes.items():
            if len_ == 1:
                # TODO: log statement about selecting along z dim of 1
                coords = coords.isel({dim: 0})
    # if len(coords.dims) != 2: #! This should really be looking at the dimensionality of the lons and lats arrays. Currently having 2D lon lat and 1D depth triggers this `if` clause
    #     raise ValueError("Expected coordinates to be 2 dimensional")
    ds = xr.merge(list(field_das.values()) + [coords])
    ds.attrs.clear()  # Clear global attributes from the merging

    ds = _maybe_rename_variables(ds, _NEMO.varnames_mapping)
    ds = _rename_nemo_depth_dims(ds)
    ds = _maybe_expand_depth_dimension(ds)
    ds = _drop_unused_dimensions_and_coords(ds, _NEMO_DIMENSION_COORD_NAMES)
    ds = _assign_dims_as_coords(ds, _NEMO_DIMENSION_COORD_NAMES)
    ds = _set_coords(ds, _NEMO_DIMENSION_COORD_NAMES)
    ds = _maybe_remove_depth_from_lonlat(ds)
    ds = _set_axis_attrs(ds, _NEMO.axis_varnames)

    expected_axes = set("XYZT")  # TODO: Update after we have support for 2D spatial fields
    if missing_axes := (expected_axes - set(ds.cf.axes)):
        raise ValueError(
            f"Dataset missing CF compliant metadata for axes "
            f"{missing_axes}. Expected 'axis' attribute to be set "
            f"on all dimension axes {expected_axes}. "
            "HINT: Add xarray metadata attribute 'axis' to dimension - e.g., ds['lat'].attrs['axis'] = 'Y'"
        )

    ds = _flip_w_to_positive_down(ds)

    ds = sgrid._attach_sgrid_metadata(
        ds,
        sgrid.SGrid2DMetadata(
            cf_role="grid_topology",
            topology_dimension=2,
            node_dimensions=("x", "y"),
            node_coordinates=("glamf", "gphif"),
            face_dimensions=(
                sgrid.FaceNodePadding("x_center", "x", sgrid.Padding.LOW),
                sgrid.FaceNodePadding("y_center", "y", sgrid.Padding.LOW),
            ),
            vertical_dimensions=(sgrid.FaceNodePadding("depth_center", "depth", sgrid.Padding.HIGH),),
        ),
    )

    # NEMO models are always in degrees
    ds["glamf"].attrs["units"] = "degrees"
    ds["gphif"].attrs["units"] = "degrees"

    # Update to use lon and lat for internal naming
    ds = ds.sgrid.rename({"gphif": "lat", "glamf": "lon"})  # TODO: Logging message about rename
    return ds


def mom6_to_sgrid(*, fields: dict[str, xr.Dataset | xr.DataArray], coords: xr.Dataset) -> xr.Dataset:
    """Create an sgrid-compliant xarray.Dataset from a dataset of MOM6 netcdf files.

    Parameters
    ----------
    fields : dict[str, xr.Dataset | xr.DataArray]
        Dictionary mapping Parcels field names (e.g., "U", "V") to xarray.DataArray objects as obtained from a
        set of MOM6 netcdf files, e.g. ``{"U": ds["ssu"], "V": ds["ssv"]}``. An xarray.Dataset can be given
        instead of a DataArray if it contains a variable with the field name, which is then used (e.g.,
        ``{"U": ds_u}`` uses ``ds_u["U"]``). U must be on ``(yh, xq)`` and V on ``(yq, xh)``. The time
        coordinate must be in ascending order.
    coords : xarray.Dataset
        MOM6 static grid dataset (e.g. ``ocean_static.nc``) containing ``geolon_c`` and ``geolat_c``
        on ``(yq, xq)``.

    Returns
    -------
    xarray.Dataset
        Dataset object following SGRID conventions to be (optionally) modified and passed to a FieldSet constructor.

    Notes
    -----
    MOM6 (https://github.com/mom-ocean/MOM6) uses a curvilinear Arakawa C-grid. The grid geometry is defined
    by the corner points ``geolon_c`` and ``geolat_c``.

    MOM6 output can be written in symmetric or non-symmetric layout. In symmetric output (required for
    regional configurations), each axis has one more corner point than cell centres
    (``len(xq) == len(xh) + 1``); in non-symmetric output they have the same length. The layout is
    detected from the grid sizes. When subsetting symmetric output, keep one more corner point than
    centre points along each axis.

    Currently only 2D (surface) velocity fields are supported.
    """
    coords = _pick_expected_coords(coords, _MOM6.expected_coords)
    field_das = {name: da.reset_coords(drop=True) for name, da in _normalize_fields(fields).items()}

    ds = xr.merge(list(field_das.values()) + [coords])
    ds.attrs.clear()  # Clear global attributes from the merging

    # Ensure only 2D (surface) + time velocity fields are present
    for var_name, da in ds.data_vars.items():
        if da.ndim > 3:
            raise ValueError(
                f"Variable {var_name} has more than 2 spatial dimensions, which is not supported by the mom6_to_sgrid converter."
            )

    ds = _drop_unused_dimensions_and_coords(ds, _MOM6_DIMENSION_COORD_NAMES)
    ds = _assign_dims_as_coords(ds, _MOM6_DIMENSION_COORD_NAMES)
    ds = _set_coords(ds, _MOM6_DIMENSION_COORD_NAMES)
    ds = _set_axis_attrs(ds, _MOM6.axis_varnames)

    for node, face in [("xq", "xh"), ("yq", "yh")]:
        if ds.sizes[node] - ds.sizes[face] not in (0, 1):
            raise ValueError(
                f"Unexpected MOM6 grid sizes: {node}={ds.sizes[node]}, {face}={ds.sizes[face]}. "
                "Expected symmetric (node = face + 1) or non-symmetric (node = face) layout."
            )

    # Symmetric layout (one more corner than centre) -> padding "none"; non-symmetric (equal sizes) -> "low"
    pad_x = sgrid.Padding.NONE if ds.sizes["xq"] == ds.sizes["xh"] + 1 else sgrid.Padding.LOW
    pad_y = sgrid.Padding.NONE if ds.sizes["yq"] == ds.sizes["yh"] + 1 else sgrid.Padding.LOW

    ds = sgrid._attach_sgrid_metadata(
        ds,
        sgrid.SGrid2DMetadata(
            cf_role="grid_topology",
            topology_dimension=2,
            node_dimensions=("xq", "yq"),
            node_coordinates=("geolon_c", "geolat_c"),
            face_dimensions=(
                sgrid.FaceNodePadding("xh", "xq", pad_x),
                sgrid.FaceNodePadding("yh", "yq", pad_y),
            ),
        ),
    )

    # MOM6 geographic coordinates are always in degrees
    ds["geolon_c"].attrs["units"] = "degrees"
    ds["geolat_c"].attrs["units"] = "degrees"

    # Use lon and lat for internal naming
    ds = ds.sgrid.rename({"geolat_c": "lat", "geolon_c": "lon"})
    return ds


def mitgcm_to_sgrid(*, fields: dict[str, xr.Dataset | xr.DataArray], coords: xr.Dataset) -> xr.Dataset:
    """Create an sgrid-compliant xarray.Dataset from a dataset of MITgcm netcdf files.

    Parameters
    ----------
    fields : dict[str, xr.Dataset | xr.DataArray]
        Dictionary mapping Parcels field names (e.g., "U", "V") to xarray.DataArray objects as obtained from a
        set of MITgcm netcdf files, e.g. ``{"U": ds["UVEL"]}``. An xarray.Dataset can be given instead of a
        DataArray if it contains a variable with the field name, which is then used (e.g., ``{"U": ds_u}`` uses
        ``ds_u["U"]``).
    coords : xarray.Dataset
        xarray.Dataset containing coordinate variables.

    Returns
    -------
    xarray.Dataset
        Dataset object following SGRID conventions to be (optionally) modified and passed to a FieldSet constructor.

    Notes
    -----
    See the MITgcm tutorial for more information on how to use MITgcm model outputs in Parcels

    """
    field_das = _normalize_fields(fields)

    coords = _pick_expected_coords(coords, _MITGCM.expected_coords)

    ds = xr.merge(list(field_das.values()) + [coords], compat="override")
    ds.attrs.clear()  # Clear global attributes from the merging

    ds = _maybe_rename_variables(ds, _MITGCM.varnames_mapping)
    ds = _set_axis_attrs(ds, _MITGCM.axis_varnames)
    ds = _maybe_swap_depth_direction(ds)

    ds = sgrid._attach_sgrid_metadata(
        ds,
        sgrid.SGrid2DMetadata(
            cf_role="grid_topology",
            topology_dimension=2,
            node_dimensions=("lon", "lat"),
            node_coordinates=("lon", "lat"),
            face_dimensions=(
                sgrid.FaceNodePadding("XC", "lon", sgrid.Padding.HIGH),
                sgrid.FaceNodePadding("YC", "lat", sgrid.Padding.HIGH),
            ),
            vertical_dimensions=(sgrid.FaceNodePadding("depth", "depth", sgrid.Padding.HIGH),),
        ),
    )

    return ds


def croco_to_sgrid(*, fields: dict[str, xr.Dataset | xr.DataArray], coords: xr.Dataset) -> xr.Dataset:
    """Create an sgrid-compliant xarray.Dataset from a dataset of CROCO netcdf files.

    Parameters
    ----------
    fields : dict[str, xr.Dataset | xr.DataArray]
        Dictionary mapping Parcels field names (e.g., "U", "V") to xarray.DataArray objects as obtained from a
        set of Croco netcdf files, e.g. ``{"U": ds["u"]}``. An xarray.Dataset can be given instead of a
        DataArray if it contains a variable with the field name, which is then used (e.g., ``{"U": ds_u}`` uses
        ``ds_u["U"]``).
    coords : xarray.Dataset
        xarray.Dataset containing coordinate variables.

    Returns
    -------
    xarray.Dataset
        Dataset object following SGRID conventions to be (optionally) modified and passed to a FieldSet constructor.

    Notes
    -----
    See the CROCO 3D tutorial for more information on how to use CROCO model outputs in Parcels

    """
    field_das = _normalize_fields(fields)

    coords = _pick_expected_coords(coords, _CROCO.expected_coords)

    ds = xr.merge(list(field_das.values()) + [coords])
    ds.attrs.clear()  # Clear global attributes from the merging

    ds = _maybe_rename_variables(ds, _CROCO.varnames_mapping)
    ds = _maybe_convert_time_from_float_to_timedelta(ds)

    ds = sgrid._attach_sgrid_metadata(
        ds,
        sgrid.SGrid2DMetadata(
            cf_role="grid_topology",
            topology_dimension=2,
            node_dimensions=("lon", "lat"),
            node_coordinates=("lon", "lat"),
            face_dimensions=(
                sgrid.FaceNodePadding("xi_u", "xi_rho", sgrid.Padding.NONE),
                sgrid.FaceNodePadding("eta_v", "eta_rho", sgrid.Padding.NONE),
            ),
            vertical_dimensions=(sgrid.FaceNodePadding("s_rho", "depth", sgrid.Padding.NONE),),
        ),
    )

    return ds


def copernicusmarine_to_sgrid(
    *, fields: dict[str, xr.Dataset | xr.DataArray], coords: xr.Dataset | None = None
) -> xr.Dataset:
    """Create an sgrid-compliant xarray.Dataset from a dataset of Copernicus Marine netcdf files.

    Parameters
    ----------
    fields : dict[str, xr.Dataset | xr.DataArray]
        Dictionary mapping Parcels field names (e.g., "U", "V") to xarray.DataArray objects as obtained from a
        set of Copernicus Marine netcdf files, e.g. ``{"U": ds["uo"]}``. An xarray.Dataset can be given instead of a
        DataArray if it contains a variable with the field name, which is then used (e.g., ``{"U": ds_u}`` uses
        ``ds_u["U"]``).
    coords : xarray.Dataset, optional
        xarray.Dataset containing coordinate variables. By default these are time, depth, latitude, longitude

    Returns
    -------
    xarray.Dataset
        Dataset object following SGRID conventions to be (optionally) modified and passed to a FieldSet constructor.

    Notes
    -----
    See https://help.marine.copernicus.eu/en/collections/9080063-copernicus-marine-toolbox for more information on the copernicusmarine toolbox.
    The toolbox to ingest data from most of the products on the Copernicus Marine Service (https://data.marine.copernicus.eu/products) into an xarray.Dataset.
    You can use indexing and slicing to select a subset of the data before passing it to this function.

    """
    field_das = _normalize_fields(fields)

    ds = xr.merge(list(field_das.values()) + ([coords] if coords is not None else []))
    ds.attrs.clear()  # Clear global attributes from the merging

    ds = _maybe_rename_coords(ds, _COPERNICUS_MARINE_AXIS_VARNAMES)
    ds = _maybe_expand_depth_dimension(ds)

    ds = _flip_w_to_positive_down(ds)

    # use dummy *_center dimensions - this is A grid data (all defined on nodes)
    ds = sgrid._attach_sgrid_metadata(
        ds,
        sgrid.SGrid2DMetadata(
            cf_role="grid_topology",
            topology_dimension=2,
            node_dimensions=("lon", "lat"),
            node_coordinates=("lon", "lat"),
            face_dimensions=(
                sgrid.FaceNodePadding("x_center", "lon", sgrid.Padding.LOW),
                sgrid.FaceNodePadding("y_center", "lat", sgrid.Padding.LOW),
            ),
            vertical_dimensions=(sgrid.FaceNodePadding("depth_center", "depth", sgrid.Padding.LOW),),
        ),
    )

    return ds


def delft3d_to_sgrid(*, fields: dict[str, xr.Dataset | xr.DataArray], coords: xr.Dataset) -> xr.Dataset:
    """Create an sgrid-compliant xarray.Dataset from a dataset of structured-grid Delft3D netcdf files.

    Parameters
    ----------
    fields : dict[str, xr.Dataset | xr.DataArray]
        Dictionary mapping Parcels field names (e.g., "U", "V") to xarray.DataArray objects as obtained from a
        set of structured-grid Delft3D netcdf files, e.g. ``{"U": ds["VELU"]}``. An xarray.Dataset can be given
        instead of a DataArray if it contains a variable with the field name, which is then used (e.g.,
        ``{"U": ds_u}`` uses ``ds_u["U"]``).
    coords : xarray.Dataset
        xarray.Dataset containing coordinate variables.

    Returns
    -------
    xarray.Dataset
        Dataset object following SGRID conventions to be (optionally) modified and passed to a FieldSet constructor.

    """
    warnings.warn(
        "The delft3d_to_sgrid function is experimental and may not work for all Delft3D datasets. "
        "Furthermore, we are not entirely confident that the SGrid layout for Delft3D is implemented correctly. "
        "Please report any issues to the Parcels GitHub repository.",
        UserWarning,
        stacklevel=2,
    )

    coords = _pick_expected_coords(coords, _DELFT3D_X.expected_coords)
    field_das = _normalize_fields(fields)

    ds = xr.merge(list(field_das.values()) + [coords], compat="override")
    ds.attrs.clear()  # Clear global attributes from the merging

    ds = _maybe_rename_variables(ds, _DELFT3D_X.varnames_mapping)
    ds = _set_coords(ds, _DELFT3D_X.axis_varnames.keys())
    ds = _set_axis_attrs(ds, _DELFT3D_X.axis_varnames)

    ds = sgrid._attach_sgrid_metadata(
        ds,
        sgrid.SGrid2DMetadata(
            cf_role="grid_topology",
            topology_dimension=2,
            node_dimensions=("M", "N"),
            node_coordinates=("lon", "lat"),
            face_dimensions=(
                sgrid.FaceNodePadding("X", "M", sgrid.Padding.LOW),
                sgrid.FaceNodePadding("Y", "N", sgrid.Padding.LOW),
            ),
            vertical_dimensions=(sgrid.FaceNodePadding("Z", "LAYER", sgrid.Padding.HIGH),),
        ),
    )

    return ds


def _detect_vertical_coordinates(
    ds: ux.UxDataset,
    known_mappings: dict[str, str] | None = None,
) -> tuple[str, str]:
    """Detect vertical coordinate dimensions for faces (zf) and centers (zc).

    Detection strategy (with fallback):
    1. Use known_mappings if provided and dimensions exist
    2. Look for CF convention axis='Z' metadata
    3. Find dimension pairs where sizes differ by exactly 1

    Parameters
    ----------
    ds : ux.UxDataset
        UxDataset to analyze for vertical coordinates.
    known_mappings : dict[str, str] | None
        Optional mapping with keys "interface" and "center" specifying
        the dimension names for layer interfaces (zf) and centers (zc).

    Returns
    -------
    tuple[str, str]
        Tuple of (interface_dim_name, center_dim_name).

    Raises
    ------
    ValueError
        If vertical coordinates cannot be detected.
    """
    ds_dims = cast(set[str], set(ds.dims))

    # Strategy 1: Use known mappings if provided and dimensions exist
    if known_mappings is not None:
        interface_dim = known_mappings.get("interface")
        center_dim = known_mappings.get("center")
        if interface_dim in ds_dims and center_dim in ds_dims:
            logger.info(
                f"Using known vertical dimension mapping: {interface_dim!r} (interfaces) and {center_dim!r} (centers)."
            )
            return interface_dim, center_dim
        logger.debug(f"Known mappings {known_mappings} not found in dataset dimensions {ds_dims}. Trying CF metadata.")

    # Strategy 2: Look for CF convention axis='Z' metadata
    z_dims = []
    for dim in ds_dims:
        if dim in ds.coords:
            coord = ds.coords[dim]
            if coord.attrs.get("axis") == "Z":
                z_dims.append(dim)
            elif coord.attrs.get("positive") in ("up", "down"):
                z_dims.append(dim)
            elif "depth" in coord.attrs.get("standard_name", "").lower():
                z_dims.append(dim)

    if len(z_dims) == 2:
        # Sort by size - interface has n+1 values, center has n
        z_dims_sorted = sorted(z_dims, key=lambda d: ds.sizes[d], reverse=True)
        interface_dim, center_dim = z_dims_sorted
        if ds.sizes[interface_dim] == ds.sizes[center_dim] + 1:
            logger.info(
                f"Detected vertical dimensions from CF metadata: {interface_dim!r} (interfaces) and {center_dim!r} (centers)."
            )
            return interface_dim, center_dim

    # Strategy 3: Find dimension pairs where sizes differ by exactly 1
    # Skip known non-vertical dimensions
    skip_dims = {"time", "n_face", "n_node", "n_edge", "n_max_face_nodes"}
    candidate_dims = [d for d in ds_dims if d not in skip_dims]

    for dim1 in candidate_dims:
        for dim2 in candidate_dims:
            if dim1 != dim2:
                size1, size2 = ds.sizes[dim1], ds.sizes[dim2]
                if size1 == size2 + 1:
                    logger.info(
                        f"Auto-detected vertical dimensions by size difference: {dim1!r} (interfaces, size={size1}) "
                        f"and {dim2!r} (centers, size={size2})."
                    )
                    return dim1, dim2

    raise ValueError(
        f"Could not detect vertical coordinate dimensions in dataset with dims {list(ds_dims)}. "
        "Please ensure the dataset has vertical layer interface and center dimensions, "
        "or rename them manually to 'zf' (interfaces) and 'zc' (centers)."
    )


def _rename_vertical_dims(
    ds: ux.UxDataset,
    interface_dim: str,
    center_dim: str,
) -> ux.UxDataset:
    """Rename vertical dimensions to zf (interfaces) and zc (centers).

    Parameters
    ----------
    ds : ux.UxDataset
        UxDataset with vertical dimensions to rename.
    interface_dim : str
        Current name of the interface dimension.
    center_dim : str
        Current name of the center dimension.

    Returns
    -------
    ux.UxDataset
        Dataset with renamed dimensions and indexed coordinates.
    """
    rename_dict = {}
    if interface_dim != "zf":
        rename_dict[interface_dim] = "zf"
    if center_dim != "zc":
        rename_dict[center_dim] = "zc"

    if rename_dict:
        logger.info(f"Renaming vertical dimensions: {rename_dict}")
        ds = ds.rename(rename_dict)

    ds = ds.set_index(zf="zf", zc="zc")
    return ds


def fesom_to_ugrid(ds: ux.UxDataset) -> ux.UxDataset:
    """Convert FESOM2 UxDataset to Parcels UGRID-compliant format.

    Renames vertical dimensions:
    - nz -> zf (vertical layer faces/interfaces)
    - nz1 -> zc (vertical layer centers)
    - nod2 -> n_node (node)
    - elem -> n_face (face)

    Parameters
    ----------
    ds : ux.UxDataset
        FESOM2 UxDataset as obtained from uxarray.

    Returns
    -------
    ux.UxDataset
        UGRID-compliant dataset ready for FieldSet.from_ugrid_conventions().

    Examples
    --------
    >>> import uxarray as ux
    >>> from parcels import FieldSet
    >>> from parcels.convert import fesom_to_ugrid
    >>> ds = ux.open_mfdataset(grid_path, data_path)
    >>> ds_ugrid = fesom_to_ugrid(ds)
    >>> fieldset = FieldSet.from_ugrid_conventions(ds_ugrid)
    """
    ds = ds.copy()

    for try_dim, target in [("elem", "n_face"), ("nod2", "n_node")]:
        if try_dim in ds.dims:
            ds = ds.rename_dims({try_dim: target})

    interface_dim, center_dim = _detect_vertical_coordinates(ds, _FESOM2_VERTICAL_DIMS)
    return _rename_vertical_dims(ds, interface_dim, center_dim)


def icon_to_ugrid(ds: ux.UxDataset) -> ux.UxDataset:
    """Convert ICON UxDataset to Parcels UGRID-compliant format.

    Renames vertical dimensions:
    - depth_2 -> zf (vertical layer faces/interfaces)
    - depth -> zc (vertical layer centers)

    Parameters
    ----------
    ds : ux.UxDataset
        ICON UxDataset as obtained from uxarray.

    Returns
    -------
    ux.UxDataset
        UGRID-compliant dataset ready for FieldSet.from_ugrid_conventions().

    Examples
    --------
    >>> import uxarray as ux
    >>> from parcels import FieldSet
    >>> from parcels.convert import icon_to_ugrid
    >>> ds = ux.open_mfdataset(grid_path, data_path)
    >>> ds_ugrid = icon_to_ugrid(ds)
    >>> fieldset = FieldSet.from_ugrid_conventions(ds_ugrid)
    """
    ds = ds.copy()
    interface_dim, center_dim = _detect_vertical_coordinates(ds, _ICON_VERTICAL_DIMS)
    return _rename_vertical_dims(ds, interface_dim, center_dim)
