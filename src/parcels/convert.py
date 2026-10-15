"""
This module provide a series of functions model outputs (which might be following their
own conventions) to metadata rich data (i.e., data following SGRID/UGRID as well as CF
conventions). Parcels needs this metadata rich data to discover grid geometries among other
things.

These functions use knowledge about the model to attach any missing metadata. The functions
emit verbose messaging so that the user is kept in the loop. The returned output is an
Xarray dataset so that users can further provide any missing metadata that was unable to
be determined before they pass it to the FieldSet constructor.
"""

from __future__ import annotations

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


_NEMO_EXPECTED_COORDS: list[tuple[str, _Status]] = [
    ("glamf", _Status.REQUIRED),
    ("gphif", _Status.REQUIRED),
    ("depthw", _Status.OPTIONAL),
]

_NEMO_DIMENSION_COORD_NAMES: list[str] = [
    "x",
    "y",
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

_NEMO_AXIS_VARNAMES: dict[str, XgcmAxisDirection] = {
    "x": "X",
    "x_center": "X",
    "y": "Y",
    "y_center": "Y",
    "depth": "Z",
    "depth_center": "Z",
    "time": "T",
}

_NEMO_VARNAMES_MAPPING: dict[str, str] = {
    "time_counter": "time",
    "depthw": "depth",
    "uo": "U",
    "vo": "V",
    "wo": "W",
}

_MITGCM_EXPECTED_COORDS: list[tuple[str, _Status]] = [(name, _Status.REQUIRED) for name in ["XG", "YG", "Zl"]]

_MITGCM_AXIS_VARNAMES: dict[str, XgcmAxisDirection] = {
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
}

_MITGCM_VARNAMES_MAPPING: dict[str, str] = {
    "XG": "lon",
    "YG": "lat",
    "Zl": "depth",
}

_COPERNICUS_MARINE_AXIS_VARNAMES: dict[XgcmAxisDirection, str] = {
    "X": "lon",
    "Y": "lat",
    "Z": "depth",
    "T": "time",
}

_DELFT3D_X_EXPECTED_COORDS: list[tuple[str, _Status]] = [(name, _Status.REQUIRED) for name in ["XZETA", "YZETA"]]

_DELFT3D_X_VARNAMES_MAPPING: dict[str, str] = {
    "XZETA": "lon",
    "YZETA": "lat",
    "SIGMA_C": "depth",
    "TIME": "time",
}

_DELFT3D_X_AXIS_VARNAMES: dict[str, XgcmAxisDirection] = {
    "M": "X",
    "N": "Y",
    "LAYER": "Z",
    "time": "T",
}


_CROCO_EXPECTED_COORDS: list[tuple[str, _Status]] = [
    (name, _Status.REQUIRED) for name in ["x_rho", "y_rho", "s_w", "time"]
]

_CROCO_VARNAMES_MAPPING: dict[str, str] = {
    "x_rho": "lon",
    "y_rho": "lat",
    "s_w": "depth",
}


def _pick_expected_coords(coords: xr.Dataset, expected_coord_names: list[tuple[str, _Status]]) -> xr.Dataset:
    coords_to_use = {}
    for name, status in expected_coord_names:
        if name in coords:
            coords_to_use[name] = coords[name]
        else:
            if status == _Status.REQUIRED:
                raise ValueError(f"Expected coordinate '{name}' not found in provided coords dataset.")
    return xr.Dataset(coords_to_use)


def _maybe_bring_other_depths_to_depth(ds: xr.Dataset):
    for var in ds.data_vars:
        for old_depth, target in [
            ("depthu", "depth_center"),
            ("depthv", "depth_center"),
            ("deptht", "depth_center"),
            ("depthw", "depth"),
        ]:
            if old_depth in ds[var].dims:
                ds[var] = ds[var].rename({old_depth: target})

    if "depth" not in ds.dims:
        logger.info("No depth dimension found in your dataset. Assuming no depth (i.e., surface data).", stacklevel=1)
        ds = ds.expand_dims({"depth": [0]})
        ds["depth"] = xr.DataArray([0], dims=["depth"])
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


def _ds_rename_using_standard_names(ds: xr.Dataset | ux.UxDataset, name_dict: dict[str, str]) -> xr.Dataset:
    for standard_name, rename_to in name_dict.items():
        name = ds.cf[standard_name].name
        ds = ds.rename({name: rename_to})
        logger.info(
            f"cf_xarray found variable {name!r} with CF standard name {standard_name!r} in dataset, renamed it to {rename_to!r} for Parcels simulation."
        )
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
        ds = ds.expand_dims(dim={"depth": [0]}, axis=1)
        logger.info("No depth dimension found in dataset. Added a singleton depth dimension.")
    return ds


# TODO is this function still needed, now that we require users to provide field names explicitly?
def _discover_U_and_V(ds: xr.Dataset, cf_standard_names_fallbacks) -> xr.Dataset:
    # Assumes that the dataset has U and V data

    if "W" not in ds:
        for cf_standard_name_W in cf_standard_names_fallbacks["W"]:
            if cf_standard_name_W in ds.cf.standard_names:
                ds = _ds_rename_using_standard_names(ds, {cf_standard_name_W: "W"})
                break

    if "U" in ds and "V" in ds:
        return ds  # U and V already present
    elif "U" in ds or "V" in ds:
        raise ValueError(
            "Dataset has only one of the two variables 'U' and 'V'. Please rename the appropriate variable in your dataset to have both 'U' and 'V' for Parcels simulation."
        )

    for cf_standard_name_U, cf_standard_name_V in cf_standard_names_fallbacks["UV"]:
        if cf_standard_name_U in ds.cf.standard_names:
            if cf_standard_name_V not in ds.cf.standard_names:
                raise ValueError(
                    f"Dataset has variable with CF standard name {cf_standard_name_U!r}, "
                    f"but not the matching variable with CF standard name {cf_standard_name_V!r}. "
                    "Please rename the appropriate variables in your dataset to have both 'U' and 'V' for Parcels simulation."
                )
        else:
            continue

        ds = _ds_rename_using_standard_names(ds, {cf_standard_name_U: "U", cf_standard_name_V: "V"})
        break
    return ds


def _assert_no_grid_metadata(ds: xr.Dataset) -> None:
    if "grid" in ds.cf.cf_roles:
        raise ValueError(
            "Dataset already has a 'grid' variable (according to cf_roles). Didn't expect there to be grid metadata on this dataset - please open an issue with more information about your dataset."
        )


def nemo_to_sgrid(*, fields: dict[str, xr.Dataset | xr.DataArray], coords: xr.Dataset):
    # TODO: Update docstring
    """Create a FieldSet from a xarray.Dataset from NEMO netcdf files.

    Parameters
    ----------
    ds : xarray.Dataset
        xarray.Dataset as obtained from a set of NEMO netcdf files.

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
    fields = fields.copy()
    coords = _pick_expected_coords(coords, _NEMO_EXPECTED_COORDS)

    for name, field_da in fields.items():
        if isinstance(field_da, xr.Dataset):
            field_da = field_da[name]
            # TODO: logging message, warn if multiple fields are in this dataset
        else:
            field_da = field_da.rename(name)

        match name:
            case "U":
                field_da = field_da.rename({"y": "y_center"})
            case "V":
                field_da = field_da.rename({"x": "x_center"})
            case _:
                pass
        field_da = field_da.reset_coords(drop=True)
        fields[name] = field_da

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
    ds = xr.merge(list(fields.values()) + [coords])
    ds = _maybe_rename_variables(ds, _NEMO_VARNAMES_MAPPING)
    ds = _maybe_bring_other_depths_to_depth(ds)
    ds = _drop_unused_dimensions_and_coords(ds, _NEMO_DIMENSION_COORD_NAMES)
    ds = _assign_dims_as_coords(ds, _NEMO_DIMENSION_COORD_NAMES)
    ds = _set_coords(ds, _NEMO_DIMENSION_COORD_NAMES)
    ds = _maybe_remove_depth_from_lonlat(ds)
    ds = _set_axis_attrs(ds, _NEMO_AXIS_VARNAMES)

    expected_axes = set("XYZT")  # TODO: Update after we have support for 2D spatial fields
    if missing_axes := (expected_axes - set(ds.cf.axes)):
        raise ValueError(
            f"Dataset missing CF compliant metadata for axes "
            f"{missing_axes}. Expected 'axis' attribute to be set "
            f"on all dimension axes {expected_axes}. "
            "HINT: Add xarray metadata attribute 'axis' to dimension - e.g., ds['lat'].attrs['axis'] = 'Y'"
        )

    if "W" in ds.data_vars:
        # Negate W to convert from up positive to down positive (as that's the direction of positive z)
        ds["W"].data *= -1

    _assert_no_grid_metadata(ds)
    ds["grid"] = xr.DataArray(
        0,
        attrs=sgrid.SGrid2DMetadata(
            cf_role="grid_topology",
            topology_dimension=2,
            node_dimensions=("x", "y"),
            node_coordinates=("glamf", "gphif"),
            face_dimensions=(
                sgrid.FaceNodePadding("x_center", "x", sgrid.Padding.LOW),
                sgrid.FaceNodePadding("y_center", "y", sgrid.Padding.LOW),
            ),
            vertical_dimensions=(sgrid.FaceNodePadding("depth_center", "depth", sgrid.Padding.HIGH),),
        ).to_attrs(),
    )

    # NEMO models are always in degrees
    ds["glamf"].attrs["units"] = "degrees"
    ds["gphif"].attrs["units"] = "degrees"

    # Update to use lon and lat for internal naming
    ds = ds.sgrid.rename({"gphif": "lat", "glamf": "lon"})  # TODO: Logging message about rename
    return ds


def mitgcm_to_sgrid(*, fields: dict[str, xr.Dataset | xr.DataArray], coords: xr.Dataset) -> xr.Dataset:
    """Create an sgrid-compliant xarray.Dataset from a dataset of MITgcm netcdf files.

    Parameters
    ----------
    fields : dict[str, xr.Dataset | xr.DataArray]
        Dictionary of xarray.DataArray objects as obtained from a set of MITgcm netcdf files.
    coords : xarray.Dataset, optional
        xarray.Dataset containing coordinate variables.

    Returns
    -------
    xarray.Dataset
        Dataset object following SGRID conventions to be (optionally) modified and passed to a FieldSet constructor.

    Notes
    -----
    See the MITgcm tutorial for more information on how to use MITgcm model outputs in Parcels

    """
    fields = fields.copy()

    for name, field_da in fields.items():
        if isinstance(field_da, xr.Dataset):
            field_da = field_da[name]
            # TODO: logging message, warn if multiple fields are in this dataset
        else:
            field_da = field_da.rename(name)
        fields[name] = field_da

    coords = _pick_expected_coords(coords, _MITGCM_EXPECTED_COORDS)

    ds = xr.merge(list(fields.values()) + [coords], compat="override")
    ds.attrs.clear()  # Clear global attributes from the merging

    ds = _maybe_rename_variables(ds, _MITGCM_VARNAMES_MAPPING)
    ds = _set_axis_attrs(ds, _MITGCM_AXIS_VARNAMES)
    ds = _maybe_swap_depth_direction(ds)

    _assert_no_grid_metadata(ds)
    ds["grid"] = xr.DataArray(
        0,
        attrs=sgrid.SGrid2DMetadata(
            cf_role="grid_topology",
            topology_dimension=2,
            node_dimensions=("lon", "lat"),
            node_coordinates=("lon", "lat"),
            face_dimensions=(
                sgrid.FaceNodePadding("XC", "lon", sgrid.Padding.HIGH),
                sgrid.FaceNodePadding("YC", "lat", sgrid.Padding.HIGH),
            ),
            vertical_dimensions=(sgrid.FaceNodePadding("depth", "depth", sgrid.Padding.HIGH),),
        ).to_attrs(),
    )

    return ds


def croco_to_sgrid(*, fields: dict[str, xr.Dataset | xr.DataArray], coords: xr.Dataset) -> xr.Dataset:
    """Create an sgrid-compliant xarray.Dataset from a dataset of CROCO netcdf files.

    Parameters
    ----------
    fields : dict[str, xr.Dataset | xr.DataArray]
        Dictionary of xarray.DataArray objects as obtained from a set of Croco netcdf files.
    coords : xarray.Dataset, optional
        xarray.Dataset containing coordinate variables.

    Returns
    -------
    xarray.Dataset
        Dataset object following SGRID conventions to be (optionally) modified and passed to a FieldSet constructor.

    Notes
    -----
    See the CROCO 3D tutorial for more information on how to use CROCO model outputs in Parcels

    """
    fields = fields.copy()

    for name, field_da in fields.items():
        if isinstance(field_da, xr.Dataset):
            field_da = field_da[name]
            # TODO: logging message, warn if multiple fields are in this dataset
        else:
            field_da = field_da.rename(name)
        fields[name] = field_da

    coords = _pick_expected_coords(coords, _CROCO_EXPECTED_COORDS)

    ds = xr.merge(list(fields.values()) + [coords])
    ds.attrs.clear()  # Clear global attributes from the merging

    ds = _maybe_rename_variables(ds, _CROCO_VARNAMES_MAPPING)
    ds = _maybe_convert_time_from_float_to_timedelta(ds)

    _assert_no_grid_metadata(ds)
    ds["grid"] = xr.DataArray(
        0,
        attrs=sgrid.SGrid2DMetadata(
            cf_role="grid_topology",
            topology_dimension=2,
            node_dimensions=("lon", "lat"),
            node_coordinates=("lon", "lat"),
            face_dimensions=(
                sgrid.FaceNodePadding("xi_u", "xi_rho", sgrid.Padding.NONE),
                sgrid.FaceNodePadding("eta_v", "eta_rho", sgrid.Padding.NONE),
            ),
            vertical_dimensions=(sgrid.FaceNodePadding("s_rho", "depth", sgrid.Padding.NONE),),
        ).to_attrs(),
    )

    return ds


def copernicusmarine_to_sgrid(
    *, fields: dict[str, xr.Dataset | xr.DataArray], coords: xr.Dataset | None = None
) -> xr.Dataset:
    """Create an sgrid-compliant xarray.Dataset from a dataset of Copernicus Marine netcdf files.

    Parameters
    ----------
    fields : dict[str, xr.Dataset | xr.DataArray]
        Dictionary of xarray.DataArray objects as obtained from a set of Copernicus Marine netcdf files.
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
    fields = fields.copy()

    for name, field_da in fields.items():
        if isinstance(field_da, xr.Dataset):
            field_da = field_da[name]
            # TODO: logging message, warn if multiple fields are in this dataset
        else:
            field_da = field_da.rename(name)
        fields[name] = field_da

    ds = xr.merge(list(fields.values()) + ([coords] if coords is not None else []))
    ds.attrs.clear()  # Clear global attributes from the merging

    ds = _maybe_rename_coords(ds, _COPERNICUS_MARINE_AXIS_VARNAMES)
    ds = _maybe_expand_depth_dimension(ds)

    if "W" in ds.data_vars:
        # Negate W to convert from up positive to down positive (as that's the direction of positive z)
        ds["W"].data *= -1

    _assert_no_grid_metadata(ds)
    ds["grid"] = xr.DataArray(
        0,
        attrs=sgrid.SGrid2DMetadata(  # use dummy *_center dimensions - this is A grid data (all defined on nodes)
            cf_role="grid_topology",
            topology_dimension=2,
            node_dimensions=("lon", "lat"),
            node_coordinates=("lon", "lat"),
            face_dimensions=(
                sgrid.FaceNodePadding("x_center", "lon", sgrid.Padding.LOW),
                sgrid.FaceNodePadding("y_center", "lat", sgrid.Padding.LOW),
            ),
            vertical_dimensions=(sgrid.FaceNodePadding("depth_center", "depth", sgrid.Padding.LOW),),
        ).to_attrs(),
    )

    return ds


def delft3d_to_sgrid(*, fields: dict[str, xr.Dataset | xr.DataArray], coords: xr.Dataset) -> xr.Dataset:
    """Create an sgrid-compliant xarray.Dataset from a dataset of structured-grid Delft3D netcdf files.

    Parameters
    ----------
    fields : dict[str, xr.Dataset | xr.DataArray]
        Dictionary of xarray.DataArray objects as obtained from a set of structured-grid Delft3D netcdf files.
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

    fields = fields.copy()
    coords = _pick_expected_coords(coords, _DELFT3D_X_EXPECTED_COORDS)

    for name, field_da in fields.items():
        if isinstance(field_da, xr.Dataset):
            field_da = field_da[name]
            # TODO: logging message, warn if multiple fields are in this dataset
        else:
            field_da = field_da.rename(name)
        fields[name] = field_da

    ds = xr.merge(list(fields.values()) + [coords], compat="override")

    ds = _maybe_rename_variables(ds, _DELFT3D_X_VARNAMES_MAPPING)
    ds = _set_coords(ds, _DELFT3D_X_AXIS_VARNAMES.keys())
    ds = _set_axis_attrs(ds, _DELFT3D_X_AXIS_VARNAMES)

    ds["grid"] = xr.DataArray(
        0,
        attrs=sgrid.SGrid2DMetadata(
            cf_role="grid_topology",
            topology_dimension=2,
            node_dimensions=("M", "N"),
            node_coordinates=("lon", "lat"),
            face_dimensions=(
                sgrid.FaceNodePadding("X", "M", sgrid.Padding.LOW),
                sgrid.FaceNodePadding("Y", "N", sgrid.Padding.LOW),
            ),
            vertical_dimensions=(sgrid.FaceNodePadding("Z", "LAYER", sgrid.Padding.HIGH),),
        ).to_attrs(),
    )

    return ds


# Known vertical dimension mappings by model
_FESOM2_VERTICAL_DIMS = {"interface": "nz", "center": "nz1"}
_ICON_VERTICAL_DIMS = {"interface": "depth_2", "center": "depth"}


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
    >>> fieldset = FieldSet.from_ugrid_conventions(ds_ugrid, mesh="flat")
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
    >>> fieldset = FieldSet.from_ugrid_conventions(ds_ugrid, mesh="flat")
    """
    ds = ds.copy()
    interface_dim, center_dim = _detect_vertical_coordinates(ds, _ICON_VERTICAL_DIMS)
    return _rename_vertical_dims(ds, interface_dim, center_dim)
