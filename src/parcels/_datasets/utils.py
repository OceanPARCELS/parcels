from collections.abc import Hashable
from typing import Any, Literal

import numpy as np
import xarray as xr

from parcels._logger import logger

_SUPPORTED_ATTR_TYPES = int | float | str | np.ndarray


def _log_mismatched_keys(d1: dict[Any, Any], d2: dict[Any, Any]) -> None:
    k1 = set(d1.keys())
    k2 = set(d2.keys())
    if len(k1 ^ k2) == 0:
        return
    logger.info("Mismatched keys:")
    logger.info(f"L: {k1 - k2!r}")
    logger.info(f"R: {k2 - k1!r}")


def assert_common_attrs_equal(
    xr_attrs_1: dict[str, _SUPPORTED_ATTR_TYPES], xr_attrs_2: dict[str, _SUPPORTED_ATTR_TYPES], *, verbose: bool = True
) -> None:
    d1, d2 = xr_attrs_1, xr_attrs_2

    common_keys = set(d1.keys()) & set(d2.keys())
    if verbose:
        _log_mismatched_keys(d1, d2)

    for key in common_keys:
        try:
            if isinstance(d1[key], np.ndarray):
                np.testing.assert_array_equal(d1[key], d2[key])
            else:
                assert d1[key] == d2[key], f"{d1[key]} != {d2[key]}"
        except AssertionError as e:
            e.add_note(f"error on key {key!r}")
            raise


def assert_common_variables_common_attrs_equal(ds1: xr.Dataset, ds2: xr.Dataset, *, verbose: bool = True) -> None:
    if verbose:
        logger.debug("Checking dataset attrs...")

    assert_common_attrs_equal(ds1.attrs, ds2.attrs, verbose=verbose)

    ds1_vars = set(ds1.variables)
    ds2_vars = set(ds2.variables)

    common_variables = ds1_vars & ds2_vars
    if len(ds1_vars ^ ds2_vars) > 0 and verbose:
        logger.info("Mismatched variables:")
        logger.info(f"L: {ds1_vars - ds2_vars}")
        logger.info(f"R: {ds2_vars - ds1_vars}")

    for var in common_variables:
        if verbose:
            logger.debug(f"Checking {var!r} attrs")
        assert_common_attrs_equal(ds1[var].attrs, ds2[var].attrs, verbose=verbose)


def dataset_repr_diff(ds1: xr.Dataset, ds2: xr.Dataset) -> str:
    """Return a text diff of two datasets."""
    repr1 = repr(ds1)
    repr2 = repr(ds2)
    import difflib

    diff = difflib.ndiff(repr1.splitlines(keepends=True), repr2.splitlines(keepends=True))
    return "".join(diff)


def _dicts_equal(d1, d2):
    # compare two dictionaries, including when their entries are lists or arrays ( == throws an error then)
    if d1.keys() != d2.keys():
        return False
    for k in d1:
        v1, v2 = d1[k], d2[k]
        # Compare lists or arrays element-wise
        if isinstance(v1, (list, np.ndarray)) and isinstance(v2, (list, np.ndarray)):
            if not np.array_equal(np.array(v1), np.array(v2)):
                return False
        else:
            if v1 != v2:
                return False
    return True


def compare_datasets(ds1, ds2, ds1_name="Dataset 1", ds2_name="Dataset 2", verbose=True):
    logger.info(f"Comparing {ds1_name} and {ds2_name}")

    def log_verbose(msg: str) -> None:
        if verbose:
            logger.debug(msg)

    log_verbose("Dataset Attributes Comparison:")
    if ds1.attrs == ds2.attrs:
        log_verbose("  Dataset attributes are identical.")
    else:
        logger.info("  Dataset attributes differ.")
        for attr_name in set(ds1.attrs.keys()) | set(ds2.attrs.keys()):
            if attr_name not in ds1.attrs:
                logger.info(f"    Attribute '{attr_name}' only in {ds2_name}")
            elif attr_name not in ds2.attrs:
                logger.info(f"    Attribute '{attr_name}' only in {ds1_name}")
            elif ds1.attrs[attr_name] != ds2.attrs[attr_name]:
                logger.info(f"    Attribute '{attr_name}' differs:")
                logger.info(f"      {ds1_name}: {ds1.attrs[attr_name]}")
                logger.info(f"      {ds2_name}: {ds2.attrs[attr_name]}")
    log_verbose("-" * 30)

    # Compare dimensions
    log_verbose("Dimensions Comparison:")
    ds1_dims = set(ds1.dims)
    ds2_dims = set(ds2.dims)
    if ds1_dims == ds2_dims:
        log_verbose("  Dimension names are identical.")
    else:
        logger.info("  Dimension names differ:")
        logger.info(f"    {ds1_name} dims: {sorted(list(ds1_dims))}")
        logger.info(f"    {ds2_name} dims: {sorted(list(ds2_dims))}")

    # For common dimensions, compare order (implicit by comparing coordinate values for sortedness)
    # and size (though size is parameterized and expected to be different)
    for dim_name in ds1_dims.intersection(ds2_dims):
        log_verbose(f"  Dimension '{dim_name}':")
        # Sizes will differ due to DIM_SIZE, so we don't strictly compare them.
        log_verbose(f"    {ds1_name} size: {ds1.sizes[dim_name]}, {ds2_name} size: {ds2.sizes[dim_name]}")
        # Check if coordinates associated with dimensions are sorted (increasing)
        if dim_name in ds1.coords and dim_name in ds2.coords:
            check_val = (
                np.timedelta64(0, "s") if isinstance(ds1[dim_name].values[0], (np.datetime64, np.timedelta64)) else 0.0
            )
            is_ds1_sorted = (
                np.all(np.diff(ds1[dim_name].values) >= check_val) if len(ds1[dim_name].values) > 1 else True
            )
            is_ds2_sorted = (
                np.all(np.diff(ds2[dim_name].values) >= check_val) if len(ds2[dim_name].values) > 1 else True
            )
            if is_ds1_sorted == is_ds2_sorted:
                log_verbose(f"    Order for '{dim_name}' is consistent (both sorted: {is_ds1_sorted})")
            else:
                logger.info(
                    f"    Order for '{dim_name}' differs: {ds1_name} sorted: {is_ds1_sorted}, {ds2_name} sorted: {is_ds2_sorted}"
                )
    log_verbose("-" * 30)

    # Compare variables (name, attributes, dimensions used)
    log_verbose("Variables Comparison:")
    ds1_vars = set(ds1.variables.keys())
    ds2_vars = set(ds2.variables.keys())

    if ds1_vars == ds2_vars:
        log_verbose("  Variable names are identical.")
    else:
        logger.info("  Variable names differ:")
        logger.info(f"    {ds1_name} vars: {sorted(list(ds1_vars - ds2_vars))}")
        logger.info(f"    {ds2_name} vars: {sorted(list(ds2_vars - ds1_vars))}")
        logger.info(f"    Common vars: {sorted(list(ds1_vars.intersection(ds2_vars)))}")

    for var_name in ds1_vars.intersection(ds2_vars):
        log_verbose(f"  Variable '{var_name}':")
        var1 = ds1[var_name]
        var2 = ds2[var_name]

        # Compare attributes
        if _dicts_equal(var1.attrs, var2.attrs):
            log_verbose("    Attributes are identical.")
        else:
            logger.info("    Attributes differ.")
            for attr_name in set(var1.attrs.keys()) | set(var2.attrs.keys()):
                if attr_name not in var1.attrs:
                    logger.info(f"      Attribute '{attr_name}' only in {ds2_name}'s '{var_name}'")
                elif attr_name not in var2.attrs:
                    logger.info(f"      Attribute '{attr_name}' only in {ds1_name}'s '{var_name}'")
                elif var1.attrs[attr_name] != var2.attrs[attr_name]:
                    logger.info(f"      Attribute '{attr_name}' differs for '{var_name}':")
                    logger.info(f"        {ds1_name}: {var1.attrs[attr_name]}")
                    logger.info(f"        {ds2_name}: {var2.attrs[attr_name]}")

        # Compare dimensions used by the variable
        if var1.dims == var2.dims:
            log_verbose(f"    Dimensions used are identical: {var1.dims}")
        else:
            logger.info("    Dimensions used differ:")
            logger.info(f"      {ds1_name}: {var1.dims}")
            logger.info(f"      {ds2_name}: {var2.dims}")
    log_verbose("=" * 30 + " End of Comparison " + "=" * 30)


def replace_arrays_with_zeros(
    ds: xr.Dataset, except_for: Literal["coords"] | list[Hashable] | None = None
) -> xr.Dataset:
    """Replace datavars in the xarray dataset with zeros, except for some.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset whose arrays will be replaced with zeros.
    except_for : "coords" or list of Hashable or None, optional
        Controls which arrays are preserved:

        - ``None``: Replace all arrays with zeros.
        - ``"coords"``: Replace all arrays with zeros except the non-index coords.
        - list: Provide a list of variable/coord names to exclude from zeroing.

    Returns
    -------
    xr.Dataset
        A copy of ``ds`` with the selected arrays replaced by zeros.
    """
    import dask.array as da

    if except_for is None:
        except_for = []
    if except_for == "coords":
        except_for = list(ds.coords.keys())

    ds = ds.copy()
    ds_keys = set(ds.data_vars) | set(ds.coords)
    for k in except_for:
        if k not in ds_keys:
            raise ValueError(f"Item {k!r} in `except_for` not a valid item in dataset. Got {except_for=!r}.")

    for k in ds_keys - set(except_for):
        data = da.zeros_like(ds[k].data)
        try:
            ds[k].data = data
        except ValueError:
            # Cannot assign to dimension coordinate, leave as is
            pass

    return ds


def _fill_with_dummy_data(d: dict[str, Any]):
    assert isinstance(d, dict)
    if "dtype" in d:
        d["data"] = np.zeros(d["shape"], dtype=d["dtype"])
        del d["dtype"]
        del d["shape"]

    for k in d:
        if isinstance(d[k], dict):
            d[k] = _fill_with_dummy_data(d[k])

    return d
