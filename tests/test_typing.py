import numpy as np
import pytest
import xarray as xr

from parcels._typing import GridIndexingType, _is_xarray_object, _validate_against_pure_literal


@pytest.mark.parametrize(
    "obj",
    [
        xr.DataArray([1, 2]),
        xr.Dataset({"a": ("x", [1, 2])}),
        xr.Variable("x", [1, 2]),
    ],
)
def test_is_xarray_object(obj):
    assert _is_xarray_object(obj)


@pytest.mark.parametrize("obj", [np.array([1, 2]), "nemo", 1, None])
def test_is_xarray_object_false(obj):
    assert not _is_xarray_object(obj)


def test_validate_against_pure_literal_rejects_xarray():
    with pytest.raises(ValueError, match="Invalid input type"):
        _validate_against_pure_literal(xr.DataArray(["nemo"]), GridIndexingType)
