"""
Typing support for Parcels.

This module contains type aliases used throughout Parcels as well as functions that are
used for runtime parameter validation (to ensure users are only using the right params).

"""

import os
from datetime import datetime
from typing import Literal, get_args

import numpy as np
from cftime import datetime as cftime_datetime

from parcels._core.mesh import TMesh  # noqa: F401
from parcels._sgrid.core import Padding

PathLike = str | os.PathLike
VectorType = Literal["3D", "3DSigma", "2D"] | None  # corresponds with `vector_type`
TimeLike = datetime | cftime_datetime | np.datetime64

CfAxisSpatial = Literal["X", "Y", "Z"]
XgridAxis = CfAxisSpatial
XgcmAxisDirection = CfAxisSpatial | Literal["T"]
CfAxis = XgcmAxisDirection
GridPosition = Literal["face"] | Padding
VectorFields = dict[str, tuple[str, str] | tuple[str, str, str]]


def _is_xarray_object(obj):  # with no imports
    try:
        return "xarray.core" in obj.__module__
    except AttributeError:
        return False


def _validate_against_pure_literal(value, typing_literal):
    """Uses a Literal type alias to validate.

    Can't be used with ``Literal[...] | None`` etc. as its not a pure literal.
    """
    # Xarray objects don't work normally in `in` statements - see https://github.com/pydata/xarray/issues/11209 (this is unlikely to be resolved anytime soon)
    if _is_xarray_object(value):
        raise ValueError(f"Invalid input type {type(value)}")

    if value not in get_args(typing_literal):
        msg = f"Invalid value {value!r}. Valid options are {get_args(typing_literal)!r}"
        raise ValueError(msg)
