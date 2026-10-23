from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import xarray as xr

EARTH_RADIUS = 6_366_707.019493707  # m


class BaseMesh(ABC):
    radius: float | None

    @abstractmethod
    def is_spherical(self) -> bool: ...

    def __eq__(self, other):
        return self.is_spherical() == other.is_spherical() and self.radius == other.radius

    def __hash__(self):
        return hash((self.is_spherical(), self.radius))


class SphericalMesh(BaseMesh):
    """Spherical mesh object with configurable planetary radius.

    Pass to FieldSet object as ``mesh=SphericalMesh(radius=...)``.
    radius is in meters; defaults to Earth radius.
    """

    def __init__(self, radius: float = EARTH_RADIUS):
        if not isinstance(radius, (int, float, np.number)):
            raise TypeError(f"radius must be a number, got {type(radius).__name__}")
        if radius <= 0:
            raise ValueError(f"radius must be positive, got {radius}")
        self.radius = radius

    @property
    def deg2m(self) -> float:
        """Meters per degree of arc."""
        assert self.radius is not None
        return self.radius * np.pi / 180.0

    def is_spherical(self):
        return True

    def __repr__(self) -> str:
        return f"SphericalMesh(radius={self.radius})"


class FlatMesh(BaseMesh):
    """Flat mesh object."""

    def __init__(self):
        self.radius = None
        return

    def __repr__(self) -> str:
        return "FlatMesh()"

    def is_spherical(self):
        return False


TMesh = SphericalMesh | FlatMesh  # corresponds with `mesh`


def _is_coordinate_in_degrees(da: xr.DataArray) -> bool:
    units = da.attrs.get("units")
    if units is None:
        raise ValueError(
            f"Coordinate {da.name!r} of your dataset has no 'units' attribute - we don't know what the spatial units are."
        )
    if isinstance(units, str) and "degree" in units.lower():
        return True
    return False


def _infer_mesh_from_coordinates(x: xr.DataArray, y: xr.DataArray) -> TMesh:
    """Infer the mesh from the units of the horizontal coordinates: degrees give a SphericalMesh, anything else a FlatMesh."""
    x_in_degrees = _is_coordinate_in_degrees(x)
    if x_in_degrees ^ _is_coordinate_in_degrees(y):
        msg = (
            f"Mismatch in units between X and Y coordinates.\n"
            f"  Coordinate {x.name!r} attrs: {x.attrs}\n"
            f"  Coordinate {y.name!r} attrs: {y.attrs}\n"
        )
        raise ValueError(msg)
    return SphericalMesh() if x_in_degrees else FlatMesh()


def is_spherical(mesh: FlatMesh | SphericalMesh):
    if isinstance(mesh, SphericalMesh):
        return True
    return False
