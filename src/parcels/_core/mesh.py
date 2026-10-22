from abc import ABC, abstractmethod

import numpy as np

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


def get_mesh(mesh: TMesh):  # TODO remove
    if isinstance(mesh, SphericalMesh):
        return mesh
    if mesh == "flat":
        return FlatMesh()
    if mesh == "spherical":
        return SphericalMesh(EARTH_RADIUS)
    raise ValueError(f"mesh must be 'flat', 'spherical', or a SphericalMesh object. Got {mesh=!r}")


def is_spherical(mesh: FlatMesh | SphericalMesh):
    if isinstance(mesh, SphericalMesh):
        return True
    return False
