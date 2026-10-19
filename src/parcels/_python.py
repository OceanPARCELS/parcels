# Generic Python helpers
import enum
from collections.abc import Mapping
from typing import TypeVar

K = TypeVar("K")
V = TypeVar("V")

NotSetType = enum.Enum("NotSetType", "VALUE")
NOTSET = NotSetType.VALUE


def repr_from_dunder_dict(obj: object) -> str:
    """Dataclass-like __repr__ implementation based on __dict__."""
    parts = [f"{k}={v!r}" for k, v in obj.__dict__.items()]
    return f"{obj.__class__.__qualname__}(" + ", ".join(parts) + ")"


def invert_non_unique_mapping(d: Mapping[K, V]) -> Mapping[V, list[K]]:
    inv_map: dict[V, list[K]] = {}
    for k, v in d.items():
        inv_map[v] = inv_map.get(v, []) + [k]
    return inv_map
