"""
Provides helpers and utils for working with SGrid conventions, as well as data objects
useful for representing the SGRID metadata model in code.

This code is best read alongside the SGrid conventions documentation:
https://sgrid.github.io/sgrid/

Note this code doesn't aim to completely cover the SGrid conventions, but aim to
cover SGrid to the extent to which Parcels is concerned.
"""

from __future__ import annotations

import enum
import re
import sys
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass
from textwrap import indent
from typing import IO, Any, Literal, Protocol, Self, cast, overload

import xarray as xr

from parcels._python import repr_from_dunder_dict

_RE_FACE_NODE_PADDING = r"(\w+):(\w+)\s*\(padding:\s*(\w+)\)"

Dim = str


def _indent_lines(lst: list[str], indent: int = 2):
    return [indent * " " + line for line in lst]


class Padding(enum.Enum):
    NONE = "none"
    LOW = "low"
    HIGH = "high"
    BOTH = "both"


def get_n_faces(n_nodes: int, padding: Padding) -> int:
    """Get number of faces along a dimension"""
    if padding in [Padding.LOW, Padding.HIGH]:
        return n_nodes
    if padding == Padding.NONE:
        return n_nodes - 1
    if padding == Padding.BOTH:
        return n_nodes + 1
    raise ValueError(f"Invalid {padding=!r}")


def get_n_nodes(n_faces: int, padding: Padding) -> int:
    """Get number of nodes along a dimension"""
    if padding in [Padding.LOW, Padding.HIGH]:
        return n_faces
    if padding == Padding.NONE:
        return n_faces + 1
    if padding == Padding.BOTH:
        return n_faces - 1
    raise ValueError(f"Invalid {padding=!r}")


class _AttrsSerializable(Protocol):
    def to_attrs(self) -> dict[str, str | int]: ...

    @classmethod
    def from_attrs(cls, d: dict[str, Hashable]) -> Self: ...


class SGrid2DMetadata(_AttrsSerializable):
    def __init__(
        self,
        cf_role: Literal["grid_topology"],
        topology_dimension: Literal[2],
        node_dimensions: tuple[Dim, Dim],
        face_dimensions: tuple[FaceNodePadding, FaceNodePadding],
        node_coordinates: tuple[Dim, Dim] | None = None,
        vertical_dimensions: tuple[FaceNodePadding] | None = None,
    ):
        if cf_role != "grid_topology":
            raise ValueError(f"cf_role must be 'grid_topology', got {cf_role!r}")

        if topology_dimension != 2:
            raise ValueError("topology_dimension must be 2 for a 2D grid")

        if not (
            isinstance(node_dimensions, tuple)
            and len(node_dimensions) == 2
            and all(isinstance(nd, str) for nd in node_dimensions)
        ):
            raise ValueError("node_dimensions must be a tuple of 2 dimensions for a 2D grid")

        if not (
            isinstance(face_dimensions, tuple)
            and len(face_dimensions) == 2
            and all(isinstance(fd, FaceNodePadding) for fd in face_dimensions)
        ):
            raise ValueError("face_dimensions must be a tuple of 2 FaceNodePadding for a 2D grid")

        if node_coordinates is not None:
            if not (
                isinstance(node_coordinates, tuple)
                and len(node_coordinates) == 2
                and all(isinstance(nd, str) for nd in node_coordinates)
            ):
                raise ValueError("node_coordinates must be a tuple of 2 dimensions for a 2D grid")

        if vertical_dimensions is not None:
            if not (
                isinstance(vertical_dimensions, tuple)
                and len(vertical_dimensions) == 1
                and isinstance(vertical_dimensions[0], FaceNodePadding)
            ):
                raise ValueError("vertical_dimensions must be a tuple of 1 FaceNodePadding for a 2D grid")

        # Required attributes
        self.cf_role = cf_role
        self.topology_dimension = topology_dimension
        self.node_dimensions = node_dimensions
        self.face_dimensions = face_dimensions

        # Optional attributes
        self.node_coordinates = node_coordinates
        self.vertical_dimensions = vertical_dimensions

        #! Some optional attributes aren't really important to Parcels, can be added later if needed
        # Optional attributes
        # # With defaults (set in init)
        # edge1_dimensions: tuple[Dim, FaceNodePadding]
        # edge2_dimensions: tuple[FaceNodePadding, Dim]

        # # Without defaults
        # edge1_coordinates: None | Any = None
        # edge2_coordinates: None | Any = None
        # face_coordinate: None | Any = None

    def __repr__(self) -> str:
        return repr_from_dunder_dict(self)

    def describe(self, buf: IO | None = None) -> None:
        """
        ASCII summary of the SGRID 2D metadata, showing the relationship
        between face and node dimensions.

        Parameters
        ----------
        buf : file-like, default: sys.stdout
            writable buffer
        """
        if buf is None:
            buf = sys.stdout
        assert buf is not None

        buf.write(_grid2d_to_ascii(self))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, SGrid2DMetadata):
            return NotImplemented
        return self.to_attrs() == other.to_attrs()

    @classmethod
    def from_attrs(cls, attrs):
        try:
            return cls(
                cf_role=attrs["cf_role"],
                topology_dimension=attrs["topology_dimension"],
                node_dimensions=cast(tuple[Dim, Dim], load_mappings(attrs["node_dimensions"])),
                face_dimensions=cast(tuple[FaceNodePadding, FaceNodePadding], load_mappings(attrs["face_dimensions"])),
                node_coordinates=_maybe_load_mappings(attrs.get("node_coordinates")),
                vertical_dimensions=_maybe_load_mappings(attrs.get("vertical_dimensions")),
            )
        except Exception as e:
            raise SGridParsingException(f"Failed to parse Grid2DMetadata from {attrs=!r}") from e

    def to_attrs(self) -> dict[str, str | int]:
        d: dict[str, str | int] = {
            "cf_role": self.cf_role,
            "topology_dimension": self.topology_dimension,
            "node_dimensions": dump_mappings(self.node_dimensions),
            "face_dimensions": dump_mappings(self.face_dimensions),
        }
        if self.node_coordinates is not None:
            d["node_coordinates"] = dump_mappings(self.node_coordinates)
        if self.vertical_dimensions is not None:
            d["vertical_dimensions"] = dump_mappings(self.vertical_dimensions)
        return d

    def rename(self, names_dict: dict[str, str]) -> Self:
        return cast(Self, _metadata_rename(self, names_dict))

    def get_value_by_id(self, id: str) -> Dim | Padding:
        """In the SGRID specification for 2D grids, different parts of the spec are identified by different "ID"s.

        Easily extract the value for a given ID.

        Example
        -------
        # Get padding 2
        >>> get_name_from_id("type2")
        "low"
        """
        return _ID_FETCHERS_GRID2DMETADATA[id](self)


class SGrid3DMetadata(_AttrsSerializable):
    def __init__(
        self,
        cf_role: Literal["grid_topology"],
        topology_dimension: Literal[3],
        node_dimensions: tuple[Dim, Dim, Dim],
        volume_dimensions: tuple[FaceNodePadding, FaceNodePadding, FaceNodePadding],
        node_coordinates: tuple[Dim, Dim, Dim] | None = None,
    ):
        if cf_role != "grid_topology":
            raise ValueError(f"cf_role must be 'grid_topology', got {cf_role!r}")

        if topology_dimension != 3:
            raise ValueError("topology_dimension must be 3 for a 3D grid")

        if not (
            isinstance(node_dimensions, tuple)
            and len(node_dimensions) == 3
            and all(isinstance(nd, str) for nd in node_dimensions)
        ):
            raise ValueError("node_dimensions must be a tuple of 3 dimensions for a 3D grid")

        if not (
            isinstance(volume_dimensions, tuple)
            and len(volume_dimensions) == 3
            and all(isinstance(fd, FaceNodePadding) for fd in volume_dimensions)
        ):
            raise ValueError("face_dimensions must be a tuple of 2 FaceNodePadding for a 2D grid")

        if node_coordinates is not None:
            if not (
                isinstance(node_coordinates, tuple)
                and len(node_coordinates) == 3
                and all(isinstance(nd, str) for nd in node_coordinates)
            ):
                raise ValueError("node_coordinates must be a tuple of 3 dimensions for a 3D grid")

        # Required attributes
        self.cf_role = cf_role
        self.topology_dimension = topology_dimension
        self.node_dimensions = node_dimensions
        self.volume_dimensions = volume_dimensions

        # Optional attributes
        self.node_coordinates = node_coordinates

        # ! Some optional attributes aren't really important to Parcels, can be added later if needed
        # Optional attributes
        # # With defaults (set in init)
        # edge1_dimensions: tuple[FaceNodePadding, Dim, Dim]
        # edge2_dimensions: tuple[Dim, FaceNodePadding, Dim]
        # edge3_dimensions: tuple[Dim, Dim, FaceNodePadding]
        # face1_dimensions: tuple[Dim, FaceNodePadding, FaceNodePadding]
        # face2_dimensions: tuple[FaceNodePadding, Dim, FaceNodePadding]
        # face3_dimensions: tuple[FaceNodePadding, FaceNodePadding, Dim]

        # # Without defaults
        # edge *i_coordinates*
        # face *i_coordinates*
        # volume_coordinates

    def __repr__(self) -> str:
        return repr_from_dunder_dict(self)

    def describe(self, buf: IO | None = None) -> None:
        """
        ASCII summary of the SGRID 3D metadata, showing the relationship
        between face and node dimensions.

        Parameters
        ----------
        buf : file-like, default: sys.stdout
            writable buffer
        """
        if buf is None:
            buf = sys.stdout
        assert buf is not None

        buf.write(_grid3d_to_ascii(self))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, SGrid3DMetadata):
            return NotImplemented
        return self.to_attrs() == other.to_attrs()

    @classmethod
    def from_attrs(cls, attrs):
        try:
            return cls(
                cf_role=attrs["cf_role"],
                topology_dimension=attrs["topology_dimension"],
                node_dimensions=cast(tuple[Dim, Dim, Dim], load_mappings(attrs["node_dimensions"])),
                volume_dimensions=cast(
                    tuple[FaceNodePadding, FaceNodePadding, FaceNodePadding], load_mappings(attrs["volume_dimensions"])
                ),
                node_coordinates=_maybe_load_mappings(attrs.get("node_coordinates")),
            )
        except Exception as e:
            raise SGridParsingException(f"Failed to parse Grid3DMetadata from {attrs=!r}") from e

    def to_attrs(self) -> dict[str, str | int]:
        d: dict[str, str | int] = {
            "cf_role": self.cf_role,
            "topology_dimension": self.topology_dimension,
            "node_dimensions": dump_mappings(self.node_dimensions),
            "volume_dimensions": dump_mappings(self.volume_dimensions),
        }
        if self.node_coordinates is not None:
            d["node_coordinates"] = dump_mappings(self.node_coordinates)
        return d

    def rename(self, dims_dict: dict[str, str]) -> Self:
        return cast(Self, _metadata_rename(self, dims_dict))

    def get_value_by_id(self, id: str) -> Dim | Padding:
        """In the SGRID specification for 3D grids, different parts of the spec are identified by different "ID"s.

        Easily extract the value for a given ID.

        Example
        -------
        # Get padding 2
        >>> get_name_from_id("type2")
        "low"
        """
        return _ID_FETCHERS_GRID3DMETADATA[id](self)


# Note that - for some optional attributes in the SGRID spec - these IDs are not available
# hence this isn't full coverage
_ID_FETCHERS_GRID2DMETADATA: dict[str, Callable[[SGrid2DMetadata], Dim | Padding]] = {
    "node_dimension1": lambda meta: meta.node_dimensions[0],
    "node_dimension2": lambda meta: meta.node_dimensions[1],
    "face_dimension1": lambda meta: meta.face_dimensions[0].face,
    "face_dimension2": lambda meta: meta.face_dimensions[1].face,
    "type1": lambda meta: meta.face_dimensions[0].padding,
    "type2": lambda meta: meta.face_dimensions[1].padding,
}

_ID_FETCHERS_GRID3DMETADATA: dict[str, Callable[[SGrid3DMetadata], Dim | Padding]] = {
    "node_dimension1": lambda meta: meta.node_dimensions[0],
    "node_dimension2": lambda meta: meta.node_dimensions[1],
    "node_dimension3": lambda meta: meta.node_dimensions[2],
    "face_dimension1": lambda meta: meta.volume_dimensions[0].face,
    "face_dimension2": lambda meta: meta.volume_dimensions[1].face,
    "face_dimension3": lambda meta: meta.volume_dimensions[2].face,
    "type1": lambda meta: meta.volume_dimensions[0].padding,
    "type2": lambda meta: meta.volume_dimensions[1].padding,
    "type3": lambda meta: meta.volume_dimensions[2].padding,
}


@dataclass
class FaceNodePadding:
    """A data class representing a face-node-padding triplet for SGrid metadata.

    In the context of a 2D grid, "face" corresponds with an edge.

    Use the .to_diagram() method to visualize the representation.
    """

    face: Dim
    node: Dim
    padding: Padding

    def __repr__(self) -> str:
        return f"FaceNodePadding(face={self.face!r}, node={self.node!r}, padding={self.padding!r})"

    def __str__(self) -> str:
        return f"{self.face}:{self.node} (padding:{self.padding.value})"

    @classmethod
    def load(cls, s: str) -> Self:
        match = re.match(_RE_FACE_NODE_PADDING, s)
        if not match:
            raise ValueError(f"String {s!r} does not match expected format for FaceNodePadding")
        face = match.group(1)
        node = match.group(2)
        padding = Padding(match.group(3).lower())
        return cls(face, node, padding)

    def to_diagram(self) -> str:
        return "\n".join(_face_node_padding_to_text(self))


def dump_mappings(parts: Iterable[FaceNodePadding | Dim]) -> str:
    """Takes in a list of edge-node-padding tuples and serializes them into a string
    according to the SGrid convention.
    """
    ret = []
    for part in parts:
        ret.append(str(part))
    return " ".join(ret)


@overload
def _maybe_dump_mappings(parts: None) -> None: ...
@overload
def _maybe_dump_mappings(parts: Iterable[FaceNodePadding | Dim]) -> str: ...


def _maybe_dump_mappings(parts):
    if parts is None:
        return None
    return dump_mappings(parts)


def load_mappings(s: str) -> tuple[FaceNodePadding | Dim, ...]:
    """Takes in a string indicating the mappings of dims and dim-dim-padding
    and returns a tuple with this data destructured.

    Treats `:` and `: ` equivalently (in line with the convention).
    """
    if not isinstance(s, str):
        raise ValueError(f"Expected string input, got {s!r} of type {type(s)}")

    s = s.replace(": ", ":")
    ret = []
    while s:
        # find next part
        match = re.match(_RE_FACE_NODE_PADDING, s)
        if match and match.start() == 0:
            # match found at start, take that as next part
            part = match.group(0)
            s_new = s[match.end() :].lstrip()
        else:
            # no FaceNodePadding match at start, assume just a Dim until next space
            part, *rest = s.split(" ", 1)
            s_new = "".join(rest)

        assert s != s_new, f"SGrid parsing did not advance, stuck at {s!r}"

        parsed: FaceNodePadding | Dim
        try:
            parsed = FaceNodePadding.load(part)
        except ValueError as e:
            e.add_note(f"Failed to parse part {part!r} from {s!r} as a dimension dimension padding string")
            try:
                # Not a FaceNodePadding, assume it's just a Dim
                assert ":" not in part, f"Part {part!r} from {s!r} not a valid dim (contains ':')"
                parsed = part
            except AssertionError as e2:
                raise e2 from e

        ret.append(parsed)
        s = s_new

    return tuple(ret)


@overload
def _maybe_load_mappings(s: None) -> None: ...
@overload
def _maybe_load_mappings(s: str) -> tuple[FaceNodePadding | Dim, ...]: ...


def _maybe_load_mappings(s):
    if s is None:
        return None
    return load_mappings(s)


class SGridParsingException(Exception):
    """Exception raised when parsing SGrid attributes fails."""

    pass


def parse_grid_attrs(attrs: dict[str, Hashable]) -> SGrid2DMetadata | SGrid3DMetadata:
    grid: SGrid2DMetadata | SGrid3DMetadata
    try:
        grid = SGrid2DMetadata.from_attrs(attrs)
    except Exception as e:
        e.add_note("Failed to parse as 2D SGrid, trying 3D SGrid")
        try:
            grid = SGrid3DMetadata.from_attrs(attrs)
        except Exception as e2:
            e2.add_note("Failed to parse as 3D SGrid")
            raise SGridParsingException("Failed to parse SGrid metadata as either 2D or 3D grid") from e2
    return grid


def _get_unique_names(grid: SGrid2DMetadata | SGrid3DMetadata) -> set[str]:
    dims = set()
    dims.update(set(grid.node_dimensions))

    for key, value in grid.__dict__.items():
        if key in ("cf_role", "topology_dimension") or value is None:
            continue
        assert isinstance(value, tuple), (
            f"Expected sgrid metadata attribute to be represented as a tuple, got {value!r}. This is an internal error to Parcels - please post an issue if you encounter this."
        )
        for item in value:
            if isinstance(item, FaceNodePadding):
                dims.add(item.face)
                dims.add(item.node)
            else:
                assert isinstance(item, str)
                dims.add(item)
    return dims


def _face_node_padding_to_text(obj: FaceNodePadding) -> list[str]:
    """Return ASCII diagram lines showing a face-node padding relationship.

    Produces a symbolic 5-node diagram like the image below, matching the
    four padding modes::

        face:node (padding:none)
            ●─────●─────●─────●─────●
            1  1  2  2  3  3  4  4  5

        face:node (padding:low)
            ─────●─────●─────●─────●─────●
              1  1  2  2  3  3  4  4  5  5

        face:node (padding:high)
            ●─────●─────●─────●─────●─────
            1  1  2  2  3  3  4  4  5  5

        face:node (padding:both)
            ─────●─────●─────●─────●─────●─────
              1  1  2  2  3  3  4  4  5  5  6
    """
    FACE_WIDTH = 5  # dashes per face segment
    padding = obj.padding

    bars = {
        Padding.NONE: "x-x-x-x-x",
        Padding.LOW: "-x-x-x-x-x",
        Padding.HIGH: "x-x-x-x-x-",
        Padding.BOTH: "-x-x-x-x-x-",
    }
    bar = bars[obj.padding]
    node_count = 0
    face_count = 0
    bar_rendered = ""
    label = ""
    for char in bar:
        if char == "x":
            bar_rendered += "●"
            label += str(node_count)
            node_count += 1
        elif char == "-":
            bar_rendered += "─" * FACE_WIDTH
            label += str(face_count).center(FACE_WIDTH)
            face_count += 1

    return [
        f"{obj.face}:{obj.node} (padding:{padding.value})",
        f"  {bar_rendered}",
        f"  {label.rstrip()}",
    ]


_TEXT_GRID2D_WITHOUT_Z = """
Staggered grid layout (symbolic 3x3 nodes):

  ↑ Y
  |
  n --u-- n --u-- n
  |       |       |
  v   ·   v   ·   v
  |       |       |
  n --u-- n --u-- n
  |       |       |
  v   ·   v   ·   v
  |       |       |
  n --u-- n --u-- n --→ X

  n = node  ({n1}, {n2})
  u = x-face  ({u})
  v = y-face  ({v})
  · = cell centre"""

_TEXT_GRID2D_WITH_Z = """
Staggered grid layout (symbolic 3x3 nodes):

  ↑ Y                     ↑ Z
  |                       |
  n --u-- n --u-- n       w
  |       |       |       |
  v   ·   v   ·   v       ·
  |       |       |       |
  n --u-- n --u-- n       w
  |       |       |       |
  v   ·   v   ·   v       ·
  |       |       |       |
  n --u-- n --u-- n --→ X w

  n = node  ({n1}, {n2})
  u = x-face  ({u})
  v = y-face  ({v})
  w = z-node  ({w})
  · = cell centre"""

_TEXT_GRID3D = """
Staggered grid layout (XY cross-section; Z-faces not shown):

  ↑ Y
  |
  n --u-- n --u-- n
  |       |       |
  v   ·   v   ·   v
  |       |       |
  n --u-- n --u-- n
  |       |       |
  v   ·   v   ·   v
  |       |       |
  n --u-- n --u-- n --→ X

  n = node  ({n1}, {n2}, {n3})
  u = x-face  ({u})
  v = y-face  ({v})
  w = z-face  ({w})  [not shown in cross-section]
  · = cell centre"""


def _grid2d_to_ascii(grid: SGrid2DMetadata) -> str:
    fd = grid.face_dimensions
    nd = grid.node_dimensions
    lines = [
        "Grid2DMetadata",
        f"  X-axis:  face={fd[0].face!r}  node={nd[0]!r}  padding={fd[0].padding.value}",
        f"  Y-axis:  face={fd[1].face!r}  node={nd[1]!r}  padding={fd[1].padding.value}",
    ]
    if grid.vertical_dimensions:
        vd = grid.vertical_dimensions[0]
        lines.append(f"  Z-axis:  face={vd.face!r}  node={vd.node!r}  padding={vd.padding.value}")
    if grid.node_coordinates:
        lines.append(f"  Coordinates: {grid.node_coordinates[0]}, {grid.node_coordinates[1]}")

    format_kwargs = dict(n1=nd[0], n2=nd[1], u=fd[0].face, v=fd[1].face)

    if grid.vertical_dimensions:
        format_kwargs["w"] = grid.vertical_dimensions[0].node
        lines += indent(_TEXT_GRID2D_WITH_Z, "  ").format(**format_kwargs).split("\n")
    else:
        lines += indent(_TEXT_GRID2D_WITHOUT_Z, "  ").format(**format_kwargs).split("\n")

    lines += ["", "  Axis padding:", ""]
    lines += _indent_lines(_face_node_padding_to_text(fd[0]))
    lines += [""]
    lines += _indent_lines(_face_node_padding_to_text(fd[1]))
    if grid.vertical_dimensions:
        lines += [""]
        lines += _indent_lines(_face_node_padding_to_text(grid.vertical_dimensions[0]))
    return "\n".join(lines)


def _grid3d_to_ascii(grid: SGrid3DMetadata) -> str:
    vd = grid.volume_dimensions
    nd = grid.node_dimensions
    lines = [
        "Grid3DMetadata",
        f"  X-axis:  face={vd[0].face!r}  node={nd[0]!r}  padding={vd[0].padding.value}",
        f"  Y-axis:  face={vd[1].face!r}  node={nd[1]!r}  padding={vd[1].padding.value}",
        f"  Z-axis:  face={vd[2].face!r}  node={nd[2]!r}  padding={vd[2].padding.value}",
    ]
    if grid.node_coordinates:
        lines.append(f"  Coordinates: {', '.join(grid.node_coordinates)}")

    lines += (
        indent(_TEXT_GRID3D, "  ")
        .format(n1=nd[0], n2=nd[1], n3=nd[2], u=vd[0].face, v=vd[1].face, w=vd[2].face)
        .split("\n")
    )

    lines += ["", "  Axis padding:", ""]
    lines += _indent_lines(_face_node_padding_to_text(vd[0]))
    lines += [""]
    lines += _indent_lines(_face_node_padding_to_text(vd[1]))
    lines += [""]
    lines += _indent_lines(_face_node_padding_to_text(vd[2]))
    return "\n".join(lines)


def _attach_sgrid_metadata(ds: xr.Dataset, grid: SGrid2DMetadata | SGrid3DMetadata) -> xr.Dataset:
    """Copies the dataset and attaches the SGRID metadata in 'grid' variable. Modifies 'conventions' attribute."""
    ds = ds.copy()
    ds["grid"] = (
        [],
        0,
        grid.to_attrs(),
    )
    ds.attrs["Conventions"] = "SGRID"
    return ds


@overload
def _metadata_rename(grid: SGrid2DMetadata, names_dict: dict[str, str]) -> SGrid2DMetadata: ...


@overload
def _metadata_rename(grid: SGrid3DMetadata, names_dict: dict[str, str]) -> SGrid3DMetadata: ...


def _metadata_rename(grid, names_dict):
    """
    Renames dimensions and coordinates in SGrid metadata.

    Similar in API to xr.Dataset.rename . Renames dimensions according to names_dict mapping
     of old dimension names to new dimension names.
    """
    names_dict = names_dict.copy()
    assert len(names_dict) == len(set(names_dict.values())), "names_dict contains duplicate target dimension names"

    existing_names = _get_unique_names(grid)
    for name in names_dict.keys():
        if name not in existing_names:
            raise ValueError(f"Name {name!r} not found in names defined in SGrid metadata {existing_names!r}")

    for name in existing_names:
        if name not in names_dict:
            names_dict[name] = name  # identity mapping for names not being renamed

    kwargs = {}
    for key, value in grid.__dict__.items():
        if isinstance(value, tuple):
            new_value = []
            for item in value:
                if isinstance(item, FaceNodePadding):
                    new_item = FaceNodePadding(
                        face=names_dict[item.face],
                        node=names_dict[item.node],
                        padding=item.padding,
                    )
                    new_value.append(new_item)
                else:
                    assert isinstance(item, str)
                    new_value.append(names_dict[item])
            kwargs[key] = tuple(new_value)
            continue

        if key in ("cf_role", "topology_dimension") or value is None:
            kwargs[key] = value
            continue

        if isinstance(value, str):
            kwargs[key] = names_dict[value]
            continue

        raise ValueError(f"Unexpected attribute {key!r} on {grid!r}")
    return type(grid)(**kwargs)
