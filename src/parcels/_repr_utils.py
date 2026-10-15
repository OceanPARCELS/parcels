"""Parcels reprs"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import xarray as xr
from dask.base import is_dask_collection

from parcels._chunk_cached_array.core import ChunkCachedArray
from parcels._core._windowed_array import WindowedArray
from parcels._python import isinstance_noimport

if TYPE_CHECKING:
    from parcels import Field, FieldSet, ParticleSet
    from parcels._core.field import VectorField
    from parcels._core.model import ModelData
    from parcels._core.spatialhash import SpatialHash
    from parcels._core.utils.time import TimeInterval


def fieldset_repr(fieldset: FieldSet) -> str:
    """Return a pretty repr for FieldSet"""
    fields = cast(
        "list[Field]", [f for f in fieldset.fields.values() if getattr(f.__class__, "__name__", "") == "Field"]
    )
    vfields = cast(
        "list[VectorField]",
        [f for f in fieldset.fields.values() if getattr(f.__class__, "__name__", "") == "VectorField"],
    )

    fields_repr = "\n".join([repr(f) for f in fields])
    vfields_repr = "\n".join([vectorfield_repr(vf, from_fieldset_repr=True) for vf in vfields])

    out = f"""<{type(fieldset).__name__}>
    fields:
{textwrap.indent(fields_repr, 8 * " ")}
    vectorfields:
{textwrap.indent(vfields_repr, 8 * " ")}
"""
    return textwrap.dedent(out).strip()


def field_repr(field: Field, level: int = 0) -> str:
    """Return a pretty repr for Field"""
    with xr.set_options(display_expand_data=False):
        out = f"""<{type(field).__name__} {field.name!r}>
    Parcels attributes:
        name            : {field.name!r}
        interp_method   : {field.interp_method!r}
        time_interval   : {field.time_interval!r}
        igrid           : {field.igrid!r}
    DataArray:
{textwrap.indent(repr(field.data), 8 * " ")}
{textwrap.indent(repr(field.grid), 4 * " ")}
"""
    return textwrap.indent(out, " " * level * 4).strip()


def vectorfield_repr(vector_field: VectorField, from_fieldset_repr=False) -> str:
    """Return a pretty repr for VectorField"""
    out = f"""<{type(vector_field).__name__} {vector_field.name!r}>
    Parcels attributes:
        name                  : {vector_field.name!r}
        interp_method         : {vector_field.interp_method!r}
        vector_type           : {vector_field.vector_type!r}
    {field_repr(vector_field.U, level=1) if not from_fieldset_repr else ""}
    {field_repr(vector_field.V, level=1) if not from_fieldset_repr else ""}
    {field_repr(vector_field.W, level=1) if not from_fieldset_repr and vector_field.W else ""}"""
    return out


def xgrid_repr(grid: Any) -> str:
    """Return a pretty repr for Grid"""
    out = f"""<{type(grid).__name__}>
    Parcels attributes:
        mesh                  : {grid._mesh}
        spatialhash           : {grid._spatialhash}
    SGRID Metadata:
{textwrap.indent(repr(grid.sgrid_metadata), 8 * " ")}
"""
    return textwrap.dedent(out).strip()


def particleset_repr(pset: ParticleSet) -> str:
    """Return a pretty repr for ParticleSet"""
    if len(pset) < 10:
        particles = [repr(p) for p in pset]
    else:
        particles = [repr(pset[i]) for i in range(7)] + ["..."] + [repr(pset[-1])]

    out = f"""<{type(pset).__name__}>
    Number of particles: {len(pset)}
    Particles:
{_format_list_items_multiline(particles, level=2, with_brackets=False)}
    Pclass:
{textwrap.indent(repr(pset._pclass), 8 * " ")}
"""
    return textwrap.dedent(out).strip()


def particlesetview_repr(pview: Any) -> str:
    """Return a pretty repr for ParticleSetView"""
    time_string = "not_yet_set" if pview.time is None or np.isnan(pview.time) else f"{pview.time:f}"
    out = f"P[{pview.particle_id}]: time={time_string}, z={pview.z:f}, y={pview.y:f}, x={pview.x:f}"
    vars = [v.name for v in pview._pclass.variables if v.to_write is True and v.name not in ["z", "y", "x", "time"]]
    for var in vars:
        out += f", {var}={getattr(pview, var):f}"

    return textwrap.dedent(out).strip()


def particleclass_repr(pclass: Any) -> str:
    """Return a pretty repr for ParticleClass"""
    vars = [repr(v) for v in pclass.variables]
    out = f"""
{_format_list_items_multiline(vars, level=1, with_brackets=False)}
"""
    return textwrap.dedent(out).strip()


def variable_repr(var: Any) -> str:
    """Return a pretty repr for Variable"""
    return f"Variable(name={var._name!r}, dtype={var.dtype!r}, initial={var.initial!r}, to_write={var.to_write!r}, attrs={var.attrs!r})"


def timeinterval_repr(ti: Any) -> str:
    """Return a pretty repr for TimeInterval"""
    return f"TimeInterval(left={ti.left!r}, right={ti.right!r})"


def particlefile_repr(pfile: Any) -> str:
    """Return a pretty repr for ParticleFile"""
    out = f"""<{type(pfile).__name__}>
    path                : {pfile.path}
    outputdt            : {pfile.outputdt!r}
    metadata            : {_format_list_items_multiline(pfile.metadata, level=2, with_brackets=False)}
"""
    return textwrap.dedent(out).strip()


def default_repr(obj: Any):
    if is_builtin_object(obj):
        return repr(obj)
    return object.__repr__(obj)


def _format_list_items_multiline(items: list[str] | dict, level: int = 1, with_brackets: bool = True) -> str:
    """Given a list of strings or a dict, formats them across multiple lines.

    Uses indentation levels of 4 spaces provided by ``level``.

    Example
    -------
    >>> output = _format_list_items_multiline(["item1", "item2", "item3"], 4)
    >>> f"my_items: {output}"
    my_items: [
        item1,
        item2,
        item3,
    ]
    """
    if len(items) == 0:
        return "[]"

    assert level >= 1, "Indentation level >=1 supported"
    indentation_str = level * 4 * " "
    indentation_str_end = (level - 1) * 4 * " "

    if isinstance(items, dict):
        entries = [f"{k!s}: {v!s}" for k, v in items.items()]
    else:
        entries = [i if isinstance(i, str) else repr(i) for i in items]

    if with_brackets:
        items_str = ",\n".join([textwrap.indent(e, indentation_str) for e in entries])
        return f"[\n{items_str}\n{indentation_str_end}]"
    else:
        return "\n".join([textwrap.indent(e, indentation_str) for e in entries])


def is_builtin_object(obj):
    return obj.__class__.__module__ == "builtins"


@dataclass
class _FieldSetDescriptionRow:
    type_: Literal["Field", "VectorField", "Context"]
    model_id: int | None
    name: str
    interp_method_or_value: str
    backend: str | None = None

    def to_dict(self) -> dict[str, str]:
        return {
            "Name": self.name,
            "Type": self.type_,
            "Grid number": str(self.model_id) if self.model_id is not None else "-",
            "Interp method / value": self.interp_method_or_value,
            "Parcels backend": self.backend if self.backend is not None else "-",
        }


def _print_table(rows: list[_FieldSetDescriptionRow]) -> str:
    import pandas as pd

    dicts = [r.to_dict() for r in rows]
    return pd.DataFrame(dicts).sort_values(["Grid number", "Type", "Name"]).to_markdown(index=False)


def _print_time_interval(time_interval: TimeInterval | None) -> str:
    if time_interval is None:
        return repr(time_interval)
    return repr((time_interval.left, time_interval.right))


def _field_backend(field: Field | VectorField) -> str | None:
    if hasattr(field, "data"):
        if isinstance(field.data, WindowedArray):
            return "WindowedArray"
        elif is_dask_collection(field.data.variable._data):
            return "Dask"
        elif isinstance(field.data.variable._data, ChunkCachedArray):
            return "ChunkCachedArray"
        elif isinstance(field.data.data, np.ndarray):
            return "NumPy"
        else:
            return type(field.data).__name__
    else:
        return None


def fieldset_describe(fieldset: FieldSet) -> str:
    rows: list[_FieldSetDescriptionRow] = []
    models: dict[int, int] = {}  # mapping of memory ID to a human readable ID

    assert fieldset._fields is not None

    for field in fieldset._fields.values():
        model_id: int

        # Set human readable model ID
        parent_id = id(_get_parent_model(field))
        models[parent_id] = models.get(parent_id, len(models))
        model_id = models[parent_id]

        type_ = cast(Literal["Field", "VectorField", "Context"], field.__class__.__name__)

        rows.append(
            _FieldSetDescriptionRow(
                type_=type_,
                model_id=model_id,
                name=field.name,
                interp_method_or_value=repr(field.interp_method),
                backend=_field_backend(field),
            )
        )
    for k, v in fieldset.context.items():
        rows.append(
            _FieldSetDescriptionRow(
                type_="Context",
                model_id=None,
                name=k,
                interp_method_or_value=repr(v),
                backend=None,
            )
        )
    return (
        _print_table(rows)
        + f"""\


mesh: {fieldset.models[0].grid._mesh}
time interval: {_print_time_interval(fieldset.time_interval)}
"""
    )


def spatialhash_describe(spatialhash: SpatialHash) -> str:
    grid = spatialhash._source_grid
    hash_table = spatialhash._hash_table
    counts = hash_table["counts"]

    n_faces = int(np.size(spatialhash._xlow))
    n_valid_faces = int(np.unique(hash_table["faces"]).size)
    n_entries = int(hash_table["faces"].size)
    n_occupied_cells = int(hash_table["keys"].size)
    n_total_cells = (spatialhash._bitwidth + 1) ** 3

    rows = {
        "Grid type": type(grid).__name__,
        "Mesh": grid._mesh,
        "Total mesh faces": f"{n_faces:,d}",
        "Valid (non-NaN) mesh faces": f"{n_valid_faces:,d}",
        "Bitwidth (current / max)": f"{spatialhash._bitwidth} / 1023  (higher = finer resolution hash grid)",
        "Total hash cells": f"{n_total_cells:,d}",
        "Occupied hash cells": f"{n_occupied_cells:,d}, {n_occupied_cells / n_total_cells * 100:.4f}%",
        "Total (hash cell --> grid face) entries": f"{n_entries:,d}",
        "Entries per occupied hash cell (avg)": f"{n_entries / n_occupied_cells:.2f}" if n_occupied_cells else "-",
        "Entries per face (avg)": f"{n_entries / n_faces:.2f}" if n_faces else "-",
        "Faces per occupied hash cell (min / mean / max)": (
            f"{counts.min():,d} / {counts.mean():.2f} / {counts.max():,d}" if n_occupied_cells else "-"
        ),
    }
    key_width = max(len(k) for k in rows)
    table = "\n".join(f"{k.ljust(key_width)} : {v}" for k, v in rows.items())

    return "Spatial Hash Grid Statistics" + "\n" + table + "\n"


def _get_parent_model(field: Field | VectorField) -> ModelData:
    if isinstance_noimport(field, "Field"):
        return field.model  # type:ignore[union-attr]
    return field.U.model  # type:ignore[union-attr]
