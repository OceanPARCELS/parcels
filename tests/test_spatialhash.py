from io import StringIO

import numpy as np
import pytest

from parcels._core.fieldset import FieldSet
from parcels._core.index_search import _latlon_rad_to_xyz
from parcels._core.spatialhash import _HASH_ENTRIES_PER_FACE, _HASH_ENTRY_BUDGET_MIN, _spherical_face_bounds
from parcels._datasets.structured.generic import datasets
from tests.utils import create_lonlat_grid, sample_points_inside_faces


def _cell_centers(grid):
    lon, lat = grid.lon, grid.lat
    clon = 0.25 * (lon[:-1, :-1] + lon[:-1, 1:] + lon[1:, :-1] + lon[1:, 1:])
    clat = 0.25 * (lat[:-1, :-1] + lat[:-1, 1:] + lat[1:, :-1] + lat[1:, 1:])
    jj, ii = np.meshgrid(np.arange(clat.shape[0]), np.arange(clat.shape[1]), indexing="ij")
    return clat, clon, jj, ii


def test_spatialhash_init():
    ds = datasets["2d_left_rotated"]
    grid = FieldSet.from_sgrid_conventions(ds).data_g.grid
    spatialhash = grid.get_spatial_hash()
    assert spatialhash is not None


def test_spatialhash_describe():
    ds = datasets["2d_left_rotated"]
    grid = FieldSet.from_sgrid_conventions(ds).data_g.grid
    spatialhash = grid.get_spatial_hash()

    io = StringIO()
    expected = """\
Spatial Hash Grid Statistics
Grid type                                       : XGrid
Mesh                                            : FlatMesh()
Total mesh faces                                : 1,711
Valid (non-NaN) mesh faces                      : 1,711
Bitwidth (current / max)                        : 1023 / 1023  (higher = finer resolution hash grid)
Total hash cells                                : 1,073,741,824
Occupied hash cells                             : 796,054, 0.0741%
Total (hash cell --> grid face) entries         : 1,080,194
Entries per occupied hash cell (avg)            : 1.36
Entries per face (avg)                          : 631.32
Faces per occupied hash cell (min / mean / max) : 1 / 1.36 / 4
"""
    spatialhash.describe(io)
    actual = io.getvalue()
    assert actual == expected


def test_invalid_positions():
    ds = datasets["2d_left_rotated"]
    grid = FieldSet.from_sgrid_conventions(ds).data_g.grid

    j, i, _ = grid.get_spatial_hash().query([np.nan, np.inf], [np.nan, np.inf])
    assert np.all(j == -3)
    assert np.all(i == -3)


def test_spherical_regional_bounds():
    """Hash-grid bounds for spherical meshes are the Cartesian bounding box of the
    (regional) grid, not the whole unit cube, so quantization resolution is not
    wasted on parts of the sphere the grid does not cover.
    """
    ds = datasets["2d_left_rotated"]
    grid = FieldSet.from_sgrid_conventions(ds).data_g.grid
    spatialhash = grid.get_spatial_hash()

    extents = np.array(
        [
            spatialhash._xmax - spatialhash._xmin,
            spatialhash._ymax - spatialhash._ymin,
            spatialhash._zmax - spatialhash._zmin,
        ]
    )
    assert np.all(extents > 0.0)
    assert np.all(extents < 2.0)  # strictly tighter than the unit cube

    # Queries at cell centers must still resolve to the correct cell
    clat, clon, jj, ii = _cell_centers(grid)
    j, i, _ = spatialhash.query(clat.ravel(), clon.ravel())
    assert np.array_equal(j, jj.ravel())
    assert np.array_equal(i, ii.ravel())

    # Points far outside the regional domain must not match any cell
    j, i, _ = spatialhash.query([-60.0, 80.0], [120.0, -150.0])
    assert np.all(j == -3)
    assert np.all(i == -3)


def test_hash_entry_budget():
    """When the requested bitwidth would blow the hash-entry budget (e.g. tilted
    regional spherical meshes, where face bounding boxes overlap 3-D blocks of
    hash cells), the resolution is reduced to fit; queries still resolve exactly.
    """
    ds = datasets["2d_left_rotated"]
    grid = FieldSet.from_sgrid_conventions(ds).data_g.grid
    spatialhash = grid.get_spatial_hash()

    budget = max(_HASH_ENTRIES_PER_FACE * np.size(spatialhash._xlow), _HASH_ENTRY_BUDGET_MIN)
    assert spatialhash._total_hash_entries(1023) > budget  # this grid requires the cap
    assert spatialhash._bitwidth < 1023
    assert spatialhash._total_hash_entries(spatialhash._bitwidth) <= budget
    assert spatialhash._hash_table["faces"].size <= budget

    clat, clon, jj, ii = _cell_centers(grid)
    j, i, _ = spatialhash.query(clat.ravel(), clon.ravel())
    assert np.array_equal(j, jj.ravel())
    assert np.array_equal(i, ii.ravel())


def test_mixed_positions():
    ds = datasets["2d_left_rotated"]
    grid = FieldSet.from_sgrid_conventions(ds).data_g.grid
    lat = grid.lat.mean()
    lon = grid.lon.mean()
    y = [lat, np.nan]
    x = [lon, np.nan]
    j, i, _ = grid.get_spatial_hash().query(y, x)
    assert j[0] == 29  # Actual value for 2d_left_rotated center
    assert i[0] == 14  # Actual value for 2d_left_rotated center
    assert j[1] == -3
    assert i[1] == -3


def test_nan_node_invalidates_touching_faces():
    """Any mesh face that touches a NaN node should not be added to the HashTable."""
    ds = datasets["2d_left_rotated"]
    grid = FieldSet.from_sgrid_conventions(ds).data_g.grid
    clat, clon, jj, ii = _cell_centers(grid)

    # `grid._ds` shares its lon/lat arrays with the module-level `datasets` fixture
    # (from_sgrid_conventions does not copy them), so deep-copy before mutating,
    # otherwise the NaN injected below leaks into every other test in the session.
    grid._ds = grid._ds.copy(deep=True)

    # Set one interior node to NaN, and calculate the indexes of faces that touch it.
    nj, ni = 10, 10
    touching = [
        (nj - 1, ni - 1),
        (nj - 1, ni),
        (nj, ni - 1),
        (nj, ni),
    ]
    grid._ds["lon"].values[nj, ni] = np.nan
    grid._ds["lat"].values[nj, ni] = np.nan
    spatialhash = grid.get_spatial_hash(reconstruct=True)

    # From the indexes of the faces that touch the NaN node, calculate their
    # face_ids.
    n_faces_x = clon.shape[1]
    invalid_ids = set()
    for j, i in touching:
        invalid_ids.add(j * n_faces_x + i)

    # Get a set of all of the face_ids that are in the table, and assert that the
    # ones touching the NaN node are not among them.
    faces_in_table = set(np.unique(spatialhash._hash_table["faces"]).tolist())
    assert invalid_ids.isdisjoint(faces_in_table)

    # The total number of mesh faces should be greater than the number in the table,
    # since the NaN faces are filtered from the table.
    n_total_faces = jj.size
    assert n_total_faces > len(faces_in_table)

    # Queries landing on those 4 faces should return GridSearchErrors (-3).
    touching_lat = np.array([clat[j, i] for j, i in touching])
    touching_lon = np.array([clon[j, i] for j, i in touching])
    j_touch, i_touch, _ = spatialhash.query(touching_lat, touching_lon)
    assert np.all(j_touch == -3)
    assert np.all(i_touch == -3)

    # All mesh cells not contacting the NaN node should resolve queries.
    mask = np.ones(clat.shape, dtype=bool)
    for j, i in touching:
        mask[j, i] = False
    j_rest, i_rest, _ = spatialhash.query(clat[mask], clon[mask])
    assert np.array_equal(j_rest, jj[mask])
    assert np.array_equal(i_rest, ii[mask])


_SPHERICAL_FACE_CASES = [
    pytest.param(1.0, 8, (0.3, 0.6), id="1deg-small-scale"),
    pytest.param(15.0, 4, (40.0, 30.0), id="15deg-medium-scale"),
    pytest.param(25.0, 4, (25.0, 10.0), id="25deg-large-scale"),
]


@pytest.mark.parametrize("grid_type", ["uxgrid", "xgrid"], ids=["triangles", "quads"])
@pytest.mark.parametrize(("face_deg", "cells_per_side", "centre"), _SPHERICAL_FACE_CASES)
def test_spherical_face_bounds_contains_face_interior(face_deg, cells_per_side, centre, grid_type):
    """_spherical_face_bounds's per-face box must contain the whole face, not just its vertices."""
    _, nodes, faces = create_lonlat_grid(grid_type, face_deg, centre=centre, n=cells_per_side)
    lon, lat, expected_face = sample_points_inside_faces(nodes, faces)

    face_lon = np.deg2rad(nodes[faces, 0])
    face_lat = np.deg2rad(nodes[faces, 1])
    verts = np.stack(_latlon_rad_to_xyz(face_lat, face_lon), axis=-1)  # (n_face, nodes_per_face, 3)

    xlow, xhigh, ylow, yhigh, zlow, zhigh = _spherical_face_bounds(verts)
    low = np.stack([xlow, ylow, zlow], axis=-1)
    high = np.stack([xhigh, yhigh, zhigh], axis=-1)

    query = np.stack(_latlon_rad_to_xyz(np.deg2rad(lat), np.deg2rad(lon)), axis=-1)

    inside_box = np.all((query >= low[expected_face]) & (query <= high[expected_face]), axis=1)
    n_outside = int(np.count_nonzero(~inside_box))
    assert n_outside == 0, (
        f"{n_outside} of {len(lon)} points lie inside their face but outside that face's "
        f"bounding box; e.g. (lon, lat)="
        f"{np.column_stack((lon, lat))[~inside_box][:3].tolist()}"
    )


def test_spherical_face_bounds_ignores_faces_without_area():
    """A face enclosing no area has no interior for an axis point to fall inside, so its
    bounds must not be widened to +-1.
    """
    faces_lonlat = [
        # the last two nodes are the same point
        [(40.0, 30.0), (40.05, 30.0), (40.05, 30.05), (40.05, 30.05)],
        # every node on the equator, so all four sit on one great circle
        [(40.0, 0.0), (40.03, 0.0), (40.08, 0.0), (40.02, 0.0)],
    ]
    verts = np.stack(
        [
            np.stack(
                _latlon_rad_to_xyz(np.deg2rad([node[1] for node in face]), np.deg2rad([node[0] for node in face])),
                axis=-1,
            )
            for face in faces_lonlat
        ]
    )
    xlow, xhigh, ylow, yhigh, zlow, zhigh = _spherical_face_bounds(verts)

    for low, high, axis in ((xlow, xhigh, "x"), (ylow, yhigh, "y"), (zlow, zhigh, "z")):
        assert np.all(low > -1.0), f"a face with no area had its {axis} bound widened to -1"
        assert np.all(high < 1.0), f"a face with no area had its {axis} bound widened to +1"


@pytest.mark.parametrize("mesh", ["flat", "spherical"])
@pytest.mark.parametrize("grid_type", ["uxgrid", "xgrid"])
@pytest.mark.parametrize(("face_deg", "cells_per_side", "centre"), _SPHERICAL_FACE_CASES)
def test_hash_locates_every_interior_point(grid_type, face_deg, cells_per_side, centre, mesh):
    """End-to-end: SpatialHash.query() must return the face a point is known to lie inside."""
    grid, nodes, faces = create_lonlat_grid(grid_type, face_deg, centre=centre, n=cells_per_side, mesh=mesh)
    spatialhash = grid.get_spatial_hash()
    lon, lat, expected_face = sample_points_inside_faces(nodes, faces, mesh=mesh)

    j, i, _ = spatialhash.query(lat, lon)
    # A UxGrid's bounds are one row of faces, so unraveling leaves j at 0 and puts the
    # face id in i; an XGrid's are the (ny-1, nx-1) cell lattice, so both indexes matter.
    expected_j, expected_i = np.unravel_index(expected_face, spatialhash._xlow.shape)

    n_wrong = int(np.count_nonzero((j != expected_j) | (i != expected_i)))
    assert n_wrong == 0, f"{mesh} search failed to locate {n_wrong} of {len(lon)} interior points"
