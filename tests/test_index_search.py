import numpy as np
import pytest

from parcels._core.fieldset import FieldSet
from parcels._core.index_search import (
    _search_1d_array,
    _search_indices_curvilinear_2d,
    curvilinear_point_in_cell,
    uxgrid_point_in_cell,
)
from parcels._datasets.structured.generic import datasets
from tests.utils import create_lonlat_grid, sample_points_inside_faces


@pytest.fixture
def field_cone():
    ds = datasets["2d_left_unrolled_cone"]
    fieldset = FieldSet.from_sgrid_conventions(ds, mesh="flat")
    return fieldset.data_g


def test_multi_array_errors_in_search_1d_array():
    arr = np.array([[1, 2, 3]])
    x = np.array([[1], [2]])
    with pytest.raises(ValueError):
        _search_1d_array(arr, x)


def test_grid_indexing_fpoints(field_cone):
    grid = field_cone.grid

    for yi_expected in range(grid.ydim - 1):
        for xi_expected in range(grid.xdim - 1):
            x = np.array([grid.lon[yi_expected, xi_expected] + 0.00001])
            y = np.array([grid.lat[yi_expected, xi_expected] + 0.00001])

            yi, eta, xi, xsi = _search_indices_curvilinear_2d(grid, y, x)
            if eta > 0.9:
                yi_expected -= 1
            if xsi > 0.9:
                xi_expected -= 1
            assert yi == yi_expected, f"Expected yi {yi_expected} but got {yi}"
            assert xi == xi_expected, f"Expected xi {xi_expected} but got {xi}"

            cell_lon = [
                grid.lon[yi, xi],
                grid.lon[yi, xi + 1],
                grid.lon[yi + 1, xi + 1],
                grid.lon[yi + 1, xi],
            ]
            cell_lat = [
                grid.lat[yi, xi],
                grid.lat[yi, xi + 1],
                grid.lat[yi + 1, xi + 1],
                grid.lat[yi + 1, xi],
            ]
            assert x > np.min(cell_lon) and x < np.max(cell_lon)
            assert y > np.min(cell_lat) and y < np.max(cell_lat)


def _point_in_cell_weights(nodes_per_face, offset=1e-4):
    """One point at the centre of a face plus one a hair inside each of its edges.

    An edge's two nodes share whatever is left after every other node takes ``offset``,
    split unevenly so the samples avoid edge midpoints. Edge-hugging points are the most
    sensitive to projection accuracy, and are where a particle crossing between faces
    actually sits.
    """
    near_edge = np.full((nodes_per_face, nodes_per_face), offset)
    rest = 1.0 - offset * (nodes_per_face - 2)
    for edge, weights in enumerate(near_edge):
        weights[edge] = rest * 0.61
        weights[(edge + 1) % nodes_per_face] = rest * 0.39

    centre = np.full((1, nodes_per_face), 1.0 / nodes_per_face)
    return np.vstack([centre, near_edge])


@pytest.mark.parametrize("grid_type", ["uxgrid", "xgrid"])
@pytest.mark.parametrize(
    ("face_deg", "cells_per_side"),
    [
        pytest.param(1.0, 8, id="1deg"),
        pytest.param(5.0, 8, id="5deg"),
        pytest.param(15.0, 4, id="15deg"),
        pytest.param(25.0, 4, id="25deg-large-scale"),
    ],
)
def test_point_in_cell_locates_interior_points_of_large_faces(grid_type, face_deg, cells_per_side):
    """A point-in-cell check must accept a point inside the face it is given."""
    grid, nodes, faces = create_lonlat_grid(grid_type, face_deg, centre=(25.0, 10.0), n=cells_per_side)
    weights = _point_in_cell_weights(faces.shape[1])
    lon, lat, expected_face = sample_points_inside_faces(nodes, faces, weights=weights)

    # A UxGrid indexes its faces in one flat list, so unraveling leaves yi at 0 and puts
    # the face id in xi (which is all uxgrid_point_in_cell reads); an XGrid's cells are a
    # 2-D lattice, so both indexes matter.
    cell_shape = (1, len(faces)) if grid_type == "uxgrid" else (cells_per_side, cells_per_side)
    yi, xi = np.unravel_index(expected_face, cell_shape)

    point_in_cell = uxgrid_point_in_cell if grid_type == "uxgrid" else curvilinear_point_in_cell
    is_in_cell, coords = point_in_cell(grid, lat, lon, yi, xi)

    rejected = is_in_cell == 0
    n_rejected = int(np.count_nonzero(rejected))
    assert n_rejected == 0, (
        f"{n_rejected} of {len(lon)} points were rejected from the face containing them; e.g. "
        f"(lon, lat)={np.column_stack((lon, lat))[rejected][:3].tolist()}"
    )

    if grid_type == "uxgrid":
        # Barycentric coordinates are normalized to sum to 1. The curvilinear branch
        # returns (xsi, eta) bilinear weights instead, which carry no such invariant.
        coord_sum = coords.sum(axis=1)
        worst = int(np.argmax(np.abs(coord_sum - 1.0)))
        assert np.allclose(coord_sum, 1.0, rtol=1e-6, atol=1e-6), (
            f"barycentric coordinates of interior points do not sum to 1; worst is "
            f"{coord_sum[worst]:.6f} at (lon, lat)=({lon[worst]:.3f}, {lat[worst]:.3f}) "
            f"in face {expected_face[worst]}"
        )
