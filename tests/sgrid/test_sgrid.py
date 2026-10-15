import io
import itertools

import hypothesis.strategies as st
import numpy as np
import pytest
import xarray as xr
from hypothesis import assume, example, given

import parcels._sgrid as sgrid
import parcels._strategies as pst
from parcels._sgrid.accessor import get_dim_position
from parcels._sgrid.core import _get_unique_names, parse_grid_attrs


def create_example_grid2dmetadata(with_vertical_dimensions: bool, with_node_coordinates: bool):
    vertical_dimensions = (
        (sgrid.FaceNodePadding("vertical_dimensions_dim1", "vertical_dimensions_dim2", sgrid.Padding.LOW),)
        if with_vertical_dimensions
        else None
    )
    node_coordinates = ("node_coordinates_var1", "node_coordinates_var2") if with_node_coordinates else None

    return sgrid.SGrid2DMetadata(
        cf_role="grid_topology",
        topology_dimension=2,
        node_dimensions=("node_dimension1", "node_dimension2"),
        face_dimensions=(
            sgrid.FaceNodePadding("face_dimension1", "node_dimension1", sgrid.Padding.LOW),
            sgrid.FaceNodePadding("face_dimension2", "node_dimension2", sgrid.Padding.LOW),
        ),
        node_coordinates=node_coordinates,
        vertical_dimensions=vertical_dimensions,
    )


def create_example_grid3dmetadata(with_node_coordinates: bool):
    node_coordinates = (
        ("node_coordinates_var1", "node_coordinates_var2", "node_coordinates_dim3") if with_node_coordinates else None
    )
    return sgrid.SGrid3DMetadata(
        cf_role="grid_topology",
        topology_dimension=3,
        node_dimensions=("node_dimension1", "node_dimension2", "node_dimension3"),
        volume_dimensions=(
            sgrid.FaceNodePadding("face_dimension1", "node_dimension1", sgrid.Padding.LOW),
            sgrid.FaceNodePadding("face_dimension2", "node_dimension2", sgrid.Padding.LOW),
            sgrid.FaceNodePadding("face_dimension3", "node_dimension3", sgrid.Padding.LOW),
        ),
        node_coordinates=node_coordinates,
    )


grid2dmetadata = create_example_grid2dmetadata(with_vertical_dimensions=True, with_node_coordinates=True)
grid3dmetadata = create_example_grid3dmetadata(with_node_coordinates=True)


@pytest.mark.parametrize(
    ("sgrid_metadata", "id", "value"),
    [
        (grid2dmetadata, "node_dimension1", "node_dimension1"),
        (grid2dmetadata, "node_dimension2", "node_dimension2"),
        (grid2dmetadata, "face_dimension1", "face_dimension1"),
        (grid2dmetadata, "face_dimension2", "face_dimension2"),
        # (grid2dmetadata, "vertical_dimensions_dim1", "vertical_dimensions_dim1"), #! ID's NOT IMPLEMENTED IN SGRID SPEC
        # (grid2dmetadata, "vertical_dimensions_dim2", "vertical_dimensions_dim2"),
        (grid3dmetadata, "node_dimension1", "node_dimension1"),
        (grid3dmetadata, "node_dimension2", "node_dimension2"),
        (grid3dmetadata, "node_dimension3", "node_dimension3"),
        (grid3dmetadata, "face_dimension1", "face_dimension1"),
        (grid3dmetadata, "face_dimension2", "face_dimension2"),
        (grid3dmetadata, "face_dimension3", "face_dimension3"),
        (grid3dmetadata, "type1", sgrid.Padding.LOW),
        (grid3dmetadata, "type2", sgrid.Padding.LOW),
        (grid3dmetadata, "type3", sgrid.Padding.LOW),
    ],
)
def test_get_value_by_id(sgrid_metadata: sgrid.SGrid2DMetadata | sgrid.SGrid3DMetadata, id, value):
    assert sgrid_metadata.get_value_by_id(id) == value


def test_get_dim_position_face_dims():
    """Face dimensions return 'face'."""
    metadata = create_example_grid2dmetadata(with_vertical_dimensions=False, with_node_coordinates=False)
    # face_dimensions = (FaceNodePadding("face_dimension1", "node_dimension1", Padding.LOW), ...)
    assert get_dim_position(metadata, "face_dimension1") == "face"
    assert get_dim_position(metadata, "face_dimension2") == "face"


def test_get_dim_position_node_dims():
    """Node dimensions return their Padding value."""
    metadata = create_example_grid2dmetadata(with_vertical_dimensions=False, with_node_coordinates=False)
    assert get_dim_position(metadata, "node_dimension1") == sgrid.Padding.LOW
    assert get_dim_position(metadata, "node_dimension2") == sgrid.Padding.LOW


def test_get_dim_position_vertical():
    """Vertical face and node dimensions are handled."""
    metadata = create_example_grid2dmetadata(with_vertical_dimensions=True, with_node_coordinates=False)
    # vertical_dimensions = (FaceNodePadding("vertical_dimensions_dim1", "vertical_dimensions_dim2", Padding.LOW),)
    assert get_dim_position(metadata, "vertical_dimensions_dim1") == "face"
    assert get_dim_position(metadata, "vertical_dimensions_dim2") == sgrid.Padding.LOW


def test_get_dim_position_unknown_dim_raises():
    """Unknown dimensions raise ValueError."""
    metadata = create_example_grid2dmetadata(with_vertical_dimensions=False, with_node_coordinates=False)
    with pytest.raises(ValueError, match="not a spatial SGRID dimension"):
        get_dim_position(metadata, "nonexistent_dim")


def dummy_sgrid_ds(grid: sgrid.SGrid2DMetadata | sgrid.SGrid3DMetadata) -> xr.Dataset:
    if isinstance(grid, sgrid.SGrid2DMetadata):
        return dummy_sgrid_2d_ds(grid)
    elif isinstance(grid, sgrid.SGrid3DMetadata):
        return dummy_sgrid_3d_ds(grid)
    else:
        raise NotImplementedError(f"Cannot create dummy SGrid dataset for grid type {type(grid)}")


def dummy_sgrid_2d_ds(grid: sgrid.SGrid2DMetadata) -> xr.Dataset:
    ds = dummy_comodo_3d_ds()

    # Can't rename dimensions that already exist in the dataset
    assume(_get_unique_names(grid) & set(ds.dims) == set())

    renamings = {}
    if grid.vertical_dimensions is None:
        ds = ds.isel(ZC=0, ZG=0)
    else:
        renamings.update({"ZC": grid.vertical_dimensions[0].face, "ZG": grid.vertical_dimensions[0].node})

    for old, new in zip(["XG", "YG"], grid.node_dimensions, strict=True):
        renamings[old] = new

    for old, face_node_padding in zip(["XC", "YC"], grid.face_dimensions, strict=True):
        renamings[old] = face_node_padding.face

    ds = ds.rename_dims(renamings)

    ds["grid"] = xr.DataArray(1, attrs=grid.to_attrs())
    ds.attrs["convention"] = "SGRID"
    return ds


def dummy_sgrid_3d_ds(grid: sgrid.SGrid3DMetadata) -> xr.Dataset:
    ds = dummy_comodo_3d_ds()

    # Can't rename dimensions that already exist in the dataset
    assume(_get_unique_names(grid) & set(ds.dims) == set())

    renamings = {}
    for old, new in zip(["XG", "YG", "ZG"], grid.node_dimensions, strict=True):
        renamings[old] = new

    for old, face_node_padding in zip(["XC", "YC", "ZC"], grid.volume_dimensions, strict=True):
        renamings[old] = face_node_padding.face

    ds = ds.rename_dims(renamings)

    ds["grid"] = xr.DataArray(1, attrs=grid.to_attrs())
    ds.attrs["convention"] = "SGRID"
    return ds


def dummy_comodo_3d_ds() -> xr.Dataset:
    T, Z, Y, X = 7, 6, 5, 4
    TIME = xr.date_range("2000", "2001", T)
    return xr.Dataset(
        {
            "data_g": (["time", "ZG", "YG", "XG"], np.random.rand(T, Z, Y, X)),
            "data_c": (["time", "ZC", "YC", "XC"], np.random.rand(T, Z, Y, X)),
            "U_A_grid": (["time", "ZG", "YG", "XG"], np.random.rand(T, Z, Y, X)),
            "V_A_grid": (["time", "ZG", "YG", "XG"], np.random.rand(T, Z, Y, X)),
            "U_C_grid": (["time", "ZG", "YC", "XG"], np.random.rand(T, Z, Y, X)),
            "V_C_grid": (["time", "ZG", "YG", "XC"], np.random.rand(T, Z, Y, X)),
        },
        coords={
            "time": (["time"], TIME, {"axis": "T"}),
        },
    )


@example(
    edge_node_padding=(
        sgrid.FaceNodePadding("edge1", "node1", sgrid.Padding.NONE),
        sgrid.FaceNodePadding("edge2", "node2", sgrid.Padding.LOW),
    )
)
@given(pst.sgrid.mappings)
def test_edge_node_mapping_metadata_roundtrip(edge_node_padding):
    serialized = sgrid.dump_mappings(edge_node_padding)
    parsed = sgrid.load_mappings(serialized)
    assert parsed == edge_node_padding


@pytest.mark.parametrize(
    "input_, expected",
    [
        (
            "edge1: node1(padding: none)",
            (sgrid.FaceNodePadding("edge1", "node1", sgrid.Padding.NONE),),
        ),
    ],
)
def test_load_dump_mappings(input_, expected):
    assert sgrid.load_mappings(input_) == expected


@example(grid2dmetadata)
@given(pst.sgrid.grid2Dmetadata())
def test_Grid2DMetadata_roundtrip(grid: sgrid.SGrid2DMetadata):
    attrs = grid.to_attrs()
    parsed = sgrid.SGrid2DMetadata.from_attrs(attrs)
    assert parsed == grid


@example(grid3dmetadata)
@given(pst.sgrid.grid3Dmetadata())
def test_Grid3DMetadata_roundtrip(grid: sgrid.SGrid3DMetadata):
    attrs = grid.to_attrs()
    parsed = sgrid.SGrid3DMetadata.from_attrs(attrs)
    assert parsed == grid


@given(pst.sgrid.grid2Dmetadata(use_standard_names=True))
def test_grid2Dmetadata_standard_names(grid: sgrid.SGrid2DMetadata):
    assert grid.node_dimensions == ("node_dimension1", "node_dimension2")
    assert grid.face_dimensions[0].face == "face_dimension1"
    assert grid.face_dimensions[1].face == "face_dimension2"
    if grid.node_coordinates is not None:
        assert grid.node_coordinates == ("node_coordinates_var1", "node_coordinates_var2")
    if grid.vertical_dimensions is not None:
        assert grid.vertical_dimensions[0].face == "vertical_dimensions_face"
        assert grid.vertical_dimensions[0].node == "vertical_dimensions_node"


@given(pst.sgrid.grid3Dmetadata(use_standard_names=True))
def test_grid3Dmetadata_standard_names(grid: sgrid.SGrid3DMetadata):
    assert grid.node_dimensions == ("node_dimension1", "node_dimension2", "node_dimension3")
    assert grid.volume_dimensions[0].face == "face_dimension1"
    assert grid.volume_dimensions[1].face == "face_dimension2"
    assert grid.volume_dimensions[2].face == "face_dimension3"
    if grid.node_coordinates is not None:
        assert grid.node_coordinates == ("node_coordinates_var1", "node_coordinates_var2", "node_coordinates_dim3")


@given(pst.sgrid.grid_metadata)
def test_parse_grid_attrs(grid):
    attrs = grid.to_attrs()
    parsed = parse_grid_attrs(attrs)
    assert parsed == grid


@pytest.mark.parametrize(
    "grid",
    [
        create_example_grid2dmetadata(with_node_coordinates=i, with_vertical_dimensions=j)
        for i, j in itertools.product([False, True], [False, True])
    ]
    + [create_example_grid3dmetadata(with_node_coordinates=i) for i in [False, True]],
)
def test_rename(grid):
    dims = _get_unique_names(grid)
    dims_dict = {dim: f"new_{dim}" for dim in dims}
    dims_dict_inv = {v: k for k, v in dims_dict.items()}

    grid_new = grid.rename(dims_dict)
    assert dims & set(_get_unique_names(grid_new)) == set()

    assert grid == grid_new.rename(dims_dict_inv)


def test_rename_errors():
    # Test various error modes of rename_dims
    grid = grid2dmetadata
    # Non-unique target dimension names
    names_dict = {
        "node_dimension1": "new_node_dimension",
        "node_dimension2": "new_node_dimension",
    }
    with pytest.raises(AssertionError, match="names_dict contains duplicate target dimension names"):
        grid.rename(names_dict)
    # Unexpected attribute in dims_dict
    names_dict = {
        "unexpected_dimension": "new_unexpected_dimension",
    }
    with pytest.raises(ValueError, match="Name 'unexpected_dimension' not found in names defined in SGrid metadata"):
        grid.rename(names_dict)


@pytest.mark.parametrize(
    ("metadata, expected"),
    [
        (
            create_example_grid2dmetadata(with_vertical_dimensions=False, with_node_coordinates=False),
            """Grid2DMetadata
  X-axis:  face='face_dimension1'  node='node_dimension1'  padding=low
  Y-axis:  face='face_dimension2'  node='node_dimension2'  padding=low

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

    n = node  (node_dimension1, node_dimension2)
    u = x-face  (face_dimension1)
    v = y-face  (face_dimension2)
    · = cell centre

  Axis padding:

  face_dimension1:node_dimension1 (padding:low)
    ─────●─────●─────●─────●─────●
      0  0  1  1  2  2  3  3  4  4

  face_dimension2:node_dimension2 (padding:low)
    ─────●─────●─────●─────●─────●
      0  0  1  1  2  2  3  3  4  4""",
        ),
        (
            create_example_grid2dmetadata(with_vertical_dimensions=True, with_node_coordinates=True),
            """Grid2DMetadata
  X-axis:  face='face_dimension1'  node='node_dimension1'  padding=low
  Y-axis:  face='face_dimension2'  node='node_dimension2'  padding=low
  Z-axis:  face='vertical_dimensions_dim1'  node='vertical_dimensions_dim2'  padding=low
  Coordinates: node_coordinates_var1, node_coordinates_var2

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

    n = node  (node_dimension1, node_dimension2)
    u = x-face  (face_dimension1)
    v = y-face  (face_dimension2)
    w = z-node  (vertical_dimensions_dim2)
    · = cell centre

  Axis padding:

  face_dimension1:node_dimension1 (padding:low)
    ─────●─────●─────●─────●─────●
      0  0  1  1  2  2  3  3  4  4

  face_dimension2:node_dimension2 (padding:low)
    ─────●─────●─────●─────●─────●
      0  0  1  1  2  2  3  3  4  4

  vertical_dimensions_dim1:vertical_dimensions_dim2 (padding:low)
    ─────●─────●─────●─────●─────●
      0  0  1  1  2  2  3  3  4  4""",
        ),
        (
            create_example_grid3dmetadata(with_node_coordinates=False),
            """Grid3DMetadata
  X-axis:  face='face_dimension1'  node='node_dimension1'  padding=low
  Y-axis:  face='face_dimension2'  node='node_dimension2'  padding=low
  Z-axis:  face='face_dimension3'  node='node_dimension3'  padding=low

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

    n = node  (node_dimension1, node_dimension2, node_dimension3)
    u = x-face  (face_dimension1)
    v = y-face  (face_dimension2)
    w = z-face  (face_dimension3)  [not shown in cross-section]
    · = cell centre

  Axis padding:

  face_dimension1:node_dimension1 (padding:low)
    ─────●─────●─────●─────●─────●
      0  0  1  1  2  2  3  3  4  4

  face_dimension2:node_dimension2 (padding:low)
    ─────●─────●─────●─────●─────●
      0  0  1  1  2  2  3  3  4  4

  face_dimension3:node_dimension3 (padding:low)
    ─────●─────●─────●─────●─────●
      0  0  1  1  2  2  3  3  4  4""",
        ),
        (
            create_example_grid3dmetadata(with_node_coordinates=True),
            """Grid3DMetadata
  X-axis:  face='face_dimension1'  node='node_dimension1'  padding=low
  Y-axis:  face='face_dimension2'  node='node_dimension2'  padding=low
  Z-axis:  face='face_dimension3'  node='node_dimension3'  padding=low
  Coordinates: node_coordinates_var1, node_coordinates_var2, node_coordinates_dim3

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

    n = node  (node_dimension1, node_dimension2, node_dimension3)
    u = x-face  (face_dimension1)
    v = y-face  (face_dimension2)
    w = z-face  (face_dimension3)  [not shown in cross-section]
    · = cell centre

  Axis padding:

  face_dimension1:node_dimension1 (padding:low)
    ─────●─────●─────●─────●─────●
      0  0  1  1  2  2  3  3  4  4

  face_dimension2:node_dimension2 (padding:low)
    ─────●─────●─────●─────●─────●
      0  0  1  1  2  2  3  3  4  4

  face_dimension3:node_dimension3 (padding:low)
    ─────●─────●─────●─────●─────●
      0  0  1  1  2  2  3  3  4  4""",
        ),
    ],
)
def test_grid_describe(metadata, expected):
    buf = io.StringIO()
    metadata.describe(buf)
    assert buf.getvalue() == expected


@pytest.mark.parametrize(
    ("face_node_padding", "expected_lines"),
    [
        (
            sgrid.FaceNodePadding("face", "node", sgrid.Padding.LOW),
            [
                "face:node (padding:low)",
                "  ─────●─────●─────●─────●─────●",
                "    0  0  1  1  2  2  3  3  4  4",
            ],
        ),
        (
            sgrid.FaceNodePadding("face", "node", sgrid.Padding.HIGH),
            [
                "face:node (padding:high)",
                "  ●─────●─────●─────●─────●─────",
                "  0  0  1  1  2  2  3  3  4  4",
            ],
        ),
        (
            sgrid.FaceNodePadding("face", "node", sgrid.Padding.BOTH),
            [
                "face:node (padding:both)",
                "  ─────●─────●─────●─────●─────●─────",
                "    0  0  1  1  2  2  3  3  4  4  5",
            ],
        ),
        (
            sgrid.FaceNodePadding("face", "node", sgrid.Padding.NONE),
            [
                "face:node (padding:none)",
                "  ●─────●─────●─────●─────●",
                "  0  0  1  1  2  2  3  3  4",
            ],
        ),
    ],
)
def test_face_node_padding_to_diagram(face_node_padding: sgrid.FaceNodePadding, expected_lines: list[str]):
    actual = face_node_padding.to_diagram()
    lines = actual.split("\n")
    assert lines == expected_lines


@given(n=st.integers(min_value=1), padding=pst.sgrid.padding)
def test_dim_sizes_roundtrip(n, padding):
    n_faces = sgrid.get_n_nodes(n, padding)
    n_nodes = sgrid.get_n_faces(n_faces, padding)
    assert n_nodes == n
