import numpy as np
import xarray as xr
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays as np_arrays

import parcels._sgrid as sgrid
import parcels._strategies as pst

TCoords1D = np.ndarray[tuple[int], np.dtype[np.float64]]
TCoords2D = np.ndarray[tuple[int, int], np.dtype[np.float64]]
TCoords = TCoords1D | TCoords2D


@st.composite
def sgrid_dataset(draw, grid: sgrid.SGrid2DMetadata | None = None) -> xr.Dataset:
    """Strategy to create Xarray Sgrid datasets for testing"""
    if grid is None:
        grid = draw(pst.sgrid.grid2Dmetadata(use_standard_names=True).filter(lambda g: g.node_coordinates is not None))
    elif grid.node_coordinates is None:
        raise ValueError("grid in Parcels must have node_coordinates set")
    assert grid is not None
    assert grid.node_coordinates is not None

    N = draw(st.integers(min_value=5, max_value=100))
    M = draw(st.integers(min_value=5, max_value=100))

    node_dim1, node_dim2 = grid.node_dimensions
    face_dim1 = grid.face_dimensions[0].face
    face_dim2 = grid.face_dimensions[1].face
    N_face = sgrid.get_n_faces(N, grid.face_dimensions[0].padding)
    M_face = sgrid.get_n_faces(M, grid.face_dimensions[1].padding)

    if has_vertical := grid.vertical_dimensions is not None:
        P = draw(st.integers(min_value=5, max_value=20))
        vert_node_dim = grid.vertical_dimensions[0].node
        vert_face_dim = grid.vertical_dimensions[0].face
        P_face = sgrid.get_n_faces(P, grid.vertical_dimensions[0].padding)

    has_curvilinear_grid = draw(st.booleans())
    coord_name1, coord_name2 = grid.node_coordinates

    c1: TCoords  # pyright: ignore[reportInvalidTypeForm]
    c2: TCoords  # pyright: ignore[reportInvalidTypeForm]
    if has_curvilinear_grid:
        c1, c2 = np.meshgrid(np.linspace(0, 100, N), np.linspace(0, 100, M), indexing="ij")
        coord1_dims = [node_dim1, node_dim2]
        coord2_dims = [node_dim1, node_dim2]
    else:
        c1 = np.linspace(0, 100, N)
        c2 = np.linspace(0, 100, M)
        coord1_dims = [node_dim1]
        coord2_dims = [node_dim2]

    num_fields = draw(st.integers(min_value=1, max_value=4))
    data_vars = {}

    for i in range(num_fields):
        dim1 = draw(st.sampled_from([node_dim1, face_dim1]))
        size1 = N if dim1 == node_dim1 else N_face

        dim2 = draw(st.sampled_from([node_dim2, face_dim2]))
        size2 = M if dim2 == node_dim2 else M_face

        shape: tuple[int, ...]
        if has_vertical and draw(st.booleans()):
            vert_dim = draw(st.sampled_from([vert_node_dim, vert_face_dim]))
            vert_size = P if vert_dim == vert_node_dim else P_face
            dims = [vert_dim, dim1, dim2]
            shape = (vert_size, size1, size2)
        else:
            dims = [dim1, dim2]
            shape = (size1, size2)

        data = draw(
            np_arrays(
                dtype=np.float64,
                shape=shape,
                elements=st.floats(min_value=1e-3, max_value=100.0, allow_nan=False, allow_infinity=False),
            )
        )
        data_vars[f"field_{i}"] = (dims, data)

    coords = {
        coord_name1: (coord1_dims, c1),
        coord_name2: (coord2_dims, c2),
    }

    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    return sgrid._attach_sgrid_metadata(ds, grid)


# Node and face dimension names per spatial axis, following the naming of the datasets
# in parcels._datasets.structured.generated and parcels._datasets.structured.generic
_AXIS_DIMS = {"X": ("XG", "XC"), "Y": ("YG", "YC"), "Z": ("ZG", "ZC")}

_TIME_BASE = xr.date_range("2000-01-01", periods=3, freq="YS")
# Time offsets are multiples of 400 days, so that the time intervals of different models can
# coincide, overlap or be disjoint, but never exactly touch (which would make
# TimeInterval.intersection raise, as it requires left < right)
_TIME_OFFSETS_DAYS = [0, 400, 800, 1200]


def reduced_dims_dataset(
    spatial_axes: frozenset[str],
    time_spec: tuple[int, int],
    field_name: str,
    *,
    value: float,
    sizes: dict[str, int] | None = None,
    paddings: dict[str, sgrid.Padding] | None = None,
) -> xr.Dataset:
    """Create a small SGRID dataset on a grid of the requested dimensionality.

    Parameters
    ----------
    spatial_axes : frozenset of str
        Subset of {"X", "Y", "Z"} present on the grid ("X" and "Y" must come as a pair, as
        Parcels requires both lon and lat coordinates on grids with a horizontal axis). An
        empty set creates a 0D (spatially constant) grid.
    time_spec : tuple of int
        (n_times, offset_days), where n_times is the number of time snapshots (0 means no time
        dimension, 1 a single snapshot) and offset_days the offset of the time axis in days.
    field_name : str
        Name of the single field in the dataset.
    value : float
        Constant value of the field data (so that tests can assert exact interpolated values).
    sizes : dict of str to int, optional
        Number of nodes per spatial axis (defaults to 4).
    paddings : dict of str to sgrid.Padding, optional
        SGRID padding per spatial axis (defaults to Padding.HIGH).

    Returns
    -------
    xarray.Dataset
        Dataset with SGRID metadata and a single constant-valued field.
    """
    spatial_axes = frozenset(spatial_axes)
    if ("X" in spatial_axes) != ("Y" in spatial_axes):
        raise ValueError("Axes 'X' and 'Y' must be present together")
    n_times, offset_days = time_spec
    if n_times == 0 and not spatial_axes:
        raise ValueError("A 0D grid requires a time dimension (a scalar field is not supported)")
    sizes = sizes or {}
    paddings = paddings or {}

    coords: dict[str, tuple] = {}
    for axis in ("Z", "Y", "X"):
        if axis in spatial_axes:
            node, face = _AXIS_DIMS[axis]
            n = sizes.get(axis, 4)
            padding = paddings.get(axis, sgrid.Padding.HIGH)
            coords[node] = ([node], np.arange(n, dtype=float), {"axis": axis, "units": "m"})
            coords[face] = ([face], np.arange(sgrid.get_n_faces(n, padding), dtype=float) + 0.5, {"axis": axis})
    if "Z" in spatial_axes:
        coords["depth"] = (["ZG"], np.arange(sizes.get("Z", 4), dtype=float))
    if {"X", "Y"} <= spatial_axes:
        coords["lon"] = (["XG"], np.arange(sizes.get("X", 4), dtype=float), {"units": "m"})
        coords["lat"] = (["YG"], np.arange(sizes.get("Y", 4), dtype=float), {"units": "m"})
    if n_times > 0:
        coords["time"] = (["time"], _TIME_BASE[:n_times] + np.timedelta64(offset_days, "D"), {"axis": "T"})

    field_dims = []
    field_shape = []
    if n_times > 0:
        field_dims.append("time")
        field_shape.append(n_times)
    for axis in ("Z", "Y", "X"):
        if axis in spatial_axes:
            field_dims.append(_AXIS_DIMS[axis][0])
            field_shape.append(sizes.get(axis, 4))
    data_vars = {field_name: (field_dims, np.full(field_shape, value, dtype=np.float64))}

    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    return ds.pipe(
        sgrid._attach_sgrid_metadata,
        sgrid.SGrid2DMetadata(
            cf_role="grid_topology",
            topology_dimension=2,
            node_dimensions=("XG", "YG"),
            face_dimensions=(
                sgrid.FaceNodePadding("XC", "XG", paddings.get("X", sgrid.Padding.HIGH)),
                sgrid.FaceNodePadding("YC", "YG", paddings.get("Y", sgrid.Padding.HIGH)),
            ),
            node_coordinates=("lon", "lat") if {"X", "Y"} <= spatial_axes else None,
            vertical_dimensions=(
                (sgrid.FaceNodePadding("ZC", "ZG", paddings.get("Z", sgrid.Padding.HIGH)),)
                if "Z" in spatial_axes
                else None
            ),
        ),
    )


@st.composite
def sgrid_dataset_reduced_dims(
    draw,
    field_name: str = "field",
    *,
    spatial_axes: frozenset[str] | None = None,
    time_spec: tuple[int, int] | None = None,
) -> xr.Dataset:
    """Strategy to create SGRID datasets on grids of varying dimensionality (0D-4D).

    Wraps `reduced_dims_dataset`, drawing the grid dimensionality (from a 0D spatially
    constant grid up to a full time-varying Z-Y-X grid), the number of time snapshots, the
    size and SGRID padding per axis, and a constant field value.
    """
    if spatial_axes is None:
        spatial_axes = draw(st.sampled_from([frozenset(), frozenset("Z"), frozenset("XY"), frozenset("XYZ")]))
    if time_spec is None:
        # A 0D (spatially constant) grid requires a time dimension (see reduced_dims_dataset)
        n_times_choices = [1, 2, 3] if not spatial_axes else [0, 1, 2, 3]
        time_spec = (draw(st.sampled_from(n_times_choices)), draw(st.sampled_from(_TIME_OFFSETS_DAYS)))
    sizes = {axis: draw(st.integers(3, 6)) for axis in spatial_axes}
    paddings = {axis: draw(st.sampled_from(list(sgrid.Padding))) for axis in spatial_axes}
    value = draw(st.floats(min_value=1e-3, max_value=100.0, allow_nan=False, allow_infinity=False))
    return reduced_dims_dataset(spatial_axes, time_spec, field_name, value=value, sizes=sizes, paddings=paddings)


@st.composite
def sgrid_datasets_varying_dims(
    draw, *, min_models: int = 2, max_models: int = 5, require_spatial_axis: bool = False
) -> list[xr.Dataset]:
    """Strategy for a list of SGRID datasets (with unique field names) on grids of varying dimensionality.

    Each dataset is drawn with `sgrid_dataset_reduced_dims`, so the list can be combined into
    a single FieldSet of models with different grid dimensionalities. If `require_spatial_axis`
    is True, 0D (spatially constant) grids are excluded.
    """
    n_models = draw(st.integers(min_models, max_models))
    datasets = []
    for i in range(n_models):
        spatial_axes = (
            draw(st.sampled_from([frozenset("Z"), frozenset("XY"), frozenset("XYZ")])) if require_spatial_axis else None
        )
        datasets.append(draw(sgrid_dataset_reduced_dims(f"field_{i}", spatial_axes=spatial_axes)))
    return datasets
