"""Provides Hypothesis strategies for generating ChunkCachedArray objects and indexers for them."""

from __future__ import annotations

import itertools

import dask.array as da
import numpy as np
from hypothesis import strategies as st
from hypothesis.extra import numpy as npst
from xarray.core.indexing import VectorizedIndexer

from parcels._chunk_cached_array import ChunkCachedArray

__all__ = [
    "chunk_cached_array",
    "chunks",
    "dask_array",
    "numpy_array",
    "out_of_bounds_vectorized_indexer",
    "out_of_bounds_vectorized_indexer_key",
    "vectorized_indexer",
    "vectorized_indexer_key",
]

_DTYPES = [np.float32, np.float64, np.int32, np.int64]

array_dtype = st.sampled_from(_DTYPES).map(np.dtype)


def numpy_array(
    *, shape=None, dtype=None, min_dims: int = 1, max_dims: int = 4, min_side: int = 1, max_side: int = 10
) -> st.SearchStrategy[np.ndarray]:
    """Strategy for generating numpy arrays suitable for wrapping in a ChunkCachedArray.

    Parameters
    ----------
    shape : tuple[int, ...] or SearchStrategy, optional
        Fixed shape (or strategy for shapes). If None, draws shapes using the ``*_dims`` and ``*_side`` bounds.
    dtype : numpy.dtype or SearchStrategy, optional
        Fixed dtype (or strategy for dtypes). If None, draws from common numeric dtypes.
    """
    if shape is None:
        shape = npst.array_shapes(min_dims=min_dims, max_dims=max_dims, min_side=min_side, max_side=max_side)
    if dtype is None:
        dtype = array_dtype
    return npst.arrays(dtype=dtype, shape=shape)


@st.composite
def _dim_chunks(draw, size: int) -> tuple[int, ...]:
    """Strategy for a partition of ``size`` into positive chunk sizes."""
    if size == 0:
        return (0,)
    # Choose a sorted set of cut points strictly inside (0, size)
    cuts = draw(st.sets(st.integers(min_value=1, max_value=size - 1), max_size=size - 1)) if size > 1 else set()
    bounds = [0, *sorted(cuts), size]
    return tuple(int(b - a) for a, b in itertools.pairwise(bounds))


@st.composite
def chunks(draw, shape: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """Strategy for explicit (possibly irregular) dask chunks compatible with ``shape``."""
    return tuple(draw(_dim_chunks(size)) for size in shape)


@st.composite
def dask_array(draw, array=None) -> da.Array:
    """Strategy for generating chunked dask arrays.

    Parameters
    ----------
    array : np.ndarray or SearchStrategy, optional
        Fixed numpy array (or strategy for arrays) to chunk. If None, uses :func:`numpy_array`.
    """
    if array is None:
        array = numpy_array()
    if isinstance(array, st.SearchStrategy):
        array = draw(array)
    return da.from_array(array, chunks=draw(chunks(array.shape)))


@st.composite
def chunk_cached_array(draw, array=None, max_cache_bytes=None) -> ChunkCachedArray:
    """Strategy for generating ChunkCachedArray objects.

    The underlying numpy array can be recovered via ``cca.array.compute()``.

    Parameters
    ----------
    array : dask.array.Array, np.ndarray or SearchStrategy, optional
        Fixed array (or strategy for arrays) to wrap. Numpy arrays are chunked using :func:`dask_array`.
        If None, uses :func:`dask_array`.
    max_cache_bytes : int, optional
        Fixed cache size. If None, draws a size ranging from one byte (i.e., no chunk fits in the cache)
        to large enough to hold the whole array.
    """
    if array is None:
        array = dask_array()
    if isinstance(array, st.SearchStrategy):
        array = draw(array)
    if isinstance(array, np.ndarray):
        array = draw(dask_array(array))
    if max_cache_bytes is None:
        max_cache_bytes = draw(st.integers(min_value=1, max_value=max(array.nbytes, 1)))
    return ChunkCachedArray(array, max_cache_bytes)


@st.composite
def _index_array(draw, size: int, shape: tuple[int, ...]) -> np.ndarray:
    """Strategy for an integer index array of ``shape`` with in-bounds (possibly negative) values for a dim of ``size``."""
    if size == 0:
        assert 0 in shape, "Only an empty index array is in bounds for a zero-length dimension"
        elements = st.just(0)
    else:
        elements = st.integers(min_value=-size, max_value=size - 1)
    dtype = draw(st.sampled_from([np.int32, np.int64, np.intp]))
    return draw(npst.arrays(dtype=dtype, shape=shape, elements=elements))


@st.composite
def vectorized_indexer_key(
    draw,
    shape: tuple[int, ...],
    *,
    allow_slices: bool = True,
    allow_broadcasting: bool = True,
    min_arrays: int = 0,
    min_index_dims: int = 1,
    max_index_dims: int = 3,
    min_index_side: int = 0,
    max_index_side: int = 5,
) -> tuple[slice | np.ndarray, ...]:
    """Strategy for keys usable in vectorized indexing of an array with ``shape``.

    Each entry of the key is either a slice or an integer array. All integer arrays
    have the same number of dimensions and mutually broadcastable shapes (as required
    by :class:`xarray.core.indexing.VectorizedIndexer`).

    Parameters
    ----------
    shape : tuple[int, ...]
        Shape of the array to be indexed.
    allow_slices : bool
        Whether entries may be slices. If False, every entry is an integer array.
    allow_broadcasting : bool
        Whether integer arrays may have different (but broadcastable) shapes. If False,
        all integer arrays have the same shape.
    min_arrays : int
        Minimum number of entries that are integer arrays.
    min_index_dims, max_index_dims : int
        Bounds on the number of dimensions of the integer arrays.
    min_index_side, max_index_side : int
        Bounds on the side lengths of the (broadcast) integer arrays.
    """
    ndim = len(shape)
    min_arrays = min(min_arrays, ndim)
    if allow_slices:
        is_array = draw(st.lists(st.booleans(), min_size=ndim, max_size=ndim).filter(lambda xs: sum(xs) >= min_arrays))
    else:
        is_array = [True] * ndim

    broadcast_shape = draw(
        npst.array_shapes(
            min_dims=min_index_dims, max_dims=max_index_dims, min_side=min_index_side, max_side=max_index_side
        )
    )

    if any(size == 0 for size, use_array in zip(shape, is_array, strict=True) if use_array):
        # Only an empty index array is in bounds for a zero-length dimension
        broadcast_shape = (0, *broadcast_shape[1:])

    key: list[slice | np.ndarray] = []
    for size, use_array in zip(shape, is_array, strict=True):
        if use_array:
            if size == 0 or not allow_broadcasting:
                index_shape = broadcast_shape
            else:
                # Collapse some dims to length 1 so they get broadcast against the other arrays
                index_shape = tuple(draw(st.sampled_from([side, 1])) for side in broadcast_shape)
            key.append(draw(_index_array(size, index_shape)))
        else:
            key.append(draw(st.slices(size)))
    return tuple(key)


def vectorized_indexer(shape: tuple[int, ...], **kwargs) -> st.SearchStrategy[VectorizedIndexer]:
    """Strategy for :class:`xarray.core.indexing.VectorizedIndexer` objects for an array with ``shape``.

    Keyword arguments are passed on to :func:`vectorized_indexer_key`.
    """
    return vectorized_indexer_key(shape, **kwargs).map(VectorizedIndexer)


@st.composite
def out_of_bounds_vectorized_indexer_key(draw, shape: tuple[int, ...], **kwargs) -> tuple[slice | np.ndarray, ...]:
    """Strategy for vectorized indexing keys where at least one integer array has an out-of-bounds index.

    An index ``i`` is out of bounds for a dimension of ``size`` if ``i >= size`` or ``i < -size``.

    Keyword arguments are passed on to :func:`vectorized_indexer_key`, except that at
    least one entry is always a non-empty integer array.
    """
    kwargs["min_arrays"] = max(kwargs.get("min_arrays", 1), 1)
    kwargs["min_index_side"] = max(kwargs.get("min_index_side", 1), 1)
    key = list(draw(vectorized_indexer_key(shape, **kwargs)))

    array_dims = [d for d, k in enumerate(key) if isinstance(k, np.ndarray)]
    d = draw(st.sampled_from(array_dims))
    size = shape[d]
    arr = key[d].copy()

    # Keep within int32 range, as that's the smallest index dtype drawn
    int32 = np.iinfo(np.int32)
    bad_value = draw(
        st.one_of(
            st.integers(min_value=size, max_value=int32.max),
            st.integers(min_value=int32.min, max_value=-size - 1),
        )
    )
    flat_position = draw(st.integers(min_value=0, max_value=arr.size - 1))
    arr.flat[flat_position] = bad_value
    key[d] = arr
    return tuple(key)


def out_of_bounds_vectorized_indexer(shape: tuple[int, ...], **kwargs) -> st.SearchStrategy[VectorizedIndexer]:
    """Strategy for :class:`xarray.core.indexing.VectorizedIndexer` objects with at least one out-of-bounds index.

    Keyword arguments are passed on to :func:`out_of_bounds_vectorized_indexer_key`.
    """
    return out_of_bounds_vectorized_indexer_key(shape, **kwargs).map(VectorizedIndexer)
