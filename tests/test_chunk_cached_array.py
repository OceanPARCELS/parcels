import dask.array as da
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from xarray.core.indexing import NumpyIndexingAdapter, VectorizedIndexer

import parcels._strategies.chunk_cached_array as ccast
from parcels._chunk_cached_array import ChunkCachedArray


def _expected_vindex(cca: ChunkCachedArray, indexer: VectorizedIndexer) -> np.ndarray:
    """Reference result for vectorized indexing, using xarray's numpy backend."""
    return NumpyIndexingAdapter(cca.array.compute()).vindex[indexer]


@st.composite
def cca_and_vectorized_indexer(draw, **kwargs) -> tuple[ChunkCachedArray, VectorizedIndexer]:
    cca = draw(ccast.chunk_cached_array())
    indexer = draw(ccast.vectorized_indexer(cca.shape, **kwargs))
    return cca, indexer


# Dask can be slow on first call, which would otherwise trip Hypothesis' deadline
no_deadline = settings(deadline=None)


# --- Strategies ---


@given(ccast.chunk_cached_array())
@no_deadline
def test_chunk_cached_array_strategy(cca):
    assert isinstance(cca, ChunkCachedArray)
    assert isinstance(cca.array, da.Array)


@given(ccast.numpy_array().flatmap(lambda arr: st.tuples(st.just(arr), ccast.dask_array(arr))))
@no_deadline
def test_dask_array_strategy_preserves_values(arrays):
    arr, darr = arrays
    np.testing.assert_array_equal(darr.compute(), arr)


@given(ccast.numpy_array().flatmap(lambda arr: st.tuples(st.just(arr), ccast.vectorized_indexer(arr.shape))))
@no_deadline
def test_vectorized_indexer_strategy_is_valid_for_numpy(args):
    arr, indexer = args
    NumpyIndexingAdapter(arr).vindex[indexer]  # shouldn't error


@given(
    ccast.numpy_array().flatmap(lambda arr: st.tuples(st.just(arr), ccast.out_of_bounds_vectorized_indexer(arr.shape)))
)
@no_deadline
def test_out_of_bounds_vectorized_indexer_strategy_raises_for_numpy(args):
    arr, indexer = args
    with pytest.raises(IndexError):
        NumpyIndexingAdapter(arr).vindex[indexer]


# --- Vectorized indexing ---


@given(cca_and_vectorized_indexer(allow_slices=False, allow_broadcasting=False, max_index_dims=1, min_index_side=1))
@no_deadline
def test_vindex_1d_integer_arrays(args):
    """The case used within Parcels: one (non-empty) 1D integer array per dimension, all of equal length."""
    cca, indexer = args
    np.testing.assert_array_equal(cca.vindex[indexer], _expected_vindex(cca, indexer))


@given(cca_and_vectorized_indexer(allow_slices=False, allow_broadcasting=False, max_index_dims=1, min_index_side=1))
@no_deadline
def test_vindex_repeated_indexing_uses_cache_consistently(args):
    cca, indexer = args
    expected = _expected_vindex(cca, indexer)
    np.testing.assert_array_equal(cca.vindex[indexer], expected)
    np.testing.assert_array_equal(cca.vindex[indexer], expected)  # now (partially) served from cache
    assert cca.cache.current_bytes <= cca.cache._max_bytes


@given(cca_and_vectorized_indexer(allow_slices=False, max_index_dims=1, max_index_side=0))
@no_deadline
def test_vindex_empty_integer_arrays(args):
    cca, indexer = args
    np.testing.assert_array_equal(cca.vindex[indexer], _expected_vindex(cca, indexer))


@given(cca_and_vectorized_indexer(allow_slices=False, max_index_dims=1, min_index_side=1))
@no_deadline
def test_vindex_1d_integer_arrays_broadcasting(args):
    cca, indexer = args
    np.testing.assert_array_equal(cca.vindex[indexer], _expected_vindex(cca, indexer))


@given(cca_and_vectorized_indexer(allow_slices=False, allow_broadcasting=False, min_index_dims=2, min_index_side=1))
@no_deadline
def test_vindex_nd_integer_arrays(args):
    cca, indexer = args
    np.testing.assert_array_equal(cca.vindex[indexer], _expected_vindex(cca, indexer))


@given(cca_and_vectorized_indexer().filter(lambda args: any(isinstance(k, slice) for k in args[1].tuple)))
@no_deadline
def test_vindex_integer_arrays_and_slices(args):
    cca, indexer = args
    np.testing.assert_array_equal(cca.vindex[indexer], _expected_vindex(cca, indexer))


@given(cca_and_vectorized_indexer())
@no_deadline
def test_vindex_any_vectorized_indexer(args):
    cca, indexer = args
    np.testing.assert_array_equal(cca.vindex[indexer], _expected_vindex(cca, indexer))


@st.composite
def cca_and_out_of_bounds_vectorized_indexer(draw, **kwargs) -> tuple[ChunkCachedArray, VectorizedIndexer]:
    cca = draw(ccast.chunk_cached_array())
    indexer = draw(ccast.out_of_bounds_vectorized_indexer(cca.shape, **kwargs))
    return cca, indexer


@given(cca_and_out_of_bounds_vectorized_indexer(allow_slices=False))
@no_deadline
def test_vindex_out_of_bounds_raises_index_error(args):
    cca, indexer = args
    with pytest.raises(IndexError):
        cca.vindex[indexer]
