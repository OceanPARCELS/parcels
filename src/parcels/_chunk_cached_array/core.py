from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from xarray.core.indexing import BasicIndexer, ExplicitlyIndexedNDArrayMixin, OuterIndexer, VectorizedIndexer
from xarray.namedarray.pycompat import is_duck_array

from .lru_cache import ByteBoundedLRUCache

if TYPE_CHECKING:
    import dask.array
    import xarray as xr


def wrap_dataset(ds: xr.Dataset, max_cache_bytes: int) -> xr.Dataset:
    """Replace all dask-backed data variables with ChunkCachedArray wrappers.

    Returns a shallow copy of the dataset. Each dask-backed data variable's
    internal ``variable._data`` is swapped for a ``ChunkCachedArray`` that
    caches chunks on vectorized indexing. Coordinate variables are loaded
    eagerly into memory to avoid dask task-graph overhead on every
    ``.isel()`` call.

    Parameters
    ----------
    ds : xr.Dataset
        Source dataset (not modified).
    max_cache_bytes : int
        Maximum cache size in bytes, per variable.

    Returns
    -------
    xr.Dataset
        Copy with dask arrays wrapped in ChunkCachedArray.
    """
    from dask.base import is_dask_collection

    ds = ds.copy()
    # Load coordinates eagerly — they are small 1D arrays and keeping them
    # as dask arrays causes expensive task-graph construction on every .isel().
    for name in list(ds.coords):
        ds[name].load()
    for name in ds.data_vars:
        var = ds[name].variable
        if is_duck_array(var._data) and is_dask_collection(var._data):
            var._data = ChunkCachedArray(var._data, max_cache_bytes)  # type: ignore[assignment, arg-type]
    return ds


class ChunkCachedArray(ExplicitlyIndexedNDArrayMixin):
    """Chunk-level LRU cache on top of a dask array for vectorized indexing.

    Implements xarray's ExplicitlyIndexed protocol so it can be used as
    a drop-in replacement for the dask array in ``da.data``. Xarray's
    ``.isel()`` with vectorized indexers will route through ``_vindex_get``,
    which uses the chunk cache. Other indexing modes delegate to the
    underlying dask array.

    On each vectorized index:
      1. Maps global indices -> (chunk_coord, local_index) per dimension.
      2. Fetches missing chunks via dask_array.blocks[...].compute().
      3. Assembles the result from cached numpy arrays.
    """

    def __init__(self, dask_array: dask.array.Array, max_cache_bytes: int) -> None:
        self.array = dask_array
        self.cache = ByteBoundedLRUCache(max_cache_bytes)

        # Precompute chunk boundaries per dimension.
        # _boundaries[d] is a 1D array of cumulative chunk sizes, e.g., [0, 15, 30].
        self._boundaries: list[np.ndarray] = []
        for dim_chunks in dask_array.chunks:
            self._boundaries.append(np.concatenate(([0], np.cumsum(dim_chunks))))

    def get_duck_array(self):
        raise NotImplementedError(
            "Full materialisation of the ChunkCachedArray is not supported, as it has serious (negative) "
            "performance implications. See discussion in https://github.com/Parcels-code/Parcels/issues/2910. "
            "Feel free to continue that discussion if you're running into this error message in your simulations."
        )

    def _raw_vindex(self, *indices: np.ndarray) -> np.ndarray:
        """Vectorized indexing with chunk caching.

        Parameters
        ----------
        *indices : np.ndarray
            One integer index array per dimension. The arrays are broadcast against each other.

        Returns
        -------
        np.ndarray
            Array with the broadcast shape of ``indices`` holding the selected values.
        """
        ndim = len(self.array.chunks)
        assert len(indices) == ndim

        # Step 0: Broadcast the index arrays and flatten them into a list of points,
        # restoring the broadcast shape at the end.
        broadcast = np.broadcast_arrays(*indices)
        out_shape = broadcast[0].shape
        indices = tuple(idx.ravel() for idx in broadcast)
        n_points = int(np.prod(out_shape))
        if n_points == 0:
            return np.empty(out_shape, dtype=self.array.dtype)

        # Step 1: Map global indices to chunk coords and local indices.
        # Normalize negative indices (e.g. -1 → last element) to positive,
        # matching standard numpy fancy-indexing semantics.
        normalized = []
        for d, idx in enumerate(indices):
            size = self.array.shape[d]
            out_of_bounds = (idx < -size) | (idx >= size)
            if out_of_bounds.any():
                raise IndexError(f"index {idx[out_of_bounds][0]} is out of bounds for axis {d} with size {size}")
            normalized.append(np.where(idx < 0, idx + size, idx))
        indices = tuple(normalized)
        chunk_ids = np.empty((ndim, n_points), dtype=np.intp)
        local_indices = np.empty((ndim, n_points), dtype=np.intp)
        for d in range(ndim):
            cid = np.searchsorted(self._boundaries[d], indices[d], side="right") - 1
            chunk_ids[d] = cid
            local_indices[d] = indices[d] - self._boundaries[d][cid]

        # Step 2: Group points by chunk using a structured array for vectorized grouping.
        # Encode each point's chunk coords as a single int for fast grouping.
        # Use np.ravel_multi_index on chunk_ids to get a flat chunk key per point.
        numblocks = np.array(self.array.numblocks, dtype=np.intp)
        flat_keys = np.ravel_multi_index(chunk_ids, numblocks)

        # Sort points by flat chunk key to group them.
        sort_order = np.argsort(flat_keys, kind="quicksort")
        sorted_flat_keys = flat_keys[sort_order]  # type: ignore[call-overload]

        # Find group boundaries.
        boundaries = np.concatenate(([0], np.flatnonzero(np.diff(sorted_flat_keys)) + 1, [n_points]))

        out = np.empty(n_points, dtype=self.array.dtype)
        for g in range(len(boundaries) - 1):
            grp_slice = slice(boundaries[g], boundaries[g + 1])
            grp_indices = sort_order[grp_slice]

            # Recover the chunk key tuple from any point in this group.
            key = tuple(int(chunk_ids[d, grp_indices[0]]) for d in range(ndim))

            chunk_data = self.cache.get(key)
            if chunk_data is None:
                chunk_data = self.array.blocks[key].compute()
                self.cache.put(key, chunk_data)

            # Vectorized fancy-index: extract all points from this chunk at once.
            local_idx = tuple(local_indices[d, grp_indices] for d in range(ndim))
            out[grp_indices] = chunk_data[local_idx]

        return out.reshape(out_shape)

    # --- ExplicitlyIndexed protocol ---

    def _vindex_get(self, indexer: VectorizedIndexer):
        # Dimensions not given an indexer in ``.isel()`` (e.g. a size-1 ``mockZ``)
        # arrive as slices. Under vectorized indexing semantics the output has the
        # broadcast dims of the array indexers first, then one dim per slice. Expand
        # each slice to an index array over its own trailing dim to match.
        key = indexer.tuple
        slices = [np.arange(n)[k] for k, n in zip(key, self.array.shape, strict=True) if isinstance(k, slice)]
        slice_grids = iter(np.meshgrid(*slices, indexing="ij", sparse=True))
        trailing = (np.newaxis,) * len(slices)
        return self._raw_vindex(
            *(next(slice_grids) if isinstance(k, slice) else np.asarray(k)[(..., *trailing)] for k in key)
        )

    def _oindex_get(self, indexer: OuterIndexer):
        # Delegate to dask for orthogonal indexing
        return self.array[indexer.tuple]

    def __getitem__(self, indexer):
        if isinstance(indexer, VectorizedIndexer):
            return self._vindex_get(indexer)
        if isinstance(indexer, OuterIndexer):
            return self._oindex_get(indexer)
        if isinstance(indexer, BasicIndexer):
            return self.array[indexer.tuple]
        return self.array[indexer]
