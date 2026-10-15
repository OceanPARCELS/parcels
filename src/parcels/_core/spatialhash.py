import sys
import warnings
from typing import IO

import numpy as np

from parcels._core.index_search import (
    GRID_SEARCH_ERROR,
    _latlon_rad_to_xyz,
    curvilinear_point_in_cell,
    uxgrid_point_in_cell,
)
from parcels._core.warnings import FieldSetWarning
from parcels._python import isinstance_noimport
from parcels._repr_utils import spatialhash_describe

# Budget on the total number of (face, hash cell) pairs in the hash table:
# max(_HASH_ENTRIES_PER_FACE * nfaces, _HASH_ENTRY_BUDGET_MIN).
# When the hash grid cell size, set by the bitwidth, would result in too many
# hash entries per face, the hash grid is coarsened by lowering the bitwidth
# The target value for the number of hash entries per face is chosen to keep
# the number of particle in cell checks as small as possible, while also
# minimizing the hash table construction memory footprint.
_HASH_ENTRIES_PER_FACE = 16
_HASH_ENTRY_BUDGET_MIN = 2**22
_HASH_MAX_BITWIDTH = 1023


class SpatialHash:
    """Custom data structure that is used for performing grid searches using Spatial Hashing. This class constructs an overlying
    uniformly spaced rectilinear grid, called the "hash grid" on top parcels.XGrid. It is particularly useful for grid searching
    on curvilinear grids. Faces in the Xgrid are related to the cells in the hash grid by determining the hash cells the bounding box
    of the unstructured face cells overlap with.

    Parameters
    ----------
    grid : parcels.XGrid
        Source grid used to construct the hash grid and hash table

    Notes
    -----
    Does not currently support queries on periodic elements.
    """

    def __init__(
        self,
        grid,
    ):
        if isinstance_noimport(grid, "XGrid"):
            self._point_in_cell = curvilinear_point_in_cell
        elif isinstance_noimport(grid, "UxGrid"):
            self._point_in_cell = uxgrid_point_in_cell
        else:
            raise ValueError("Expected `grid` to be a parcels.XGrid or parcels.UxGrid")

        self._source_grid = grid
        self._bitwidth = _HASH_MAX_BITWIDTH  # Max integer to use per coordinate in quantization (10 bits = 0..1023)

        if isinstance_noimport(grid, "XGrid"):
            self._coord_dim = 2  # Number of computational coordinates is 2 (bilinear interpolation)
            if self._source_grid._mesh.is_spherical():
                lon = np.deg2rad(self._source_grid.lon)
                lat = np.deg2rad(self._source_grid.lat)
                x, y, z = _latlon_rad_to_xyz(lat, lon)

                # The 4 nodes of each face, wound around it
                nodes = np.stack((x, y, z), axis=-1)
                verts = np.stack(
                    (
                        nodes[:-1, :-1],
                        nodes[:-1, 1:],
                        nodes[1:, 1:],
                        nodes[1:, :-1],
                    ),
                    axis=-2,
                )

                # Compute the exact bounding box of each face. A face's edges are
                # great-circle arcs, so its true x/y/z extent is often not spanned by
                # its node coordinates alone.
                self._xlow, self._xhigh, self._ylow, self._yhigh, self._zlow, self._zhigh = _spherical_face_bounds(
                    verts
                )

                # Boundaries of the hash grid are the Cartesian bounding box of the
                # transformed grid, so that regional domains retain full quantization
                # resolution instead of spreading it over the whole unit cube
                self._xmin = np.nanmin(self._xlow)
                self._xmax = np.nanmax(self._xhigh)
                self._ymin = np.nanmin(self._ylow)
                self._ymax = np.nanmax(self._yhigh)
                self._zmin = np.nanmin(self._zlow)
                self._zmax = np.nanmax(self._zhigh)

                degenerate_mask = _find_degenerate_xgrid_faces(x, y, z)
                degeneracy_count = np.sum(degenerate_mask)
                if degeneracy_count > 0:
                    degen_locs = np.argwhere(degenerate_mask)  # shape (N, 2), columns are (j, i)
                    max_shown = np.min([degeneracy_count, 5])
                    shown = degen_locs[:max_shown]
                    loc_str = ", ".join(f"(j={loc[0]}, i={loc[1]})" for loc in shown)
                    warnings.warn(
                        f"Grid contains {degeneracy_count} degenerate faces that span a large portion of the "
                        "hash grid. This is most likely due to a mesh that isn't fully defined (e.g., points corresponding to land with lat/lon masked to 0). "
                        "You may experience runtime crashes due to high memory usage in the hash table or cell lookup failures for particles"
                        "in the vicinity of these degenerate cells."
                        f"First degenerate face location(s): {loc_str}.",
                        FieldSetWarning,
                        stacklevel=2,
                    )

            else:
                # Boundaries of the hash grid are the bounding box of the source grid
                self._xmin = np.nanmin(self._source_grid.lon)
                self._xmax = np.nanmax(self._source_grid.lon)
                self._ymin = np.nanmin(self._source_grid.lat)
                self._ymax = np.nanmax(self._source_grid.lat)
                # setting min and max below is needed for mesh="flat"
                self._zmin = 0.0
                self._zmax = 0.0
                x = self._source_grid.lon
                y = self._source_grid.lat

                _xbound = np.stack(
                    (
                        x[:-1, :-1],
                        x[:-1, 1:],
                        x[1:, 1:],
                        x[1:, :-1],
                    ),
                    axis=-1,
                )
                _ybound = np.stack(
                    (
                        y[:-1, :-1],
                        y[:-1, 1:],
                        y[1:, 1:],
                        y[1:, :-1],
                    ),
                    axis=-1,
                )
                # Compute bounding box of each face
                self._xlow = np.min(_xbound, axis=-1)
                self._xhigh = np.max(_xbound, axis=-1)
                self._ylow = np.min(_ybound, axis=-1)
                self._yhigh = np.max(_ybound, axis=-1)
                self._zlow = np.zeros_like(self._xlow)
                self._zhigh = np.zeros_like(self._xlow)

        elif isinstance_noimport(grid, "UxGrid"):
            self._coord_dim = grid.uxgrid.n_max_face_nodes  # Number of barycentric coordinates
            if self._source_grid._mesh.is_spherical():
                # Reshape node coordinates to (nfaces, nnodes_per_face)
                nids = self._source_grid.uxgrid.face_node_connectivity.values
                lon = self._source_grid.uxgrid.node_lon.values[nids]
                lat = self._source_grid.uxgrid.node_lat.values[nids]
                vx, vy, vz = _latlon_rad_to_xyz(np.deg2rad(lat), np.deg2rad(lon))
                verts = np.stack([vx, vy, vz], axis=-1)  # (nfaces, 3 nodes, 3)

                self._xlow, self._xhigh, self._ylow, self._yhigh, self._zlow, self._zhigh = _spherical_face_bounds(
                    verts
                )

                # Boundaries of the hash grid are the Cartesian bounding box of the
                # transformed grid, so that regional domains retain full quantization
                # resolution instead of spreading it over the whole unit cube
                self._xmin = self._xlow.min()
                self._xmax = self._xhigh.max()
                self._ymin = self._ylow.min()
                self._ymax = self._yhigh.max()
                self._zmin = self._zlow.min()
                self._zmax = self._zhigh.max()

                self._xlow = np.atleast_2d(self._xlow)
                self._xhigh = np.atleast_2d(self._xhigh)
                self._ylow = np.atleast_2d(self._ylow)
                self._yhigh = np.atleast_2d(self._yhigh)
                self._zlow = np.atleast_2d(self._zlow)
                self._zhigh = np.atleast_2d(self._zhigh)

            else:
                # Boundaries of the hash grid are the bounding box of the source grid
                self._xmin = self._source_grid.uxgrid.node_lon.min().values
                self._xmax = self._source_grid.uxgrid.node_lon.max().values
                self._ymin = self._source_grid.uxgrid.node_lat.min().values
                self._ymax = self._source_grid.uxgrid.node_lat.max().values
                # setting min and max below is needed for mesh="flat"
                self._zmin = 0.0
                self._zmax = 0.0
                # Reshape node coordinates to (nfaces, nnodes_per_face)
                nids = self._source_grid.uxgrid.face_node_connectivity.values
                lon = self._source_grid.uxgrid.node_lon.values[nids]
                lat = self._source_grid.uxgrid.node_lat.values[nids]

                # Compute bounding box of each face
                self._xlow = np.atleast_2d(np.min(lon, axis=-1))
                self._xhigh = np.atleast_2d(np.max(lon, axis=-1))
                self._ylow = np.atleast_2d(np.min(lat, axis=-1))
                self._yhigh = np.atleast_2d(np.max(lat, axis=-1))
                self._zlow = np.zeros_like(self._xlow)
                self._zhigh = np.zeros_like(self._xlow)

        # Cap the quantization resolution so the hash table stays within a fixed entry
        # budget.
        budget = max(_HASH_ENTRIES_PER_FACE * np.size(self._xlow), _HASH_ENTRY_BUDGET_MIN)
        if self._total_hash_entries(self._bitwidth) > budget:
            # Binary search for the largest bitwidth whose table fits the budget. The
            # entry count is not perfectly monotone in bitwidth (cell-boundary flooring
            # effects), so the result may sit marginally below the true maximum; any
            # in-budget bitwidth is valid. At bitwidth 1 the count equals nfaces, which
            # is always within budget, so the search cannot fail.
            lo, hi = 1, self._bitwidth
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if self._total_hash_entries(mid) <= budget:
                    lo = mid
                else:
                    hi = mid - 1
            self._bitwidth = lo

        # Generate the mapping from the hash indices to unstructured grid elements
        self._hash_table = self._initialize_hash_table()

    def _total_hash_entries(self, bitwidth):
        """Total number of (face, hash cell) pairs the hash table would hold at a given
        quantization resolution, i.e. the summed hash-cell count of all face bounding boxes.
        """
        xqlow, yqlow, zqlow = quantize_coordinates(
            self._xlow,
            self._ylow,
            self._zlow,
            self._xmin,
            self._xmax,
            self._ymin,
            self._ymax,
            self._zmin,
            self._zmax,
            bitwidth,
        )
        xqhigh, yqhigh, zqhigh = quantize_coordinates(
            self._xhigh,
            self._yhigh,
            self._zhigh,
            self._xmin,
            self._xmax,
            self._ymin,
            self._ymax,
            self._zmin,
            self._zmax,
            bitwidth,
        )
        nx = xqhigh.astype(np.int64) - xqlow + 1
        ny = yqhigh.astype(np.int64) - yqlow + 1
        nz = zqhigh.astype(np.int64) - zqlow + 1
        # NaN values are not allowed in the SpatialHash table, so faces with a NaN
        # bounding box do not contribute to the entry count
        valid_face = _generate_valid_mask(self._xlow, self._xhigh, self._ylow, self._yhigh, self._zlow, self._zhigh)
        return int(np.where(valid_face, nx * ny * nz, 0).sum())

    def _initialize_hash_table(self):
        """Create a mapping that relates unstructured grid faces to hash indices by determining
        which faces overlap with which hash cells
        """
        # Quantize the bounding box in each direction
        xqlow, yqlow, zqlow = quantize_coordinates(
            self._xlow,
            self._ylow,
            self._zlow,
            self._xmin,
            self._xmax,
            self._ymin,
            self._ymax,
            self._zmin,
            self._zmax,
            self._bitwidth,
        )

        xqhigh, yqhigh, zqhigh = quantize_coordinates(
            self._xhigh,
            self._yhigh,
            self._zhigh,
            self._xmin,
            self._xmax,
            self._ymin,
            self._ymax,
            self._zmin,
            self._zmax,
            self._bitwidth,
        )
        xqlow = xqlow.ravel().astype(np.int32, copy=False)
        yqlow = yqlow.ravel().astype(np.int32, copy=False)
        zqlow = zqlow.ravel().astype(np.int32, copy=False)
        xqhigh = xqhigh.ravel().astype(np.int32, copy=False)
        yqhigh = yqhigh.ravel().astype(np.int32, copy=False)
        zqhigh = zqhigh.ravel().astype(np.int32, copy=False)
        nx = (xqhigh - xqlow + 1).astype(np.int32, copy=False)
        ny = (yqhigh - yqlow + 1).astype(np.int32, copy=False)
        nz = (zqhigh - zqlow + 1).astype(np.int32, copy=False)

        # prevent NaN values from entering the SpatialHash table by setting their
        # num_hash_per_face equal to 0
        valid_face = _generate_valid_mask(self._xlow, self._xhigh, self._ylow, self._yhigh, self._zlow, self._zhigh)
        num_hash_per_face = np.where(valid_face.ravel(), nx * ny * nz, 0).astype(
            np.int32, copy=False
        )  # Since nx, ny, nz are in the 10-bit range, their product fits in int32
        # Sums over faces can exceed int32, so accumulate in int64
        total_hash_entries = int(num_hash_per_face.sum(dtype=np.int64))
        # Entry indices fit in int32 for all but extreme cases; fall back to int64 when needed
        idx_dtype = np.int64 if total_hash_entries > np.iinfo(np.int32).max else np.int32

        # Every face overlaps at least one hash cell (nx, ny, nz >= 1 since quantization
        # is monotone), and contributes one hash entry per cell of its quantized bounding
        # box. Entries are generated in face-major order: face_ids maps each entry to its
        # face, and intra enumerates the cells of that face's box (0..num_hash_per_face-1).
        nface = np.size(self._xlow)
        face_ids = np.repeat(np.arange(nface, dtype=np.uint32), num_hash_per_face)
        face_starts = np.concatenate(([0], np.cumsum(num_hash_per_face, dtype=np.int64)))[:-1]
        intra = np.arange(total_hash_entries, dtype=idx_dtype) - np.repeat(
            face_starts.astype(idx_dtype, copy=False), num_hash_per_face
        )

        # Derive (xi, yi, zi) cell offsets within each face's box from intra,
        # then shift by the per-face low corner to get quantized cell coordinates
        ny_nz = np.repeat(ny * nz, num_hash_per_face)
        nz_rep = np.repeat(nz, num_hash_per_face)

        xi = intra // ny_nz
        rem = intra % ny_nz
        yi = rem // nz_rep
        zi = rem % nz_rep

        xq = np.repeat(xqlow, num_hash_per_face) + xi
        yq = np.repeat(yqlow, num_hash_per_face) + yi
        zq = np.repeat(zqlow, num_hash_per_face) + zi

        # Vectorized morton encode for all entries at once, already in face-major order
        morton_codes = _encode_quantized_morton3d(xq, yq, zq)
        del intra, rem, xi, yi, zi, ny_nz, nz_rep, xq, yq, zq

        # Sort entries by morton code. Each (code, face) pair is fused into one uint64
        # with the code in the high 32 bits and the face id in the low 32 bits: unsigned
        # comparison then orders by code, with ties broken by ascending face id. Sorting
        # the fused array in place avoids the argsort permutation array and the gather
        # copies it would imply. Pairs are unique, so the ordering is deterministic.
        packed = morton_codes.astype(np.uint64)
        del morton_codes
        packed <<= np.uint64(32)
        np.bitwise_or(packed, face_ids, out=packed)
        del face_ids
        # Perform a single sort on the packed (morton_code | face_id ) list
        packed.sort()
        # Trunctating back to a uint32 keeps the lower 32 bits (the face_id's)
        face_sorted = packed.astype(np.uint32)
        # Purge the face ids from the packed list to retain only the morton codes
        packed >>= np.uint64(32)
        # Cast the morton codes back to uint32
        morton_codes_sorted = packed.astype(np.uint32)
        del packed

        # Get a list of unique morton codes and their corresponding starts and counts (CSR format).
        # The codes are already sorted at this point, first by morton code, then by face_id
        # Starting indices of the matrix rows are located by finding indices where the morton codes differ
        starts = np.concatenate(([0], np.flatnonzero(morton_codes_sorted[1:] != morton_codes_sorted[:-1]) + 1))
        # The unique keys for the hash table are the unique morton codes
        keys = morton_codes_sorted[starts]
        # The number of faces per hash keys (morton codes) is easily calculated as the difference betwee the start values
        counts = np.diff(np.concatenate((starts, [morton_codes_sorted.size])))

        # The flat face id is stored (4 bytes per entry); query() unravels the gathered
        # candidates to (j, i) on demand, instead of holding two precomputed int64
        # index arrays (16 bytes per entry) for the lifetime of the grid.
        hash_table = {
            "keys": keys,
            "starts": starts,
            "counts": counts,
            "faces": face_sorted,
        }
        return hash_table

    def query(self, y, x):
        """
        Queries the hash table and finds the closes face in the source grid for each coordinate pair.

        Parameters
        ----------
        y : array_like
            y-coordinates in degrees (lat) to query of shape (N,) where N is the number of queries.
        x : array_like
            x-coordinates in degrees (lon) to query of shape (N,) where N is the number of queries.

        Returns
        -------
        j : ndarray, shape (N,)
            j-indices of the located face in the source grid for each query. If no face was found, GRID_SEARCH_ERROR is returned.
        i : ndarray, shape (N,)
            i-indices of the located face in the source grid for each query. If no face was found, GRID_SEARCH_ERROR is returned.
        coords : ndarray, shape (N, 2)
            The local coordinates (xsi, eta) of the located face in the source grid for each query.
            If no face was found, (-1.0, -1.0)
        """
        keys = self._hash_table["keys"]
        starts = self._hash_table["starts"]
        counts = self._hash_table["counts"]
        faces = self._hash_table["faces"]

        y = np.asarray(y)
        x = np.asarray(x)
        if self._source_grid._mesh.is_spherical():
            # Convert coords to Cartesian coordinates (x, y, z)
            lat = np.deg2rad(y)
            lon = np.deg2rad(x)
            qx, qy, qz = _latlon_rad_to_xyz(lat, lon)
        else:
            # For Cartesian grids, use the coordinates directly
            qx = x
            qy = y
            qz = np.zeros_like(qx)

        query_codes = _encode_morton3d(
            qx,
            qy,
            qz,
            self._xmin,
            self._xmax,
            self._ymin,
            self._ymax,
            self._zmin,
            self._zmax,
            bitwidth=self._bitwidth,
        ).ravel()
        num_queries = query_codes.size

        # Locate each query in the unique key array
        pos = np.searchsorted(keys, query_codes)  # pos is shape (num_queries,)

        # Valid hits: inside range with finite query coordinates and query codes give exact morton code match.
        valid = (pos < len(keys)) & np.isfinite(x) & np.isfinite(y)
        # Clip pos to valid range to avoid out-of-bounds indexing
        pos = np.clip(pos, 0, len(keys) - 1)
        # Further filter out false positives from searchsorted by checking for exact code match
        valid[valid] &= query_codes[valid] == keys[pos[valid]]

        # Pre-allocate i and j indices of the best match for each query
        # Default values to -1 (no match case)
        j_best = np.full(num_queries, GRID_SEARCH_ERROR, dtype=np.int32)
        i_best = np.full(num_queries, GRID_SEARCH_ERROR, dtype=np.int32)

        # How many matches each query has; hit_counts[i] is the number of hits for query i
        hit_counts = np.where(valid, counts[pos], 0).astype(np.int32)  # has shape (num_queries,)
        if hit_counts.sum() == 0:
            return (
                j_best.reshape(query_codes.shape),
                i_best.reshape(query_codes.shape),
                np.full((num_queries, self._coord_dim), -1.0, dtype=np.float32),
            )

        # Now, for each query, we need to gather the candidate (j,i) indices from the hash table
        # Each j,i pair needs to be repeated hit_counts[i] times, only when there are hits.

        # Boolean array for keeping track of which queries have candidates
        has_hits = hit_counts > 0  # shape (num_queries,), True for queries that had candidates

        # A quick lookup array that maps all candindates back to its query index
        q_index_for_candidate = np.repeat(
            np.arange(num_queries, dtype=np.int32), hit_counts
        )  # shape (hit_counts.sum(),)
        # Map all candidates to positions in the hash table
        hash_positions = pos[q_index_for_candidate]  # shape (hit_counts.sum(),)

        # Now that we have the positions in the hash table for each table, we can gather the (j,i) pairs for each candidate
        # We do this in a vectorized way by using a CSR-like approach
        # starts[pos[q_index_for_candidate]] gives the starting point in the hash table for each candidate
        # hit_counts gives the number of candidates for each query

        # We need to build an array that gives the offset within each query's candidates
        offsets = np.concatenate(([0], np.cumsum(hit_counts))).astype(np.int32)  # shape (num_queries+1,)
        total = int(offsets[-1])  # total number of candidates across all queries

        # Now, for each candidate, we need a simple array that tells us its "local candidate id" within its query
        # This way, we can easily take the starts[pos[q_index_for_candidate]] and add this local id to get the absolute index
        # We calculate this by computing the "global candidate number" (0..total-1) and subtracting the offsets of the corresponding query
        # This gives us an array that goes from 0..hit_counts[i]-1 for each query i
        intra = np.arange(total, dtype=np.int32) - np.repeat(offsets[:-1], hit_counts)  # shape (hit_counts.sum(),)

        # starts[pos[q_index_for_candidate]] + intra gives a list of positions in the hash table that we can
        # use to quickly gather the (i,j) pairs for each query
        source_idx = starts[hash_positions].astype(np.int32) + intra

        # Gather all candidate face ids in one shot and unravel them to (j, i) pairs;
        # only the gathered candidates are unraveled, not the whole table
        face_all = faces[source_idx]
        j_all, i_all = np.unravel_index(face_all, self._xlow.shape)

        # Now we need to construct arrays that repeats the y and x coordinates for each candidate
        # to enable vectorized point-in-cell checks
        y_rep = np.repeat(y, hit_counts)  # shape (hit_counts.sum(),)
        x_rep = np.repeat(x, hit_counts)  # shape (hit_counts.sum(),)

        # For each query we perform a point in cell check.
        is_in_face, coordinates = self._point_in_cell(self._source_grid, y_rep, x_rep, j_all, i_all)

        coords_best = np.full((num_queries, coordinates.shape[1]), -1.0, dtype=np.float32)

        # For each query that has hits, we need to find the first candidate that was inside the face
        f_indices = np.flatnonzero(is_in_face)  # Indices of all faces that contained the point
        # For each true position, find which query it belongs to by searching offsets
        # Query index q satisfies offsets[q] <= pos < offsets[q+1].
        q = np.searchsorted(offsets[1:], f_indices, side="right")

        uniq_q, q_idx = np.unique(q, return_index=True)
        keep = has_hits[uniq_q]

        if keep.any():
            uniq_q = uniq_q[keep]
            pos_first = f_indices[q_idx[keep]]

            # Directly scatter: the code wants the first True inside each slice
            j_best[uniq_q] = j_all[pos_first]
            i_best[uniq_q] = i_all[pos_first]
            coords_best[uniq_q] = coordinates[pos_first]

        return (
            j_best.reshape(query_codes.shape),
            i_best.reshape(query_codes.shape),
            coords_best.reshape((num_queries, coordinates.shape[1])),
        )

    def describe(self, buf: IO | None = None) -> None:
        """
        Summary of the SpatialHash's hash-table statistics (resolution, occupancy,
        entry counts).

        Parameters
        ----------
        buf : file-like, default: sys.stdout
            writable buffer
        """
        if buf is None:
            buf = sys.stdout
        assert buf is not None

        buf.write(spatialhash_describe(self))


def _dilate_bits(n):
    """
    Takes a 10-bit integer n, in range [0,1023], and "dilates" its bits so that
    there are two zeros between each bit of n in the result.

    This is a preparation step for building a 3D Morton code:
    - One axis (x, y, or z) is dilated like this.
    - Then the three dilated coordinates are bitwise interleaved
      to produce the full 30-bit Morton code.

    Example:
        Input n:  b9 b8 b7 b6 b5 b4 b3 b2 b1 b0
        Output:   b9 0 0 b8 0 0 b7 0 0 ... b0 0 0
    """
    n = np.asarray(n, dtype=np.uint32)

    # Step 1: Keep only the lowest 10 bits of n
    # Mask = 0x3FF = binary 11 1111 1111
    n &= np.uint32(0x000003FF)

    # Step 2: First spreading stage
    # Shift left by 16 and OR with original.
    # This spreads the bits apart, but introduces overlaps.
    # Mask 0xff0000ff clears out the unwanted overlaps.
    n = (n | (n << np.uint32(16))) & np.uint32(0xFF0000FF)

    # Step 3: Second spreading stage
    # Similar idea: shift left by 8, OR, then mask.
    # Now the bits are further separated.
    n = (n | (n << np.uint32(8))) & np.uint32(0x0300F00F)

    # Step 4: Third spreading stage
    # Shift by 4, OR, mask again.
    # At this point, there are 1 or 2 zeros between many of the bits.
    n = (n | (n << np.uint32(4))) & np.uint32(0x030C30C3)

    # Step 5: Final spreading stage
    # Shift by 2, OR, mask.
    # After this, each original bit is isolated with exactly two zeros
    # between it and the next bit, ready for 3D Morton interleaving.
    n = (n | (n << np.uint32(2))) & np.uint32(0x09249249)

    # Return the dilated value.
    return n


def _find_degenerate_xgrid_faces(x, y, z, threshold_factor=10):
    """Identify faces in structured grids that potentially span large portions of
    the underlying hash grid (e.g., due to the mesh being incomplete, with 0.0 stored in missing lon/lat points). Such degenerate faces can result in high memory requirements
    for the hash table.

    Detection is based on the maximum great-circle edge length of each cell.  A cell
    is flagged as degenerate when its longest edge exceeds ``threshold_factor`` multiplied by
    the 99th percentile of all edge lengths.

    Parameters
    ----------
    x, y, z : ndarray, shape (ny, nx)
        Unit-sphere Cartesian coordinates of the grid nodes.
    threshold_factor : float, optional
        Multiplier applied to the 99th-percentile edge length to set the threshold.
        Default is 10.

    Returns
    -------
    degenerate : ndarray of bool, shape (ny-1, nx-1)
        True for each cell whose maximum edge length exceeds the threshold.
    """

    # Chord length between two sets of points on the unit sphere, shape (ny-1, nx-1)
    def _chord(p1, p2):
        return np.sqrt(((p1 - p2) ** 2).sum(axis=-1))

    pts = np.stack([x, y, z], axis=-1)
    c00, c01 = pts[:-1, :-1], pts[:-1, 1:]
    c10, c11 = pts[1:, :-1], pts[1:, 1:]

    # Maximum chord across all four edges and both diagonals
    max_chord = np.maximum.reduce(
        [
            _chord(c00, c01),
            _chord(c10, c11),
            _chord(c00, c10),
            _chord(c01, c11),
            _chord(c00, c11),
            _chord(c01, c10),
        ]
    )

    threshold = threshold_factor * np.percentile(max_chord, 99)
    return max_chord > threshold


# The 6 points where x, y, or z reaches its absolute max/min over the whole unit
# sphere: the poles (+-z) and the equator/prime-meridian crossings (+-x, +-y).
_AXIS_POINTS = np.array(
    [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ]
)


def _spherical_face_bounds(verts):
    """Exact per-face x/y/z bounding box for convex faces on the unit sphere,
    computed in closed form (no sampling).

    A face's edges are great-circle arcs, so the true min/max of x, y, or z
    reached anywhere on a face is often not one of its node coordinates. The
    surface can bulge past every node along an edge, or, for a face containing a
    pole or an equator/prime-meridian crossing, reach its extreme at a single
    interior point. The spatial hash grid must include these points. They are
    computed based on the assumption that a face's extrema sit at one of the following:

      1. The nodes themselves.
      2. Any of the 6 points in `_AXIS_POINTS` that could lie inside the face. If this
         is the case then that particular face's extrema is +-1 for that particular bound.
      3. For each edge, the point(s) along that great-circle arc where x, y, or z
         is extremized. This extreme does not fall on a node when a cos representing
         the point in space hits a min/max.

    Parameters
    ----------
    verts : ndarray, shape (nfaces, nodes_per_face, 3)
        Unit-sphere xyz coordinates of each face's nodes, in winding order.
        The last axis is (x, y, z).

    Returns
    -------
    xlow, xhigh, ylow, yhigh, zlow, zhigh : ndarray, shape (nfaces,)
    """
    # --- Step 1: Calculate the bounds from the nodes themselves ----
    nodes_per_face = verts.shape[-2]

    low = np.min(verts, axis=-2)  # (..., 3), so low[..., 0] holds every face's x bound
    high = np.max(verts, axis=-2)

    # --- Step 2: Check which faces the global extrema fall into (using a gnomonic projection) ---
    # A point is inside a triangle of nodes when it can be written as a combination of
    # the three node vectors using no negative weights. Solving for those weights needs
    # exactly three nodes, so a face with more nodes is split into a fan of triangles
    # sharing its first node, and a point inside any one of those triangles is inside
    # the face.
    inside = np.zeros(verts.shape[:-2] + (len(_AXIS_POINTS),), dtype=bool)  # (..., 6)
    for node in range(1, nodes_per_face - 1):
        M = np.stack([verts[..., 0, :], verts[..., node, :], verts[..., node + 1, :]], axis=-1)

        # A triangle with no area encloses nothing. Rather
        # than let np.linalg.solve raise on it, conver those faces' weights to NaN.
        weights = np.full(M.shape[:-1] + (len(_AXIS_POINTS),), np.nan)
        has_area = np.abs(np.linalg.det(M)) > 1e-14
        if has_area.any():
            weights[has_area] = np.linalg.solve(M[has_area], _AXIS_POINTS.T)

        # An axis point is in the triangle if its weights are >=0 (1e-12 accounts for
        # machine precision).
        inside |= np.all(weights >= -1e-12, axis=-2)

    # For faces that contain a global extreme point, override the bound to use that point (which
    # will be either 1 or -1 on the unit sphere). _AXIS_POINTS alternates between the two
    # ends of an axis, so the even entries are the +1 ends and the odd ones the -1 ends.
    high = np.where(inside[..., 0::2], 1.0, high)
    low = np.where(inside[..., 1::2], -1.0, low)

    # --- Step 3: per-edge extrema ------------------------------------------------
    # Loop on the pairs of adjacent nodes that make up each face's edges, wrapping
    # from the last node back to the first.
    for node in range(nodes_per_face):
        a = verts[..., node, :]
        b = verts[..., (node + 1) % nodes_per_face, :]
        # Compute the total angular length of the arc created by points a and b
        cos_delta = np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)  # (...)
        delta = np.arccos(cos_delta)  # (...)

        # Construct a unit vector perpendicular to a that points towards b. (a, u) then
        # forms an orthonormal basis for that plane, so any point on the arc created
        # by the vertices (a,b) is in that plane, and can be computed as cos(theta)*a + sin(theta)*u
        # for theta in [0, delta]. theta=0 recovers a and theta=delta recovers b.
        u = b - cos_delta[..., None] * a
        edge_length = np.linalg.norm(u, axis=-1)
        # An edge between two identical nodes has no direction to normalize; dividing
        # the resulting zero vector by 1 keeps it zero rather than making it NaN.
        u = u / np.where(edge_length == 0.0, 1.0, edge_length)[..., None]

        # Using the identity Acos(theta) + Bsin(theta) = Rcos(theta - phi) where
        # R is an amplitude R = sqrt(A^2 + B^2) and phi = arctan2(B, A). The edge's
        # max or min are then given by the location where theta=0, pi, plus the phase
        # shift of phi.
        amp = np.hypot(a, u)  # (..., 3): wave amplitude R, per coordinate
        phase = np.arctan2(u, a)  # (..., 3): wave phase phi, per coordinate

        # Compute the values of theta that result in a bounding box extremum.
        theta_max = phase % (2 * np.pi)  # (..., 3)
        theta_min = (phase + np.pi) % (2 * np.pi)  # (..., 3)

        # Calculate whether or not an extremum-generating theta falls into the
        # arc constructed by each edge.
        valid_max = theta_max <= delta[..., None]  # (..., 3)
        valid_min = theta_min <= delta[..., None]  # (..., 3)

        # For theta that fall on the arc of each edge, keep whichever is more extreme:
        # the currently computed bound, or this edge's candidate for that coordinate.
        high = np.maximum(high, np.where(valid_max, amp, -np.inf))
        low = np.minimum(low, np.where(valid_min, -amp, np.inf))

    return low[..., 0], high[..., 0], low[..., 1], high[..., 1], low[..., 2], high[..., 2]


def quantize_coordinates(x, y, z, xmin, xmax, ymin, ymax, zmin, zmax, bitwidth=1023):
    """
    Normalize (x, y, z) to [0, 1] over their bounding box, then quantize to 10 bits each (0..1023).

    Parameters
    ----------
    x, y, z : array_like
        Input coordinates to quantize. Can be scalars or arrays (broadcasting applies).
    xmin, xmax : float
        Minimum and maximum bounds for x coordinate.
    ymin, ymax : float
        Minimum and maximum bounds for y coordinate.
    zmin, zmax : float
        Minimum and maximum bounds for z coordinate.

    Returns
    -------
    xq, yq, zq : ndarray, dtype=uint32
        The quantized coordinates, each in range [0, 1023], same shape as the broadcasted input coordinates.
    """
    # Convert inputs to ndarray for consistent dtype/ufunc behavior.
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    # --- 1) Normalize each coordinate to [0, 1] over its bounding box. ---
    # Compute denominators once (avoid division by zero if bounds equal).
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin

    # Normalize to [0,1]; if a range is degenerate, map to 0 to avoid NaN/inf.
    with np.errstate(invalid="ignore"):
        xn = np.where(dx != 0, (x - xmin) / dx, 0.0)
        yn = np.where(dy != 0, (y - ymin) / dy, 0.0)
        zn = np.where(dz != 0, (z - zmin) / dz, 0.0)

    # --- 2) Quantize to (0..bitwidth). ---
    # Multiply by bitwidth, round down, and clip to be safe against overshoot.
    # Clip in float space before casting: out-of-range queries (e.g., points outside
    # a regional domain) would otherwise wrap around when a negative float is cast to uint32.
    # NaN queries produce arbitrary codes here; they are discarded downstream by the
    # finite-coordinate mask in SpatialHash.query.
    with np.errstate(invalid="ignore"):
        xq = np.clip(xn * bitwidth, 0, bitwidth).astype(np.uint32)
        yq = np.clip(yn * bitwidth, 0, bitwidth).astype(np.uint32)
        zq = np.clip(zn * bitwidth, 0, bitwidth).astype(np.uint32)

    return xq, yq, zq


def _encode_quantized_morton3d(xq, yq, zq):
    xq = np.asarray(xq)
    yq = np.asarray(yq)
    zq = np.asarray(zq)

    # --- 3) Bit-dilate each 10-bit number so each bit is separated by two zeros. ---
    # _dilate_bits maps:  b9..b0  ->  b9 0 0 b8 0 0 ... b0 0 0
    dx3 = _dilate_bits(xq).astype(np.uint32)
    dy3 = _dilate_bits(yq).astype(np.uint32)
    dz3 = _dilate_bits(zq).astype(np.uint32)

    # --- 4) Interleave the dilated bits into a single Morton code. ---
    # Bit layout (from LSB upward): x0,y0,z0, x1,y1,z1, ..., x9,y9,z9
    # We shift z's bits by 2, y's by 1, x stays at 0, then OR them together.
    # Cast to a wide type before shifting/OR to be safe when arrays are used.
    code = (dz3 << 2) | (dy3 << 1) | dx3

    # Since our compact type fits in 30 bits, uint32 is enough.
    return code.astype(np.uint32)


def _encode_morton3d(x, y, z, xmin, xmax, ymin, ymax, zmin, zmax, bitwidth=1023):
    """
    Quantize (x, y, z) to 10 bits each (0..1023), dilate the bits so there are
    two zeros between successive bits, and interleave them into a 3D Morton code.

    Parameters
    ----------
    x, y, z : array_like
        Input coordinates to encode. Can be scalars or arrays (broadcasting applies).
    xmin, xmax : float
        Minimum and maximum bounds for x coordinate.
    ymin, ymax : float
        Minimum and maximum bounds for y coordinate.
    zmin, zmax : float
        Minimum and maximum bounds for z coordinate.

    Returns
    -------
    code : ndarray, dtype=uint32
        The resulting Morton codes, same shape as the broadcasted input coordinates.

    Notes
    -----
    - Works with scalars or NumPy arrays (broadcasting applies).
    - Output is up to 30 bits returned as uint32.
    """
    # Convert inputs to ndarray for consistent dtype/ufunc behavior.
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    xq, yq, zq = quantize_coordinates(x, y, z, xmin, xmax, ymin, ymax, zmin, zmax, bitwidth)

    # --- 3) Bit-dilate each 10-bit number so each bit is separated by two zeros. ---
    # _dilate_bits maps:  b9..b0  ->  b9 0 0 b8 0 0 ... b0 0 0
    dx3 = _dilate_bits(xq).astype(np.uint32)
    dy3 = _dilate_bits(yq).astype(np.uint32)
    dz3 = _dilate_bits(zq).astype(np.uint32)

    # --- 4) Interleave the dilated bits into a single Morton code. ---
    # Bit layout (from LSB upward): x0,y0,z0, x1,y1,z1, ..., x9,y9,z9
    # We shift z's bits by 2, y's by 1, x stays at 0, then OR them together.
    # Cast to a wide type before shifting/OR to be safe when arrays are used.
    code = (dz3 << 2) | (dy3 << 1) | dx3

    # Since our compact type fits in 30 bits, uint32 is enough.
    return code.astype(np.uint32)


def _generate_valid_mask(xlow, xhigh, ylow, yhigh, zlow, zhigh):
    """
    Flag faces whose bounding box is fully defined, i.e. none of their 6 bounds
    is NaN (a NaN indicates a corner node with a missing/masked coordinate).

    Parameters
    ----------
    xlow, xhigh : array_like
        Per-face bounding box in x.
    ylow, yhigh : array_like
        Per-face bounding box in y.
    zlow, zhigh : array_like
        Per-face bounding box in z.

    Returns
    -------
    valid_face : ndarray of bool
        Same shape as the inputs; True where the face's bounding box is finite,
        False where it contains a NaN.
    """
    invalid_face = (
        np.isnan(xlow) | np.isnan(xhigh) | np.isnan(ylow) | np.isnan(yhigh) | np.isnan(zlow) | np.isnan(zhigh)
    )

    return ~invalid_face
