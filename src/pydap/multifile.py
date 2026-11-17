"""Cross-file (multi-granule) DAP4 dataset views.

This module implements :func:`open_mfdataset`, which presents a sequence of
DAP4 granule URLs as a single, lazily-evaluated :class:`~pydap.model.DatasetType`
view concatenated along one numeric coordinate dimension.

Design overview
---------------

* Opening a view downloads only DMR metadata, the concat-dimension coordinate
  of every granule, the non-concat coordinate variables, and static (no concat
  axis) variables needed for compatibility checks. Dynamic observation
  variables are never downloaded eagerly.
* Every dynamic variable in the view is backed by a
  :class:`MultiFileProxy` which translates a logical (merged-axis) index into
  one positive-step hyperslab request per granule run, then reassembles the
  pieces in the requested axis order.
* All DAP4 requests ask for checksums; the decoder verifies them
  (see :mod:`pydap.handlers.dap`), so truncated or corrupt responses never
  surface as half-populated arrays.
* Dynamic payloads are bounded by ``max_buffer_bytes``: only the payloads of
  one granule run at a time are held, plus bookkeeping indices and merged
  coordinates (which are explicitly not counted against the budget).
"""

from __future__ import annotations

import copy
import threading
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import numpy as np

from pydap.exceptions import BufferLimitExceededError
from pydap.handlers.dap import BaseProxyDap4
from pydap.lib import walk
from pydap.model import BaseType, DatasetType, GroupType
from pydap.net import create_session

# Attributes whose numeric meaning must match across granules.
_NUMERIC_MEANING_ATTRS = (
    "units",
    "calendar",
    "_FillValue",
    "scale_factor",
    "add_offset",
)
_SUPPORTED_SCHEMES = ("https", "dap4", "http")
_VALID_OVERLAP = ("error", "first", "last")


# ---------------------------------------------------------------------------
# Argument validation (offline, discoverable without any network access)
# ---------------------------------------------------------------------------


def _validate_open_arguments(
    urls, concat_dim, keep_variables, overlap, max_buffer_bytes
):
    if not isinstance(urls, (list, tuple)) or len(urls) == 0:
        raise TypeError("`urls` must be a non-empty sequence of DAP4 dataset URLs.")
    if not all(isinstance(url, str) for url in urls):
        raise TypeError("every element of `urls` must be a string.")
    normalized = [_normalize_url(url) for url in urls]
    if len(set(normalized)) != len(normalized):
        raise ValueError("`urls` must not contain duplicate URLs.")

    if not isinstance(concat_dim, str) or not concat_dim:
        raise TypeError("`concat_dim` must be a non-empty absolute variable path.")
    if not _is_absolute_path(concat_dim):
        raise ValueError(
            "`concat_dim` must be an absolute path starting with '/', got "
            f"{concat_dim!r}."
        )

    if not isinstance(keep_variables, (list, tuple)) or len(keep_variables) == 0:
        raise TypeError(
            "`keep_variables` must be a non-empty sequence of variable paths."
        )
    if not all(isinstance(path, str) for path in keep_variables):
        raise TypeError("every element of `keep_variables` must be a string.")
    if len(set(keep_variables)) != len(keep_variables):
        raise ValueError("`keep_variables` must not contain duplicate paths.")
    for path in keep_variables:
        if not _is_absolute_path(path):
            raise ValueError(
                "every kept variable path must be absolute (start with '/'), "
                f"got {path!r}."
            )

    if overlap not in _VALID_OVERLAP:
        raise ValueError(f"`overlap` must be one of {_VALID_OVERLAP}, got {overlap!r}.")

    # bool is a subclass of int: reject it explicitly per the API contract.
    if isinstance(max_buffer_bytes, bool) or not isinstance(max_buffer_bytes, int):
        raise TypeError("`max_buffer_bytes` must be a positive integer (not a bool).")
    if max_buffer_bytes <= 0:
        raise ValueError("`max_buffer_bytes` must be a positive integer.")
    return normalized


def _is_absolute_path(path):
    return path.startswith("/") and path.strip() != "/" and "//" not in path


def _normalize_url(url):
    """Validate an input URL and return the (possibly dap4:// -> https://) form."""
    parts = urlsplit(url)
    if parts.scheme not in _SUPPORTED_SCHEMES:
        raise ValueError(
            f"URL {url!r} has unsupported scheme {parts.scheme!r}; expected one of "
            "https://, dap4:// or (for local tests) http://."
        )
    if parts.query:
        raise ValueError(
            f"URL {url!r} carries a constraint expression; this version only accepts "
            "base dataset URLs without '?...'."
        )
    if parts.fragment:
        raise ValueError(f"URL {url!r} must not contain a URL fragment.")
    if not parts.netloc or not parts.path:
        raise ValueError(f"URL {url!r} is not a complete dataset URL.")
    if parts.scheme == "dap4":
        return "https://" + url[len("dap4://") :]
    return url


# ---------------------------------------------------------------------------
# Buffer accounting
# ---------------------------------------------------------------------------


class BufferBudget:
    """Tracks simultaneously-live dynamic payload bytes.

    Coordinates, index metadata and final result arrays are never registered
    here; only decoded dynamic observation payloads (and transient reorder /
    write buffers) count.
    """

    def __init__(self, limit):
        self.limit = int(limit)
        self._live = 0
        self.peak = 0
        self._lock = threading.RLock()

    def acquire(self, nbytes, context=""):
        nbytes = int(nbytes)
        with self._lock:
            if nbytes > self.limit:
                raise BufferLimitExceededError(
                    "A required observation slice of "
                    f"{nbytes} bytes exceeds max_buffer_bytes={self.limit}"
                    f" ({context}). Reduce the slice or increase the buffer budget."
                )
            if self._live + nbytes > self.limit:
                raise BufferLimitExceededError(
                    "Dynamic data buffer budget exceeded: need "
                    f"{self._live + nbytes} live bytes but limit is {self.limit}"
                    f" ({context})."
                )
            self._live += nbytes
            self.peak = max(self.peak, self._live)

    def release(self, nbytes):
        with self._lock:
            self._live -= int(nbytes)
            if self._live < 0:
                self._live = 0

    @property
    def live(self):
        return self._live


# ---------------------------------------------------------------------------
# Coordinate merging
# ---------------------------------------------------------------------------


@dataclass
class _GranulePlan:
    """Per-granule bookkeeping for a view."""

    index: int
    url: str
    dataset: object
    coord: np.ndarray  # granule-local concat coordinate, native dtype
    # merged-axis position -> granule-local position for coordinates selected
    # from this granule (after overlap resolution)
    merged_to_local: dict = field(default_factory=dict)
    # ordered list of merged-axis positions owned by this granule
    merged_positions: list = field(default_factory=list)


def _check_finite_coordinate(coord, url):
    if coord.ndim != 1:
        raise ValueError(
            f"Concat coordinate in {url!r} must be one-dimensional; got shape "
            f"{coord.shape}."
        )
    if coord.dtype.kind not in ("i", "u", "f") or coord.dtype not in (
        np.dtype("int64"),
        np.dtype("float64"),
    ):
        raise ValueError(
            f"Concat coordinate in {url!r} must be int64 or float64; got "
            f"{coord.dtype}."
        )
    if coord.size and not np.all(np.isfinite(coord)):
        raise ValueError(
            f"Concat coordinate in {url!r} contains non-finite values; all "
            "coordinate values must be finite."
        )


def _check_strict_monotonic(coord, url):
    if coord.size <= 1:
        return
    diffs = np.diff(
        coord.astype(np.float64 if coord.dtype.kind == "f" else coord.dtype)
    )
    # Exact numeric comparison; never coerce big integers to float first.
    if np.all(diffs > 0) or np.all(diffs < 0):
        return
    raise ValueError(
        f"Concat coordinate in {url!r} is not strictly monotonic (strictly "
        "increasing or strictly decreasing)."
    )


def _merge_coordinates(plans, overlap, url_for):
    """Merge granule coordinates into a single ascending coordinate array.

    Returns ``(merged_coord, plans)`` where each plan gains its
    ``merged_positions`` / ``merged_to_local`` mapping. Duplicate coordinates
    across granules are resolved according to ``overlap``.
    """
    # Build typed arrays of values / granule indices / local indices. Keeping
    # the native int64/float64 dtype means integers larger than 2**53 are
    # compared exactly and never coerced through float.
    total = int(sum(plan.coord.size for plan in plans))
    sortable = np.empty(total, dtype=plans[0].coord.dtype)
    granule_idx = np.empty(total, dtype=np.int64)
    local_idx = np.empty(total, dtype=np.int64)
    k = 0
    for plan in plans:
        n = plan.coord.size
        sortable[k : k + n] = plan.coord
        granule_idx[k : k + n] = plan.index
        local_idx[k : k + n] = np.arange(n)
        k += n
    # Stable sort by value; ties keep granule (URL) order.
    order = np.argsort(sortable, kind="stable")

    merged_values = []
    owners = []  # (granule index, local index) per merged position
    i = 0
    while i < total:
        value = sortable[order[i]]
        j = i + 1
        while j < total and sortable[order[j]] == value:
            j += 1
        group = order[i:j]
        if group.size > 1:
            owners_in_group = sorted(
                (int(granule_idx[h]), int(local_idx[h])) for h in group
            )
            distinct_granules = sorted({g for g, _ in owners_in_group})
            if overlap == "error":
                raise ValueError(
                    "Duplicate concat coordinate value "
                    f"{value!r} found across granules "
                    f"{[url_for(g) for g in distinct_granules]}; pass "
                    "overlap='first' or overlap='last' to resolve."
                )
            chosen_granule = distinct_granules[0 if overlap == "first" else -1]
            chosen_local = next(
                local for granule, local in owners_in_group if granule == chosen_granule
            )
            merged_values.append(value)
            owners.append((chosen_granule, chosen_local))
        else:
            h = int(group[0])
            merged_values.append(sortable[h])
            owners.append((int(granule_idx[h]), int(local_idx[h])))
        i = j

    merged_coord = np.asarray(merged_values, dtype=plans[0].coord.dtype)
    for merged_pos, (granule, local) in enumerate(owners):
        plans[granule].merged_positions.append(merged_pos)
        plans[granule].merged_to_local[merged_pos] = local
    return merged_coord, plans


# ---------------------------------------------------------------------------
# Metadata compatibility
# ---------------------------------------------------------------------------


def _attr_compatible(value_a, value_b):
    """Missing attributes only match missing attributes; NaNs compare equal."""
    if value_a is None or value_b is None:
        return value_a is None and value_b is None
    if isinstance(value_a, float) and isinstance(value_b, float):
        if np.isnan(value_a) and np.isnan(value_b):
            return True
    if isinstance(value_a, np.ndarray) or isinstance(value_b, np.ndarray):
        return _arrays_equal(np.asarray(value_a), np.asarray(value_b))
    return value_a == value_b


def _fail(msg):
    raise ValueError(msg)


def _find_variable(dataset, path):
    """Fetch a BaseType by absolute path; raise if absent."""
    try:
        var = dataset[path]
    except (KeyError, TypeError):
        var = None
    if not isinstance(var, BaseType):
        return None
    return var


def _validate_granule_metadata(plans, concat_dim, keep_variables, coord_attr_names):
    """Cross-granule metadata compatibility checks (raises ValueError)."""
    first = plans[0].dataset
    first_url = plans[0].url

    # 1. Concat coordinate dtype / units / calendar consistency.
    for plan in plans[1:]:
        if plan.dataset[concat_dim].dtype != first[concat_dim].dtype:
            _fail(
                f"Concat coordinate dtype mismatch: {first_url!r} has "
                f"{first[concat_dim].dtype}, {plan.url!r} has "
                f"{plan.dataset[concat_dim].dtype} at {concat_dim!r}."
            )
        for attr in coord_attr_names:
            a = first[concat_dim].attributes.get(attr)
            b = plan.dataset[concat_dim].attributes.get(attr)
            if not _attr_compatible(a, b):
                _fail(
                    f"Concat coordinate attribute {attr!r} at {concat_dim!r} "
                    f"differs between {first_url!r} ({a!r}) and {plan.url!r} "
                    f"({b!r})."
                )

    # 2. Kept variable checks.
    for path in keep_variables:
        ref = _find_variable(first, path)
        if ref is None:
            _fail(
                f"Kept variable {path!r} does not exist in first URL " f"{first_url!r}."
            )
        if ref.dtype.kind not in ("i", "u", "f"):
            _fail(
                f"Kept variable {path!r} must be a fixed-width numeric array; "
                f"got dtype {ref.dtype}."
            )
        ref_dims = list(ref.dims)
        concat_occurrences = ref_dims.count(concat_dim)
        if concat_occurrences > 1:
            _fail(
                f"Kept variable {path!r} uses the concat dimension "
                f"{concat_dim!r} more than once (axes {ref_dims})."
            )
        is_static = concat_occurrences == 0

        for plan in plans[1:]:
            url = plan.url
            var = _find_variable(plan.dataset, path)
            if var is None:
                _fail(f"Kept variable {path!r} is missing from granule {url!r}.")
            if var.dtype != ref.dtype:
                _fail(
                    f"dtype mismatch for {path!r}: {first_url!r} has "
                    f"{ref.dtype}, {url!r} has {var.dtype}."
                )
            # Axes identified by full dimension path.
            if set(var.dims) != set(ref_dims):
                _fail(
                    f"Dimension set mismatch for {path!r}: {first_url!r} uses "
                    f"{ref_dims}, {url!r} uses {list(var.dims)}."
                )
            for attr in _NUMERIC_MEANING_ATTRS:
                a = ref.attributes.get(attr)
                b = var.attributes.get(attr)
                if not _attr_compatible(a, b):
                    _fail(
                        f"Attribute {attr!r} of variable {path!r} differs "
                        f"between {first_url!r} ({a!r}) and {url!r} ({b!r})."
                    )
            # Non-concat axis lengths must match.
            for dim in ref_dims:
                if dim == concat_dim:
                    continue
                ref_len = _dimension_length(first, ref, dim)
                new_len = _dimension_length(plan.dataset, var, dim)
                if ref_len != new_len:
                    _fail(
                        f"Length of dimension {dim!r} used by {path!r} differs: "
                        f"{first_url!r} has {ref_len}, {url!r} has {new_len}."
                    )

        if is_static:
            _validate_static_variable(plans, path)
        else:
            _validate_non_concat_coordinates(plans, path, ref, concat_dim)


def _dimension_length(dataset, var, dim_path):
    """Resolve a fully-qualified dimension length within the right group."""
    # Dimensions are group scoped; walk the group chain of the variable.
    local_name = dim_path.strip("/").split("/")[-1]
    node = var.parent
    seen = set()
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        dims = getattr(node, "attributes", {}).get("dimensions")
        if dims and local_name in dims:
            return int(dims[local_name])
        if node is getattr(node, "dataset", None):
            # Reached the root dataset; its own .dimensions mapping is next.
            root_dims = getattr(node, "dimensions", None)
            if root_dims and local_name in root_dims:
                return int(root_dims[local_name])
            break
        node = getattr(node, "parent", None)
    raise ValueError(
        f"Could not resolve dimension {dim_path!r} for variable {var.id!r}."
    )


def _arrays_equal(a, b):
    """Exact array equality with NaN-aware comparison for floating dtypes."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind == "f" or b.dtype.kind == "f":
        return np.array_equal(a, b, equal_nan=True)
    return np.array_equal(a, b)


def _materialize(var):
    """Return a numpy array for a BaseType, forcing a fetch for remote proxies."""
    data = var.data
    if isinstance(data, np.ndarray):
        return data
    # Remote proxy (or any non-ndarray data object): explicit slicing triggers
    # the DAP request, including for 0-dim variables whose plain ``.data``
    # would hand back the proxy itself.
    return np.asarray(var[...].data)


def _validate_non_concat_coordinates(plans, path, ref_var, concat_dim):
    """Same-path non-concat coordinate values must match across granules."""
    ref_dims = [d for d in ref_var.dims if d != concat_dim]
    first = plans[0]
    ref_values = {}
    for dim in ref_dims:
        coord_var = _find_variable(first.dataset, dim)
        if coord_var is None:
            # Named dimension without a coordinate variable: length was
            # already checked; nothing value-wise to compare.
            continue
        ref_values[dim] = _materialize(coord_var)
    for plan in plans[1:]:
        for dim, ref_array in ref_values.items():
            other = _find_variable(plan.dataset, dim)
            if other is None:
                _fail(
                    f"Coordinate variable {dim!r} required by {path!r} is "
                    f"missing from {plan.url!r}."
                )
            other_array = _materialize(other)
            if not _arrays_equal(ref_array, other_array):
                _fail(
                    f"Coordinate values of {dim!r} (used by {path!r}) differ "
                    f"between {first.url!r} and {plan.url!r}; equal lengths do "
                    "not imply equal coordinates."
                )


def _validate_static_variable(plans, path):
    """Variables without the concat axis must be identical everywhere."""
    ref = _materialize(plans[0].dataset[path])
    for plan in plans[1:]:
        other = _materialize(plan.dataset[path])
        if not _arrays_equal(ref, other):
            _fail(
                f"Static variable {path!r} (no concat dimension) has different "
                f"values in {plans[0].url!r} and {plan.url!r}."
            )


# ---------------------------------------------------------------------------
# Proxy and view
# ---------------------------------------------------------------------------


class MultiFileProxy:
    """Numpy-array-like proxy for a variable spread across granules.

    The proxy owns no observation payload between calls; every ``__getitem__``
    resolves the logical index into per-granule positive-step hyperslabs,
    fetches them through the granule's own :class:`BaseProxyDap4`, reorders
    axes to the view convention and concatenates along the merged axis.
    """

    def __init__(
        self,
        variable_path,
        view_dims,
        view_shape,
        dtype,
        plans,
        concat_dim,
        budget,
        granule_axis_orders,
        view=None,
        checksums=True,
    ):
        self.id = variable_path
        self.dims = list(view_dims)
        self.shape = tuple(view_shape)
        self.dtype = np.dtype(dtype)
        self._plans = plans
        self._concat_dim = concat_dim
        self._budget = budget
        self._view = view
        # granule index -> list of dimension paths in that granule's order
        self._granule_axis_orders = granule_axis_orders
        self.checksums = checksums
        self.ce = None

    def __len__(self):
        return self.shape[0]

    def __iter__(self):
        return iter(np.asarray(self[:]))

    # -- index handling ---------------------------------------------------
    def _normalize_key(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        if any(k is None for k in key):
            raise IndexError("newaxis indexing is not supported by multi-file views.")
        if any(isinstance(k, np.ndarray) or isinstance(k, list) for k in key):
            raise IndexError(
                "Advanced/array indexing is not supported; use integers or slices."
            )
        if Ellipsis in key:
            expand = len(self.shape) - (len(key) - 1)
            expanded = []
            for k in key:
                if k is Ellipsis:
                    expanded.extend([slice(None)] * expand)
                else:
                    expanded.append(k)
            key = tuple(expanded)
        if len(key) < len(self.shape):
            key = key + (slice(None),) * (len(self.shape) - len(key))
        if len(key) > len(self.shape):
            raise IndexError(
                f"too many indices for variable {self.id!r}: {len(key)} > "
                f"{len(self.shape)}."
            )
        normalized = []
        for k, size in zip(key, self.shape):
            if isinstance(k, (bool, np.bool_)):
                raise IndexError("boolean scalar indexing is not supported.")
            if isinstance(k, (int, np.integer)):
                k = int(k)
                if k < 0:
                    k += size
                if not 0 <= k < size:
                    raise IndexError(
                        f"index {k} out of bounds for axis of size {size} "
                        f"({self.id!r})."
                    )
                normalized.append(k)
            elif isinstance(k, slice):
                if k.step == 0:
                    raise ValueError("slice step cannot be zero.")
                normalized.append(k)
            else:
                raise IndexError(
                    f"unsupported index type {type(k).__name__} for {self.id!r}."
                )
        return tuple(normalized)

    def _axis_indices(self, key):
        """Return per-axis explicit index arrays and which axes are integer."""
        indices = []
        integer_axes = set()
        for axis, (k, size) in enumerate(zip(key, self.shape)):
            if isinstance(k, int):
                integer_axes.add(axis)
                indices.append(np.array([k], dtype=np.int64))
            else:
                start, stop, step = k.indices(size)
                indices.append(np.arange(start, stop, step, dtype=np.int64))
        return indices, integer_axes

    def __getitem__(self, key):
        if self._view is not None:
            self._view._ensure_open()
        key = self._normalize_key(key)
        indices, integer_axes = self._axis_indices(key)
        concat_axis = self.dims.index(self._concat_dim)
        wanted_concat = indices[concat_axis]
        if wanted_concat.size == 0:
            out_shape = tuple(
                0 if axis == concat_axis else idx.size
                for axis, idx in enumerate(indices)
                if axis not in integer_axes
            )
            return np.empty(out_shape, dtype=self.dtype)

        out = np.empty(tuple(idx.size for idx in indices), dtype=self.dtype)
        placements = self._group_by_granule(wanted_concat)
        reservations = []
        # Map each requested merged-axis position to its row in the output.
        out_row = {int(merged): row for row, merged in enumerate(wanted_concat)}
        try:
            for plan, pairs in placements:
                local_positions = np.array(
                    [local for _, local in pairs], dtype=np.int64
                )
                out_positions = np.array(
                    [out_row[int(merged)] for merged, _ in pairs], dtype=np.int64
                )
                before = len(reservations)
                piece = self._fetch_granule_piece(
                    plan, indices, concat_axis, local_positions, reservations
                )
                # Place along the concat axis; every other axis carries the
                # full logical selection, so use plain slices there.
                placement = [slice(None)] * out.ndim
                placement[concat_axis] = out_positions
                out[tuple(placement)] = piece
                # The payload has been copied into the result array (which is
                # not budgeted); release this granule's live payload so that
                # only one granule's observation payload counts at a time.
                for nbytes in reservations[before:]:
                    self._budget.release(nbytes)
                del reservations[before:]
        finally:
            for nbytes in reservations:
                self._budget.release(nbytes)
        # Drop axes indexed with an integer, matching numpy semantics.
        if integer_axes:
            out = np.squeeze(out, axis=tuple(sorted(integer_axes)))
        return out

    def _group_by_granule(self, wanted_concat):
        groups = []
        current = None
        for merged_pos in wanted_concat.tolist():
            owner = next(
                plan for plan in self._plans if merged_pos in plan.merged_to_local
            )
            local = owner.merged_to_local[merged_pos]
            if current is None or current[0] is not owner:
                current = (owner, [])
                groups.append(current)
            current[1].append((merged_pos, local))
        return groups

    def _fetch_granule_piece(
        self, plan, view_indices, concat_axis, local_positions, reservations
    ):
        granule = plan.dataset
        var = granule[self.id]
        granule_dims = list(self._granule_axis_orders[plan.index])

        # Build granule-local axis indices in the *granule's* axis order.
        granule_axis_index = {
            dim: view_indices[axis] for axis, dim in enumerate(self.dims)
        }
        granule_keys = []
        for dim in granule_dims:
            if dim == self._concat_dim:
                granule_keys.append(local_positions)
            else:
                granule_keys.append(granule_axis_index[dim])

        # Coalesce the concat positions into positive-step runs and fetch each
        # run separately, keeping live payloads bounded.
        runs = _runs(local_positions)
        pieces = []
        for run_start, run_stop in runs:
            run_local = np.arange(run_start, run_stop, dtype=np.int64)
            keys = list(granule_keys)
            concat_local_axis = granule_dims.index(self._concat_dim)
            keys[concat_local_axis] = run_local
            piece = self._fetch_single_run(plan, var, granule_dims, keys, reservations)
            pieces.append(piece)
        granule_array = np.concatenate(pieces, axis=concat_local_axis)

        # Reorder granule axes back into view axis order (data follows axes).
        view_order = [granule_dims.index(dim) for dim in self.dims]
        granule_array = np.transpose(granule_array, view_order)
        # The piece covers only this run's concat positions; shrink the concat
        # axis to match before handing it back for placement.
        expected_concat = int(local_positions.size)
        if granule_array.shape[concat_axis] != expected_concat:
            raise RuntimeError(
                f"Concat-axis size mismatch while assembling {self.id!r} from "
                f"{plan.url!r}: got {granule_array.shape[concat_axis]}, "
                f"expected {expected_concat}."
            )
        return granule_array

    def _fetch_single_run(self, plan, var, granule_dims, keys, reservations):
        # Translate explicit index arrays to positive-step slices where
        # possible; fetch an enclosing span for non-contiguous axes and index
        # locally afterwards.
        fetch_slices = []
        local_index = []
        needs_local_index = False
        for idx in keys:
            slc = _as_positive_slice(idx)
            if slc is not None:
                fetch_slices.append(slc)
                local_index.append(None)
            else:
                needs_local_index = True
                span = slice(int(idx.min()), int(idx.max()) + 1, 1)
                fetch_slices.append(span)
                local_index.append(idx - idx.min())
        data = self._request(var, plan, tuple(fetch_slices), reservations)
        if needs_local_index:
            local_key = tuple(
                idx if idx is not None else slice(None) for idx in local_index
            )
            data = data[local_key]
        return data

    def _request(self, var, plan, granule_slices, reservations):
        # Budget reservation based on the requested payload size.
        requested_shape = tuple(
            _slice_length(slc, size) for slc, size in zip(granule_slices, var.shape)
        )
        nbytes = int(np.prod(requested_shape)) * self.dtype.itemsize
        self._budget.acquire(nbytes, context=f"{self.id} from {plan.url}")
        reservations.append(nbytes)
        proxy = var.data
        if not isinstance(proxy, BaseProxyDap4):
            # Defensive: open_url always installs BaseProxyDap4 for DAP4.
            raise RuntimeError(
                f"Variable {self.id!r} in {plan.url!r} is not backed by a "
                "DAP4 remote proxy."
            )
        data = np.asarray(proxy[tuple(granule_slices)])
        if data.shape != requested_shape:
            raise RuntimeError(
                f"Unexpected shape for {self.id!r} from {plan.url!r}: got "
                f"{data.shape}, expected {requested_shape}."
            )
        return data


def _slice_length(slc, size):
    start, stop, step = slc.indices(size)
    return len(range(start, stop, step))


def _as_positive_slice(idx):
    if idx.size == 0:
        return slice(0, 0, 1)
    if idx.size == 1:
        return slice(int(idx[0]), int(idx[0]) + 1, 1)
    diffs = np.diff(idx)
    if np.all(diffs == 1):
        return slice(int(idx[0]), int(idx[-1]) + 1, 1)
    if np.all(diffs == diffs[0]) and diffs[0] > 0:
        return slice(int(idx[0]), int(idx[-1]) + 1, int(diffs[0]))
    return None


def _runs(positions):
    """Collapse an ascending int64 array into inclusive (start, stop) runs."""
    if positions.size == 0:
        return []
    runs = []
    start = prev = int(positions[0])
    for value in positions[1:].tolist():
        if value == prev + 1:
            prev = value
            continue
        runs.append((start, prev + 1))
        start = prev = value
    runs.append((start, prev + 1))
    return runs


class MultiFileDataset(DatasetType):
    """A closeable, context-manager compatible multi-granule dataset view."""

    def __init__(self, name, session, owns_session, budget):
        super().__init__(name, session=session)
        self._mf_owns_session = owns_session
        self._mf_budget = budget
        self._mf_closed = False
        self._mf_lock = threading.RLock()
        self._mf_plans = []
        self._mf_concat_dim = None
        self._mf_keep_variables = None
        self._merged_coord = None

    # -- lifecycle --------------------------------------------------------
    def close(self):
        with self._mf_lock:
            if self._mf_closed:
                return
            self._mf_closed = True
            if self._mf_owns_session and self._session is not None:
                self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _ensure_open(self):
        if self._mf_closed:
            raise RuntimeError(
                "This multi-file dataset view has been closed; open a new "
                "view to read data."
            )

    @property
    def buffer_budget(self):
        return self._mf_budget

    def write_merged_netcdf(self, output_file):
        # Local import to keep netCDF4 optional for the rest of the package.
        from pydap.netcdf_writer import write_view_to_netcdf

        self._ensure_open()
        return write_view_to_netcdf(self, output_file)


# ---------------------------------------------------------------------------
# View construction
# ---------------------------------------------------------------------------


def _build_view_dataset(plans, merged_coord, concat_dim, keep_variables, budget):
    first = plans[0].dataset
    view = MultiFileDataset(
        name=first.name,
        session=first.session,
        owns_session=False,
        budget=budget,
    )
    view._mf_plans = plans
    view._mf_concat_dim = concat_dim
    view._mf_keep_variables = list(keep_variables)
    view._merged_coord = merged_coord

    # Replicate the group skeleton / dimensions of the first granule, with
    # the concat dimension length replaced by the merged length.
    _copy_skeleton(first, view, concat_dim, merged_coord.size)

    # Add the coordinate variables and kept variables.
    needed_paths = set(keep_variables)
    # Include coordinate variables backing every dimension of kept vars.
    for path in keep_variables:
        ref = first[path]
        for dim in ref.dims:
            needed_paths.add(dim)

    for path in sorted(needed_paths):
        ref = first[path]
        if not isinstance(ref, BaseType):
            continue
        leaf_name = ref.name
        if path == concat_dim:
            data = merged_coord
            var = BaseType(
                leaf_name,
                data=data,
                dims=list(ref.dims),
                attributes=copy.deepcopy(ref.attributes),
            )
        elif concat_dim in ref.dims:
            # Dynamic variable: proxy only.
            shape = list(ref.shape)
            axis = ref.dims.index(concat_dim)
            shape[axis] = merged_coord.size
            granule_orders = {
                plan.index: list(plan.dataset[path].dims) for plan in plans
            }
            proxy = MultiFileProxy(
                path,
                view_dims=ref.dims,
                view_shape=tuple(shape),
                dtype=ref.dtype,
                plans=plans,
                concat_dim=concat_dim,
                budget=budget,
                granule_axis_orders=granule_orders,
                view=view,
            )
            var = BaseType(
                leaf_name,
                data=proxy,
                dims=list(ref.dims),
                attributes=copy.deepcopy(ref.attributes),
            )
        else:
            # Static / non-concat coordinate variable: single copy from the
            # first granule (values were verified identical).
            data = _materialize(ref)
            var = BaseType(
                leaf_name,
                data=data,
                dims=list(ref.dims),
                attributes=copy.deepcopy(ref.attributes),
            )
        view[path.lstrip("/")] = var
    view.assign_dataset_recursive(view)
    return view


def _copy_skeleton(src_dataset, view, concat_dim, merged_length):
    """Create groups/dimensions/attributes on the view from the first granule.

    Only the structural metadata is copied. Variables are attached separately
    by the caller. Concat-dimension sizes are replaced with the merged length
    at every group scope where the dimension path appears.
    """
    view.attributes = copy.deepcopy(src_dataset.attributes)

    # Root-level named dimensions (parser stores them on dataset.dimensions).
    root_dims = dict(getattr(src_dataset, "dimensions", {}))
    concat_local = concat_dim.strip("/").split("/")[-1]
    if concat_dim.count("/") == 1 and concat_local in root_dims:
        root_dims[concat_local] = merged_length
    view.dimensions = root_dims

    # Recreate the group hierarchy with group attributes and scoped dimensions.
    for group in walk(src_dataset, GroupType):
        group_path = group.id  # e.g. '/g' or '/g/sub'
        if group_path == "/" or not group_path:
            continue
        attrs = copy.deepcopy(group.attributes)
        dims = dict(attrs.get("dimensions", {}))
        local_concat = concat_dim.strip("/").split("/")[-1]
        if concat_dim.startswith(group_path.rstrip("/") + "/") and local_concat in dims:
            dims[local_concat] = merged_length
        attrs["dimensions"] = dims
        view.createGroup(group_path, attributes=attrs)


def open_mfdataset(
    urls,
    *,
    concat_dim,
    keep_variables,
    overlap="error",
    session=None,
    max_buffer_bytes=8388608,
):
    """Open a lazy, concatenated view over multiple DAP4 granule URLs.

    Parameters
    ----------
    urls : sequence of str
        Non-empty, duplicate-free DAP4 dataset URLs (``https://``,
        ``dap4://`` or, for local test servers, ``http://``). Constraint
        expressions in the URL are not accepted.
    concat_dim : str
        Absolute path of the one-dimensional numeric coordinate variable
        along which granules are concatenated (e.g. ``"/time"``).
    keep_variables : sequence of str
        Non-empty, duplicate-free absolute paths of fixed-width numeric
        variables to expose in the view.
    overlap : {"error", "first", "last"}, default "error"
        How to resolve coordinate values present in more than one granule.
    session : requests.Session, optional
        Caller-owned session. When provided, pydap never closes it. When
        omitted, a session is created and owned by the view (closed on
        ``view.close()``).
    max_buffer_bytes : int, default 8388608
        Maximum simultaneously-live dynamic observation payload, in bytes.
        Coordinates and index metadata do not count toward this budget.

    Returns
    -------
    MultiFileDataset
        A :class:`~pydap.model.DatasetType`-compatible, closeable view.
    """
    from pydap.client import open_url

    normalized = _validate_open_arguments(
        urls, concat_dim, keep_variables, overlap, max_buffer_bytes
    )
    owns_session = session is None
    if owns_session:
        session = create_session()
    budget = BufferBudget(max_buffer_bytes)

    plans = []
    view = None
    try:
        for index, url in enumerate(normalized):
            dataset = open_url(
                url,
                session=session,
                protocol="dap4",
                checksums=True,
                batch=False,
            )
            coord_var = _find_variable(dataset, concat_dim)
            if coord_var is None:
                raise ValueError(
                    f"Concat coordinate variable {concat_dim!r} not found in "
                    f"{url!r}."
                )
            coord = _materialize(coord_var)
            _check_finite_coordinate(coord, url)
            _check_strict_monotonic(coord, url)
            plans.append(
                _GranulePlan(index=index, url=url, dataset=dataset, coord=coord)
            )

        coord_attr_names = ("units", "calendar")
        # Metadata + static value compatibility (may fetch static/coords only).
        _validate_granule_metadata(plans, concat_dim, keep_variables, coord_attr_names)
        merged_coord, plans = _merge_coordinates(
            plans, overlap, url_for=lambda i: plans[i].url
        )
        view = _build_view_dataset(
            plans, merged_coord, concat_dim, keep_variables, budget
        )
        view._mf_owns_session = owns_session
        return view
    except BaseException:
        if owns_session:
            session.close()
        raise
