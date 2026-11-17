# Multi-file DAP4 views (`open_mfdataset`)

Observations are commonly distributed across many granule files that share one
time (or other concatenation) axis. `pydap.client.open_mfdataset` presents
such a collection as a single, lazily-evaluated dataset view along that axis.
Only metadata, the concatenation coordinate and the variables needed for
compatibility checks are read while opening; dynamic observation variables are
downloaded on demand as constrained DAP4 hyperslabs, so taking a small slice
never pulls whole granules. No xarray or other concatenation dependency is
required.

## Opening a view

```python
from pydap.client import open_mfdataset

urls = [
    "https://example.org/data/granule_001.nc",
    "https://example.org/data/granule_002.nc",
    "https://example.org/data/granule_003.nc",
]

view = open_mfdataset(
    urls,
    concat_dim="/time",                 # absolute path of the concat coordinate
    keep_variables=["/temperature", "/lon", "/lat"],
    overlap="error",                    # 'error' | 'first' | 'last'
    max_buffer_bytes=8 * 1024 * 1024,   # live dynamic-payload budget
)
```

`dap4://` URLs are accepted and translated to `https://`; plain `http://` is
allowed for local test servers. URLs must be base dataset URLs without a
`?dap4.ce=...` constraint and must be unique.

The returned object is a `DatasetType`-compatible view that supports the
context-manager protocol and explicit closing:

```python
with open_mfdataset(urls, concat_dim="/time",
                    keep_variables=["/temperature"]) as view:
    ...
# or
view.close()
```

A session passed in by the caller is never closed by the library; when no
session is supplied, pydap creates one and closes it with the view.

## Coordinate and overlap semantics

* Every granule must expose a 1-D, finite `int64` or `float64` coordinate
  variable at the same path as the concat dimension, strictly monotonic
  within the granule (ascending or descending; empty coordinates are
  allowed). `dtype`, `units` and `calendar` must match across granules, and a
  missing attribute is compatible only with the same attribute missing.
* The merged coordinate is always ascending. Integer coordinates larger than
  `2**53` stay distinguishable: comparisons use the native integer dtype and
  never go through floating point first.
* A coordinate value appearing in more than one granule is an error with
  `overlap="error"`. With `"first"` or `"last"`, the granule that appears
  first (or last) in `urls` owns that coordinate. Ownership is decided once
  per coordinate and determines the whole row of **every** kept variable;
  sources are never mixed per variable. Equality is exact numeric equality
  with no tolerance.

## Accessing variables and slicing

Variables are reached by absolute path and expose the usual `dtype`,
`shape` and `dims`. Axis order follows the first URL even when later granules
store the same variable with its axes permuted; the data are reordered to
match. Slicing uses normal NumPy semantics on the merged logical axes:

```python
temp = view["/temperature"]
print(temp.dtype, temp.shape, temp.dims)

piece = temp[0:100:2, 10:20, :].data      # constrained remote hyperslabs
scalar = temp[-1, 0, 0].data             # negative indexing
reversed_ = temp[::-1].data              # negative step
empty = temp[0:0].data                   # empty slice, no request issued
```

Integer indices, ordinary slices, positive and negative steps, negative
indices and empty slices are supported. A zero step raises `ValueError`;
advanced/array indexing is not supported in this version. Requests are issued
per granule run, and DAP4 CRC32 checksums are verified on every response, so
truncated or corrupt data raise instead of returning a half-populated array.

Static variables (those without the concat axis) and non-concat coordinate
variables are checked for identical values across granules at open time and
stored once. Variable `dtype`, dimension sets, `units`, `calendar`,
`_FillValue`, `scale_factor`, `add_offset`, non-concat dimension lengths and
coordinate values must be compatible; incompatibilities raise `ValueError`
naming the URL and variable/dimension path. Remaining variable, group and
global attributes are taken from the first URL without automatic merging.
Missing-value markers and the other raw numeric semantics are preserved
without CF decoding.

## Exporting one merged NetCDF file

```python
from pydap.client import to_netcdf

to_netcdf(view, output_file="merged.nc4")
# equivalently:
view.write_merged_netcdf("merged.nc4")
```

Dynamic variables are streamed in blocks sized against
`max_buffer_bytes`, so the live payload stays bounded regardless of the total
observation count; coordinates and index metadata are not counted against that
budget. The output is written to a temporary file in the destination
directory and renamed into place only after a fully successful write. If an
HTTP error, truncation, checksum failure or local write error occurs, an
existing destination file is left byte-for-byte unchanged and every temporary
resource is cleaned up.

The previous single-URL and multi-URL `to_netcdf(...)` call styles continue to
write one file per granule as before.

## Buffer accounting

If a single necessary observation slice is itself larger than
`max_buffer_bytes`, a `pydap.exceptions.BufferLimitExceededError` is raised.
The final array returned to a normal `.data` call is not counted against the
budget; transient decoded, reordered and write buffers are. A view's
`view.buffer_budget.peak` reports the largest simultaneously-live dynamic
payload observed, which is useful when tuning block sizes.

## Design notes and limitations

* The logical index is split into per-granule contiguous runs; each run
  becomes one positive-step DAP4 hyperslab (negative steps are resolved
  locally after fetching). This keeps request fragmentation proportional to
  the number of granules a slice actually touches, at the cost of fetching
  the enclosing span when a non-concat axis uses a non-contiguous stride.
* Granule content is assumed immutable for the lifetime of a view; remote
  version changes during a session are not handled in this version.
* Only fixed-width numeric arrays are supported as kept variables; strings,
  sequences and structures are out of scope.
