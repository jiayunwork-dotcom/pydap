"""Streaming NetCDF output for multi-granule :class:`MultiFileDataset` views.

The writer never materializes a dynamic (concat-axis) variable: it walks the
merged axis in blocks sized to fit the view's dynamic-data buffer budget and
streams each block through the existing :class:`~pydap.multifile.MultiFileProxy`
machinery (remote hyperslab -> DAP4 decoder with checksum verification ->
axis reorder).

Output is written to a temporary file in the same directory and renamed into
place only after a fully successful write, so a failure (HTTP, checksum,
truncation, local I/O) leaves any pre-existing target file byte-for-byte
unchanged.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

from pydap.lib import walk
from pydap.model import BaseType, GroupType

# Attributes that are pydap/DAP bookkeeping rather than scientific metadata.
_SKIP_ATTRS = {"Maps", "path", "_DAP4_Checksum_CRC32", "_FillValue"}
_FILL_ALIASES = {"missing_value", "fmissing_value"}


def _split_group(path):
    """'/g/t' -> ('/g', 't'); '/t' -> ('/', 't')."""
    parts = path.strip("/").split("/")
    if len(parts) == 1:
        return "/", parts[0]
    return "/" + "/".join(parts[:-1]), parts[-1]


def _nc_group(nc, group_path):
    if group_path == "/":
        return nc
    node = nc
    for part in group_path.strip("/").split("/"):
        if part not in node.groups:
            node = node.createGroup(part)
        else:
            node = node.groups[part]
    return node


def _set_attributes(ncvar, attributes):
    for key, value in attributes.items():
        if key in _SKIP_ATTRS or key in _FILL_ALIASES:
            continue
        ncvar.setncattr(key, value)


def write_view_to_netcdf(view, output_file):
    """Write a multi-file view to a single merged NetCDF4 file.

    Parameters
    ----------
    view : MultiFileDataset
        The open multi-granule view.
    output_file : str or pathlib.Path
        Destination file. An existing file is replaced atomically only on
        success; failures leave it untouched.

    Returns
    -------
    str
        The resolved output path.
    """
    from netCDF4 import Dataset

    view._ensure_open()
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Stage in the destination directory so the final rename is atomic on the
    # same filesystem.
    fd, tmp_name = tempfile.mkstemp(
        prefix=output_file.name + ".", suffix=".tmp", dir=output_file.parent
    )
    os.close(fd)
    nc = None
    try:
        nc = Dataset(tmp_name, "w", format="NETCDF4")
        budget = view.buffer_budget

        # 1. Dimensions, per group scope.
        root_dims = dict(getattr(view, "dimensions", {}))
        for name, size in root_dims.items():
            nc.createDimension(name, int(size))
        for group in walk(view, GroupType):
            target = _nc_group(nc, group.id)
            for name, size in group.attributes.get("dimensions", {}).items():
                if name not in target.dimensions:
                    target.createDimension(name, int(size))

        # 2. Global / group attributes.
        for key, value in view.attributes.items():
            if key != "dimensions":
                nc.setncattr(key, value)
        for group in walk(view, GroupType):
            target = _nc_group(nc, group.id)
            for key, value in group.attributes.items():
                if key not in ("dimensions", "path", "Maps"):
                    target.setncattr(key, value)

        # 3. Variables.
        dynamic_vars = []
        for var in walk(view, BaseType):
            group_path, leaf = _split_group(var.id)
            target = _nc_group(nc, group_path)
            dim_names = [d.strip("/").split("/")[-1] for d in var.dims]
            fill = var.attributes.get("_FillValue")
            ncvar = target.createVariable(
                leaf,
                np.dtype(var.dtype).newbyteorder("="),
                dimensions=dim_names,
                fill_value=fill,
            )
            _set_attributes(ncvar, var.attributes)
            from pydap.multifile import MultiFileProxy

            if isinstance(var.data, MultiFileProxy):
                dynamic_vars.append((var, ncvar))
            else:
                # Coordinates, static variables: already materialized locally.
                ncvar[...] = np.asarray(var.data)

        # 4. Stream dynamic variables in budget-sized blocks along the merged
        #    concat axis. Only one block's payload is live at a time.
        for var, ncvar in dynamic_vars:
            concat_axis = var.dims.index(view._mf_concat_dim)
            merged_length = int(var.shape[concat_axis])
            itemsize = int(np.dtype(var.dtype).itemsize)
            # Other axes contribute a fixed footprint per concat row.
            row_elems = int(
                np.prod(
                    [s for ax, s in enumerate(var.shape) if ax != concat_axis],
                    dtype=np.int64,
                )
            )
            row_bytes = max(1, row_elems * itemsize)
            block_rows = max(1, budget.limit // row_bytes)
            if block_rows * row_bytes > budget.limit:
                # A single row already fits (guaranteed by the budget check on
                # reads); keep block_rows at one when rows alone fill budget.
                block_rows = 1
            for start in range(0, merged_length, block_rows):
                stop = min(merged_length, start + block_rows)
                key = [slice(None)] * var.ndim
                key[concat_axis] = slice(start, stop, 1)
                block = np.asarray(var.data[tuple(key)])
                ncvar[tuple(key)] = block
                del block

        nc.sync()
        nc.close()
        nc = None
        os.replace(tmp_name, output_file)
        return str(output_file)
    except BaseException:
        if nc is not None:
            try:
                nc.close()
            except Exception:
                pass
        if os.path.exists(tmp_name):
            try:
                os.remove(tmp_name)
            except OSError:
                pass
        raise
