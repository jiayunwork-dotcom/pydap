"""A controllable, in-process DAP4 HTTP test server (protocol double).

This module is only used by the pydap test-suite. It implements enough of the
DAP4 over HTTP protocol to exercise the client end to end:

* ``<url>.dmr`` returns a DMR document built from in-memory granule
  definitions.
* ``<url>.dap?dap4.ce=...&dap4.checksum=true`` returns a chunked DAP4 binary
  response, honoring positive-step hyperslab constraints and embedding
  little-endian CRC32 per-variable checksums.

The server keeps a full log of every request, including the parsed
constraint expression, so tests can assert on laziness and slicing behavior.
Fault injection hooks allow truncating responses, corrupting checksums and
failing individual HTTP requests.
"""

from __future__ import annotations

import re
import threading
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote
from xml.sax.saxutils import escape

import numpy as np

from pydap.server.devel import get_open_port

_NUMPY_TO_DAP4 = {
    "int8": "Int8",
    "uint8": "Byte",
    "int16": "Int16",
    "uint16": "UInt16",
    "int32": "Int32",
    "uint32": "UInt32",
    "int64": "Int64",
    "uint64": "UInt64",
    "float32": "Float32",
    "float64": "Float64",
}


def _dap4_type(dtype: np.dtype) -> str:
    name = np.dtype(dtype).name
    if name.startswith(("str", "bytes", "object")):
        # The protocol double only models fixed-width numeric granules, which
        # is all that open_mfdataset accepts.
        raise TypeError(
            f"dap4_test_server only serves fixed-width numeric arrays, got {dtype}"
        )
    return _NUMPY_TO_DAP4[name]


def _attr_xml(name, value, indent):
    """Render a scalar / 1-D numeric-or-string attribute as DMR XML."""
    lines = []
    if isinstance(value, (list, tuple, np.ndarray)):
        values = list(value)
        dap_type = _dap4_type(np.asarray(values).dtype) if values else "String"
        lines.append(f'{indent}<Attribute name="{escape(name)}" type="{dap_type}">')
        for val in values:
            lines.append(f"{indent}    <Value>{_format_attr_value(val)}</Value>")
        lines.append(f"{indent}</Attribute>")
    else:
        if isinstance(value, bool):
            dap_type = "Byte"
        elif isinstance(value, (int, np.integer)):
            dap_type = "Int64"
        elif isinstance(value, (float, np.floating)):
            dap_type = "Float64"
        else:
            dap_type = "String"
        lines.append(f'{indent}<Attribute name="{escape(name)}" type="{dap_type}">')
        lines.append(f"{indent}    <Value>{_format_attr_value(value)}</Value>")
        lines.append(f"{indent}</Attribute>")
    return lines


def _format_attr_value(value):
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        if np.isinf(value):
            return "inf" if value > 0 else "-inf"
    return escape(str(value))


class Granule:
    """An in-memory DAP4 granule.

    Parameters
    ----------
    name : str
        Dataset name used in the DMR and as part of the URL.
    dimensions : dict
        Mapping of fully qualified dimension path (e.g. ``/time`` or
        ``/g/x``) to its integer length.
    variables : dict
        Mapping of fully qualified variable path to a ``(array, dims,
        attributes)`` triple where ``dims`` is a tuple of fully qualified
        dimension paths in the axis order used by this granule and
        ``attributes`` is a plain dict (may be empty).
    attributes : dict
        Global / group attributes keyed by group path (``"/`` for global).
    """

    def __init__(self, name, dimensions, variables, attributes=None):
        self.name = name
        self.dimensions = dict(dimensions)
        self.variables = dict(variables)
        self.attributes = attributes or {"/": {}}

    def dims_for(self, path):
        return self.variables[path][1]

    def array_for(self, path):
        return self.variables[path][0]

    def attrs_for(self, path):
        return self.variables[path][2]


def build_dmr(granule: Granule) -> str:
    """Build a DMR document for a granule."""
    # Group layout derived from variable / dimension paths.
    group_paths = {"/"}
    for path in list(granule.dimensions) + list(granule.variables):
        parts = path.strip("/").split("/")
        for i in range(1, len(parts)):
            group_paths.add("/" + "/".join(parts[:i]))
    group_paths = sorted(group_paths, key=lambda p: (p.count("/"), p))

    lines = [
        '<?xml version="1.0" encoding="ISO-8859-1"?>',
        '<Dataset xmlns="http://xml.opendap.org/ns/DAP/4.0#" '
        'dmrVersion="1.0" name="%s">' % escape(granule.name),
    ]

    def emit_group(group_path, indent_level):
        indent = "    " * indent_level
        attrs = dict(granule.attributes.get(group_path, {}))
        if group_path != "/":
            gname = group_path.strip("/").split("/")[-1]
            lines.append(f'{indent}<Group name="{escape(gname)}">')
        # Dimensions declared at this level.
        for dim_path, size in granule.dimensions.items():
            parent = "/" + "/".join(dim_path.strip("/").split("/")[:-1])
            if parent == group_path or (group_path == "/" and dim_path.count("/") == 1):
                dname = dim_path.strip("/").split("/")[-1]
                lines.append(
                    f"{indent}    <Dimension name="
                    f'"{escape(dname)}" size="{int(size)}"/>'
                )
        # Variables declared at this level.
        for var_path, (array, dims, var_attrs) in granule.variables.items():
            parent = "/" + "/".join(var_path.strip("/").split("/")[:-1])
            if parent == group_path or (group_path == "/" and var_path.count("/") == 1):
                vname = var_path.strip("/").split("/")[-1]
                lines.append(
                    f"{indent}    <{_dap4_type(array.dtype)} "
                    f'name="{escape(vname)}">'
                )
                for dim in dims:
                    lines.append(f'{indent}        <Dim name="{escape(dim)}"/>')
                for key, value in var_attrs.items():
                    lines.extend(_attr_xml(key, value, indent + "        "))
                lines.append(f"{indent}    </{_dap4_type(array.dtype)}>")
        # Attributes attached to this group/dataset.
        for key, value in attrs.items():
            lines.extend(_attr_xml(key, value, indent + "    "))
        # Recurse into child groups.
        children = sorted(
            p
            for p in group_paths
            if p != "/"
            and "/" + "/".join(p.strip("/").split("/")[:-1]) == group_path
            and (group_path != "/" or True)
        )
        for child in children:
            if group_path == "/" or child.startswith(group_path.rstrip("/") + "/"):
                emit_group(child, indent_level + 1)
        if group_path != "/":
            lines.append(f"{indent}</Group>")

    emit_group("/", 0)
    lines.append("</Dataset>")
    return "\n".join(lines) + "\n"


_HYPERSLAB_RE = re.compile(r"\[(-?\d+):(-?\d+):(-?\d+)\]")


def parse_ce(ce: str, granule=None):
    """Parse a positive-step DAP4 constraint expression.

    Returns a list of ``(var_path, (slice, ...))`` tuples in request order.
    Leaf-relative references (no leading '/', as emitted by some pydap
    consolidation paths) are resolved against the granule when given.
    """
    out = []
    for part in ce.split(";"):
        part = part.strip()
        if not part:
            continue
        match = re.match(r"^(/?[^\[\]]+)((?:\[[^\]]*\])*)$", part)
        if not match:
            raise ValueError(f"Unsupported constraint expression fragment: {part!r}")
        var_path = match.group(1)
        if not var_path.startswith("/") and granule is not None:
            candidates = [
                p for p in granule.variables if p.strip("/").split("/")[-1] == var_path
            ]
            if len(candidates) == 1:
                var_path = candidates[0]
        slabs = []
        for start, step, stop in _HYPERSLAB_RE.findall(match.group(2)):
            slabs.append(slice(int(start), int(stop) + 1, int(step)))
        out.append((var_path, tuple(slabs)))
    return out


def _chunk(chunk_type_byte: int, payload: bytes) -> bytes:
    header = ((chunk_type_byte & 0xFF) << 24) | (len(payload) & 0x00FFFFFF)
    return header.to_bytes(4, "big") + payload


def encode_dap4(granule: Granule, projections, checksums: bool) -> bytes:
    """Encode a constrained DAP4 binary response for a granule.

    ``projections`` is the output of :func:`parse_ce`.
    """
    projected_vars = {}
    projected_dims = {}
    for path, slabs in projections:
        array = granule.array_for(path)
        sliced = np.asarray(array[slabs] if slabs else array)
        projected_vars[path] = (
            sliced,
            granule.dims_for(path),
            granule.attrs_for(path),
        )
        # Track the projected length of every dimension used.
        for dim, axis_length in zip(granule.dims_for(path), sliced.shape):
            projected_dims[dim] = int(axis_length)

    # The response DMR mirrors the projected variables (DAP4 convention),
    # carrying the post-constraint dimension sizes.
    projected = Granule(
        name=granule.name,
        dimensions=projected_dims,
        variables=projected_vars,
        attributes=granule.attributes,
    )
    dmr = build_dmr(projected).encode("ascii")
    body = bytearray()
    body += _chunk(0x04, dmr)  # not-last, little-endian data

    payload = bytearray()
    for path, _slabs in projections:
        array = np.asarray(projected.array_for(path))
        raw = array.astype(array.dtype.newbyteorder("<"), copy=False).tobytes()
        payload += raw
        if checksums:
            payload += (zlib.crc32(raw) & 0xFFFFFFFF).to_bytes(4, "little")
    body += _chunk(0x04, bytes(payload))
    body += _chunk(0x05, b"")  # last chunk
    return bytes(body)


class _DAP4Handler(BaseHTTPRequestHandler):
    server_version = "pydap-test-dap4/1.0"

    def log_message(self, *args):  # silence stderr noise
        pass

    @property
    def state(self):
        return self.server.pydap_state

    def _send(self, status, content_type, body, extra_headers=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for key, value in extra_headers or []:
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        try:
            self._do_get()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:  # never drop the connection silently
            try:
                self._send(500, "text/plain", str(exc).encode())
            except Exception:
                pass

    def _do_get(self):
        state = self.state
        path = self.path
        state["requests"].append({"path": path, "method": "GET"})

        # Fault injection: forced HTTP failures.
        fail_paths = state.get("fail_paths", set())
        for needle in fail_paths:
            if needle in path:
                self._send(state.get("fail_status", 500), "text/plain", b"injected")
                return

        m = re.match(
            r"^/(?P<granule>[^/?]+)(?P<ext>\.dmr|\.dap)(?P<query>\?.*)?$", path
        )
        if not m:
            self._send(404, "text/plain", b"not found")
            return
        granule = state["granules"].get(m.group("granule"))
        if granule is None:
            self._send(404, "text/plain", b"unknown granule")
            return

        query = m.group("query") or ""
        params = {}
        for fragment in query.lstrip("?").split("&"):
            if "=" in fragment:
                key, value = fragment.split("=", 1)
                params[unquote(key)] = unquote(value)
        checksums = params.get("dap4.checksum", "false").lower() == "true"

        if m.group("ext") == ".dmr":
            self._send(200, "text/plain; charset=ascii", build_dmr(granule).encode())
            return

        ce = params.get("dap4.ce", "")
        try:
            projections = (
                parse_ce(ce, granule)
                if ce
                else [(path, ()) for path in granule.variables]
            )
        except ValueError:
            self._send(400, "text/plain", b"bad constraint expression")
            return
        state["constraints"].append(
            [
                (var_path, tuple((s.start, s.stop, s.step) for s in slabs))
                for var_path, slabs in projections
            ]
        )
        body = encode_dap4(granule, projections, checksums)

        # Fault injection: corrupt the checksum bytes of the last variable.
        if state.get("corrupt_checksum"):
            body = bytearray(body)
            body[-8] ^= 0xFF
            body = bytes(body)
        # Fault injection: truncate mid-response.
        truncate = state.get("truncate")
        if truncate:
            body = body[: max(0, len(body) - truncate)]
        self._send(200, "application/octet-stream", body)


class DAP4TestServer:
    """A threaded HTTP server hosting a dictionary of :class:`Granule` objects.

    The server binds to an ephemeral localhost port and records all requests
    in ``self.state['requests']`` and parsed constraints in
    ``self.state['constraints']``.
    """

    def __init__(self, granules):
        self.granules = list(granules)
        self.state = {
            "granules": {g.name: g for g in self.granules},
            "requests": [],
            "constraints": [],
        }
        self._httpd = None
        self._thread = None

    @property
    def port(self):
        return self._httpd.server_address[1]

    def url_for(self, granule_name, scheme="http"):
        return f"{scheme}://127.0.0.1:{self.port}/{granule_name}"

    def start(self):
        port = get_open_port()
        self._httpd = ThreadingHTTPServer(("127.0.0.1", port), _DAP4Handler)
        self._httpd.pydap_state = self.state
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def shutdown(self):
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._thread.join(timeout=10)
            self._httpd = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.shutdown()

    # -- fault injection helpers -----------------------------------------
    def set_truncate(self, nbytes):
        self.state["truncate"] = nbytes

    def set_corrupt_checksum(self, flag=True):
        self.state["corrupt_checksum"] = flag

    def set_http_failure(self, needle, status=500):
        self.state.setdefault("fail_paths", set()).add(needle)
        self.state["fail_status"] = status

    def clear_faults(self):
        self.state["truncate"] = None
        self.state["corrupt_checksum"] = False
        self.state["fail_paths"] = set()

    def reset_log(self):
        self.state["requests"] = []
        self.state["constraints"] = []

    @property
    def dap_paths(self):
        return [r["path"] for r in self.state["requests"] if ".dap" in r["path"]]
