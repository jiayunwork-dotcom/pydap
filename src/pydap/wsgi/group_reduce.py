"""Server-side grouped aggregation (``group_reduce``) and lazy result stages.

This module implements a memory-bounded, single-pass grouping engine plus the
``where``, ``select`` and ``limit`` result-stage functions that operate on the
grouped result without materialising it on the client.

The public entry points are:

* :func:`group_reduce` -- a pydap server-side function;
* :func:`where`, :func:`select`, :func:`limit` -- server-side functions that
  wrap another function result and describe lazy, ordered processing stages;
* :class:`GroupReduceConfig` -- the per-middleware configuration object.

A short, replayable description of the wire protocol lives in
:mod:`pydap.parsers.group_protocol`.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import tempfile
from collections import OrderedDict

import numpy as np

from pydap.exceptions import ConstraintExpressionError, ServerError
from pydap.model import BaseType, DatasetType, SequenceType

# DAP2 representable scalar column types accepted as keys / values.
_DAP2_SCALAR_CHARS = ("S", "U", "i", "I", "u", "l", "L", "d")
_NUMERIC_CHARS = ("i", "I", "u", "l", "L", "d")
_OPS = ("count", "sum", "mean", "min", "max")
# Aggregate state layout: [count(int64), sum(float64), min(float64), max(float64)]
_N_STATE = 4
_INT32_MIN = np.iinfo(np.int32).min
_INT32_MAX = np.iinfo(np.int32).max


class GroupReduceError(ConstraintExpressionError):
    """Raised for any malformed ``group_reduce`` declaration."""


def _fail(message):
    raise GroupReduceError(message)


# ---------------------------------------------------------------------------
# Descriptor validation (runs before any data row is read)
# ---------------------------------------------------------------------------


def _load_json(argument, label):
    if not isinstance(argument, str):
        _fail("%s must be a JSON-encoded string." % label)
    try:
        return json.loads(argument)
    except (ValueError, TypeError):
        _fail("%s is not valid JSON: %r" % (label, argument))


def validate_descriptors(sequence, keys_raw, aggregates_raw):
    """Validate the ``keys_json`` / ``aggregates_json`` descriptors.

    The validation only inspects the sequence *description* (column names and
    dtypes), so it completes before a single data row is read.
    """
    keys = _load_json(keys_raw, "keys_json")
    if not isinstance(keys, list) or not keys:
        _fail("keys_json must be a non-empty JSON list of field names.")
    if not all(isinstance(k, str) and k for k in keys):
        _fail("keys_json entries must be non-empty field name strings.")
    if len(set(keys)) != len(keys):
        _fail("keys_json contains duplicate keys: %r" % keys)

    aggregates = _load_json(aggregates_raw, "aggregates_json")
    if not isinstance(aggregates, list) or not aggregates:
        _fail("aggregates_json must be a non-empty JSON list.")

    columns = list(sequence.keys())
    for index, item in enumerate(aggregates):
        if not isinstance(item, dict):
            _fail("aggregate #%d must be a JSON object." % index)
        name = item.get("name")
        op = item.get("op")
        field = item.get("field", None)
        if not isinstance(name, str) or not name:
            _fail("aggregate #%d has an invalid name: %r" % (index, name))
        if op not in _OPS:
            _fail(
                "aggregate %r uses unknown op %r; supported ops are %s."
                % (name, op, ", ".join(_OPS))
            )
        if op == "count":
            if field is not None:
                _fail("count aggregate %r must declare field as null." % name)
        else:
            if not isinstance(field, str) or not field:
                _fail(
                    "aggregate %r (%s) must point at a numeric field."
                    % (name, op)
                )
            if field not in columns:
                _fail(
                    "aggregate %r references unknown field %r." % (name, field)
                )
            dtype = sequence[field].dtype
            if dtype is None or dtype.char not in _NUMERIC_CHARS:
                _fail(
                    "aggregate %r field %r must be numeric (Int32, UInt32 or "
                    "Float64); got dtype %r." % (name, field, dtype)
                )
            if dtype.char == "?":
                _fail("boolean columns cannot be aggregated: %r." % field)

    names = [item["name"] for item in aggregates]
    if len(set(names)) != len(names):
        _fail("aggregate names must be unique: %r" % names)
    conflict = set(names) & set(keys)
    if conflict:
        _fail("aggregate names must not clash with key names: %r" % sorted(conflict))

    for key in keys:
        if key not in columns:
            _fail("unknown grouping field %r." % key)
        dtype = sequence[key].dtype
        if dtype is None or dtype.char not in _DAP2_SCALAR_CHARS:
            _fail(
                "grouping field %r must be a DAP2 scalar string, Int32, "
                "UInt32 or Float64; got dtype %r." % (key, dtype)
            )

    return keys, aggregates


# ---------------------------------------------------------------------------
# Row validation and aggregate state updates
# ---------------------------------------------------------------------------


def _as_python(value):
    """Normalise numpy / python scalars to plain python values."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _check_numeric(value, field):
    if isinstance(value, bool):
        _fail("field %r contains boolean data, which is not a valid number." % field)
    number = _as_python(value)
    if isinstance(number, bool):
        _fail("field %r contains boolean data, which is not a valid number." % field)
    if not isinstance(number, (int, float, np.integer, np.floating)):
        _fail(
            "field %r contains non-numeric value %r (%s)."
            % (field, value, type(value).__name__)
        )
    number = float(number)
    if not math.isfinite(number):
        _fail("field %r contains non-finite value %r." % (field, value))
    return number


def _new_state(numeric_fields):
    # count, sum, min, max per numeric aggregate, in declaration order.
    return {
        "count": 0,
        "numeric": [
            {"sum": 0.0, "min": math.inf, "max": -math.inf}
            for _ in numeric_fields
        ],
    }


def _update_state(state, row_values, numeric_fields):
    state["count"] += 1
    if state["count"] > _INT32_MAX:
        raise ServerError(
            "count aggregate overflowed Int32 (more than %d rows in a group)."
            % _INT32_MAX
        )
    for slot, field in zip(state["numeric"], numeric_fields):
        number = _check_numeric(row_values[field], field)
        with np.errstate(over="ignore"):
            new_sum = slot["sum"] + number
        # Detect overflow explicitly rather than silently producing inf.
        if not math.isfinite(new_sum):
            raise ServerError(
                "numeric aggregation overflowed while summing field %r." % field
            )
        slot["sum"] = new_sum
        if number < slot["min"]:
            slot["min"] = number
        if number > slot["max"]:
            slot["max"] = number


def _merge_state(target, other):
    target["count"] += other["count"]
    if target["count"] > _INT32_MAX:
        raise ServerError("count aggregate overflowed Int32 during spill merge.")
    for a, b in zip(target["numeric"], other["numeric"]):
        with np.errstate(over="ignore"):
            merged = a["sum"] + b["sum"]
        if not math.isfinite(merged):
            raise ServerError("numeric aggregation overflowed during spill merge.")
        a["sum"] = merged
        if b["min"] < a["min"]:
            a["min"] = b["min"]
        if b["max"] > a["max"]:
            a["max"] = b["max"]


def _finalize_aggregate(op, count, slot):
    if op == "sum":
        result = slot["sum"]
    elif op == "mean":
        result = slot["sum"] / count
    elif op == "min":
        result = slot["min"]
    elif op == "max":
        result = slot["max"]
    else:  # pragma: no cover - guarded by validation
        raise ServerError("Unknown aggregate op %r." % op)
    result = float(result)
    if not math.isfinite(result):
        raise ServerError("aggregate result for op %r is not finite." % op)
    return result


# ---------------------------------------------------------------------------
# Spill run support (pickle stream: one (key, state) tuple per record)
# ---------------------------------------------------------------------------


class _SpillRun:
    """A sorted run of (key, state) pairs stored in a temporary file."""

    def __init__(self, path, size):
        self.path = path
        self.size = size
        self._handle = None

    def open(self):
        self._handle = open(self.path, "rb")
        return self

    def read_one(self):
        try:
            return pickle.load(self._handle)
        except EOFError:
            return None

    def close(self):
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def destroy(self):
        self.close()
        try:
            os.remove(self.path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Lazy, memory-bounded grouped data wrappers
# ---------------------------------------------------------------------------


class _GroupReduceData:
    """Single-pass grouping engine exposed as a structured-array-like object.

    The source sequence is iterated at most once, lazily, on the first read.
    At most ``max_groups`` group states reside in memory at any moment; extra
    groups are flushed to sorted spill runs under ``tempdir`` and merged with
    a k-way external merge. No result row is yielded before the whole input
    has been validated.
    """

    def __init__(
        self,
        source_iter,
        keys,
        aggregates,
        source_columns,
        key_dtypes,
        max_groups,
        tempdir,
    ):
        self._source_iter = source_iter
        self._keys = list(keys)
        self._aggregates = aggregates
        self._source_columns = source_columns
        self._key_dtypes = key_dtypes
        self._max_groups = max_groups
        self._tempdir = tempdir
        # Width of each post-ingestion batched merge pass. Bounding it keeps
        # both the final k-way merge heap and the per-pass open-file count
        # independent of the total number of distinct groups.
        self._merge_width = max(32, max_groups * 4)
        self._numeric_fields = [
            item["field"] for item in aggregates if item["op"] != "count"
        ]
        # column index of every field we actually read from each row
        self._needed = list(
            dict.fromkeys(list(self._keys) + list(self._numeric_fields))
        )
        self._needed_index = {name: source_columns.index(name) for name in self._needed}

        self._closed = False
        self._consumed = False
        self._runs = []
        self._temp_dir_obj = None
        self._disk_bytes = 0
        self._peak_disk_bytes = 0
        self._peak_groups = 0
        self._spill_count = 0
        self._source_rows = 0
        self._mem = OrderedDict()

        # output dtype: keys keep their source kind, counts are Int32, the
        # rest Float64.
        dtype_fields = []
        for key in self._keys:
            char = key_dtypes[key].char
            if char in ("S", "U"):
                # Rows carry python ``str`` values; the summary child exposes
                # a numpy unicode dtype so DDS/DODS map it to DAP2 String.
                dtype_fields.append((key, object))
            elif char in ("i", "l"):
                dtype_fields.append((key, np.int32))
            elif char in ("I", "L", "u"):
                dtype_fields.append((key, np.uint32))
            else:
                dtype_fields.append((key, np.float64))
        for item in aggregates:
            if item["op"] == "count":
                dtype_fields.append((item["name"], np.int32))
            else:
                dtype_fields.append((item["name"], np.float64))
        self._dtype = np.dtype(dtype_fields)
        self._names = [name for name, _ in dtype_fields]

    # -- properties expected by the DODS/DDS machinery ---------------------

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        # The length is unknown until the source is consumed; pydap treats a
        # missing/unknown sequence length as a scalar-sized description.
        return ()

    def __len__(self):  # pragma: no cover - never called by the response path
        raise TypeError("group_reduce result length is not known until read")

    @property
    def peak_disk_bytes(self):
        return self._peak_disk_bytes

    @property
    def peak_resident_groups(self):
        return self._peak_groups

    @property
    def spill_count(self):
        return self._spill_count

    @property
    def source_rows(self):
        return self._source_rows

    # -- lifecycle ----------------------------------------------------------

    def close(self):
        if self._closed:
            return
        self._closed = True
        for run in self._runs:
            run.destroy()
        self._runs = []
        if self._temp_dir_obj is not None:
            try:
                self._temp_dir_obj.cleanup()
            except OSError:
                pass
            self._temp_dir_obj = None
        # Release the source iterator eagerly.
        self._source_iter = None
        self._mem = None

    def _ensure_tempdir(self):
        if self._temp_dir_obj is None:
            self._temp_dir_obj = tempfile.TemporaryDirectory(
                prefix="pydap-group-reduce-", dir=self._tempdir
            )
        return self._temp_dir_obj.name

    # -- ingestion ----------------------------------------------------------

    def _normalise_key_value(self, field, value):
        value = _as_python(value)
        char = self._key_dtypes[field].char
        if char in ("S", "U"):
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            if not isinstance(value, str):
                _fail(
                    "grouping field %r contains non-string value %r."
                    % (field, value)
                )
            return value
        if isinstance(value, bool):
            _fail("grouping field %r contains boolean data." % field)
        if char in ("i", "l"):
            if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
                _fail(
                    "grouping field %r declared Int32 but got %r."
                    % (field, value)
                )
            if not (_INT32_MIN <= int(value) <= _INT32_MAX):
                _fail("grouping field %r value %r is outside Int32." % (field, value))
            return int(value)
        if char in ("I", "L", "u"):
            if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
                _fail(
                    "grouping field %r declared UInt32 but got %r."
                    % (field, value)
                )
            if not (0 <= int(value) <= np.iinfo(np.uint32).max):
                _fail("grouping field %r value %r is outside UInt32." % (field, value))
            return int(value)
        # float key
        number = _check_numeric(value, field)
        return number

    def _spill(self):
        ordered = sorted(self._mem.items(), key=lambda kv: kv[0])
        directory = self._ensure_tempdir()
        fd, path = tempfile.mkstemp(prefix="run-", suffix=".pkl", dir=directory)
        size = 0
        with os.fdopen(fd, "wb") as handle:
            for pair in ordered:
                pickle.dump(pair, handle, protocol=pickle.HIGHEST_PROTOCOL)
                size = handle.tell()
        self._disk_bytes += size
        self._peak_disk_bytes = max(self._peak_disk_bytes, self._disk_bytes)
        self._runs.append(_SpillRun(path, size))
        self._spill_count += 1
        self._mem.clear()
        # Runs are compacted in bounded batches *after* ingestion (see
        # _finalize_runs), never while scanning the source, so the input
        # traversal stays single-pass and each record is rewritten only
        # O(log total_groups / log merge_width) times.

    def _write_merged(self, runs):
        """Merge several sorted runs into one new run (bounded heap)."""
        import heapq

        directory = self._ensure_tempdir()
        streams = [run.open() for run in runs]
        heap = []
        for index, run in enumerate(streams):
            pair = run.read_one()
            if pair is not None:
                heapq.heappush(heap, (pair[0], index, pair[1]))

        fd, merged_path = tempfile.mkstemp(
            prefix="run-merged-", suffix=".pkl", dir=directory
        )
        merged_size = 0
        current_key = None
        merged_state = None
        with os.fdopen(fd, "wb") as handle:
            while heap:
                key, index, state = heapq.heappop(heap)
                nxt = streams[index].read_one()
                if nxt is not None:
                    heapq.heappush(heap, (nxt[0], index, nxt[1]))
                if current_key is None or key != current_key:
                    if merged_state is not None:
                        pickle.dump(
                            (current_key, merged_state),
                            handle,
                            protocol=pickle.HIGHEST_PROTOCOL,
                        )
                    current_key = key
                    merged_state = {
                        "count": 0,
                        "numeric": [
                            {"sum": 0.0, "min": math.inf, "max": -math.inf}
                            for _ in self._numeric_fields
                        ],
                    }
                _merge_state(merged_state, state)
            if merged_state is not None:
                pickle.dump(
                    (current_key, merged_state),
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            merged_size = handle.tell()

        for run in streams:
            run.close()
        old_size = sum(run.size for run in runs)
        # Inputs and output coexist on disk while the merge runs.
        self._peak_disk_bytes = max(
            self._peak_disk_bytes, self._disk_bytes + merged_size
        )
        for run in runs:
            run.destroy()
        self._disk_bytes -= old_size
        merged = _SpillRun(merged_path, merged_size)
        self._disk_bytes += merged_size
        self._peak_disk_bytes = max(
            self._peak_disk_bytes, self._disk_bytes
        )
        return merged

    def _finalize_runs(self):
        """Repeatedly batch-merge spill runs until only a few remain.

        Every pass merges groups of at most ``_merge_width`` runs, so the
        merge heap never exceeds that width and the number of passes grows
        only logarithmically with the number of distinct groups.
        """
        width = self._merge_width
        while len(self._runs) > width:
            merged_runs = []
            for start in range(0, len(self._runs), width):
                batch = self._runs[start : start + width]
                merged_runs.append(self._write_merged(batch))
            self._runs = merged_runs

    def _ingest(self):
        if self._consumed:
            return
        try:
            for raw_row in self._source_iter:
                self._source_rows += 1
                row = {}
                for name in self._needed:
                    row[name] = raw_row[self._needed_index[name]]
                key = tuple(
                    self._normalise_key_value(name, row[name]) for name in self._keys
                )
                state = self._mem.get(key)
                if state is None:
                    if len(self._mem) >= self._max_groups:
                        self._spill()
                    state = _new_state(self._numeric_fields)
                    self._mem[key] = state
                    self._peak_groups = max(self._peak_groups, len(self._mem))
                _update_state(state, row, self._numeric_fields)
            # Compact the spill runs down to at most one merge width's worth
            # before any result row can be produced.
            self._finalize_runs()
            self._consumed = True
        except BaseException:
            self.close()
            raise

    # -- output -------------------------------------------------------------

    def _result_row(self, key, state):
        row = list(key)
        numeric_iter = iter(state["numeric"])
        for item in self._aggregates:
            if item["op"] == "count":
                if state["count"] > _INT32_MAX:  # pragma: no cover
                    raise ServerError("count aggregate overflowed Int32.")
                row.append(np.int32(state["count"]))
            else:
                slot = next(numeric_iter)
                row.append(
                    np.float64(
                        _finalize_aggregate(
                            item["op"], state["count"], slot
                        )
                    )
                )
        return tuple(row)

    def _iter_in_memory(self):
        for key, state in sorted(self._mem.items(), key=lambda kv: kv[0]):
            yield self._result_row(key, state)

    def _iter_merged(self):
        # k-way merge over the spill runs plus the final in-memory partition.
        import heapq

        streams = []
        for run in self._runs:
            run.open()
            streams.append(run)
        heap = []
        for index, run in enumerate(streams):
            pair = run.read_one()
            if pair is not None:
                heapq.heappush(heap, (pair[0], index, pair[1]))
        mem_iter = iter(sorted(self._mem.items(), key=lambda kv: kv[0]))
        mem_next = next(mem_iter, None)

        def push_memory():
            nonlocal mem_next
            if mem_next is not None:
                return mem_next
            return None

        current_key = None
        merged_state = None
        try:
            while heap or mem_next is not None:
                take_mem = False
                if mem_next is not None and (
                    not heap or mem_next[0] <= heap[0][0]
                ):
                    take_mem = True
                if take_mem:
                    key, state = mem_next
                    mem_next = next(mem_iter, None)
                else:
                    key, index, state = heapq.heappop(heap)
                    nxt = streams[index].read_one()
                    if nxt is not None:
                        heapq.heappush(heap, (nxt[0], index, nxt[1]))
                if current_key is None or key != current_key:
                    if merged_state is not None:
                        yield self._result_row(current_key, merged_state)
                    current_key = key
                    merged_state = {
                        "count": 0,
                        "numeric": [
                            {"sum": 0.0, "min": math.inf, "max": -math.inf}
                            for _ in self._numeric_fields
                        ],
                    }
                _merge_state(merged_state, state)
            if merged_state is not None:
                yield self._result_row(current_key, merged_state)
        finally:
            for run in streams:
                run.close()

    def iterrows(self):
        """Iterate the final, key-sorted result rows exactly once."""
        try:
            # Phase 1: consume and validate the entire source. Nothing is
            # yielded until this completes successfully.
            self._ingest()
            # Phase 2: stream the ordered, bounded-memory result.
            if not self._runs:
                iterator = self._iter_in_memory()
            else:
                iterator = self._iter_merged()
            for row in iterator:
                yield row
        except BaseException:
            self.close()
            raise
        # Normal completion: release everything as soon as the last row left
        # the generator.
        self.close()

    def prepare(self):
        """Run the full ingestion/validation phase without emitting rows.

        After this returns successfully the input has been fully consumed and
        the response can be serialised safely. It is idempotent.
        """
        try:
            self._ingest()
        except BaseException:
            self.close()
            raise
        return self

    def iterdata(self):
        return self.iterrows()

    def __iter__(self):
        return self.iterrows()


class _StageData:
    """Lazy where/select/limit pipeline over a :class:`_GroupReduceData`.

    Stages are applied in the exact order in which the client declared them;
    filtering always evaluates against full grouped rows (all columns are
    computed even if a later ``select`` drops them).
    """

    def __init__(self, engine, columns, stages):
        self._engine = engine
        # ``columns`` is an ordered list of (name, numpy dtype) pairs.
        self._columns = list(columns)
        self._stages = list(stages)
        self._dtype = np.dtype(
            [(name, dtype) for name, dtype in self._columns]
        )

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return ()

    @property
    def columns(self):
        return list(self._columns)

    @property
    def stages(self):
        return list(self._stages)

    def _column_index(self, name):
        for index, (candidate, _) in enumerate(self._columns):
            if candidate == name:
                return index
        raise ConstraintExpressionError(
            "unknown result column %r; available columns: %s"
            % (name, [c for c, _ in self._columns])
        )

    def _check_predicate(self, predicate):
        if not isinstance(predicate, dict):
            _fail("where predicate must be a JSON object.")
        kind = predicate.get("kind")
        if kind == "and":
            for sub in predicate.get("predicates", []):
                self._check_predicate(sub)
            return
        if kind != "cmp":
            _fail("unsupported where predicate %r." % kind)
        field = predicate.get("field")
        op = predicate.get("op")
        if op not in ("=", "!=", "<", "<=", ">", ">="):
            _fail("unsupported comparison operator %r." % op)
        index = self._column_index(field)
        value = predicate.get("value")
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            _fail("where constant must be a string or a finite number.")
        if isinstance(value, float) and not math.isfinite(value):
            _fail("where constant must be finite.")
        column_dtype = self._columns[index][1]
        if isinstance(value, str) and column_dtype.kind != "O":
            _fail(
                "string constant cannot be compared to numeric column %r." % field
            )
        if not isinstance(value, str) and column_dtype.kind == "O":
            _fail("numeric constant cannot be compared to string column %r." % field)

    def with_stage(self, stage):
        stages = self._stages + [stage]
        columns = self._columns
        kind = stage[0]
        if kind == "where":
            self._check_predicate(stage[1])
        elif kind == "select":
            wanted = stage[1]
            known = [name for name, _ in self._columns]
            for name in wanted:
                if name not in known:
                    raise ConstraintExpressionError(
                        "select references column %r that is not present; "
                        "available columns: %s" % (name, known)
                    )
            columns = [(name, dict(self._columns)[name]) for name in wanted]
        elif kind == "limit":
            pass
        else:  # pragma: no cover
            _fail("unknown result stage %r." % kind)
        return _StageData(self._engine, columns, stages)

    def prepare(self):
        """Force the full grouping/validation phase before serialization."""
        return self._engine.prepare()

    @staticmethod
    def _apply_op(op, left, right):
        if op == "=":
            return left == right
        if op == "!=":
            return left != right
        if op == "<":
            return left < right
        if op == "<=":
            return left <= right
        if op == ">":
            return left > right
        if op == ">=":
            return left >= right
        raise ConstraintExpressionError("unsupported operator %r" % op)  # pragma

    def _eval_predicate(self, predicate, row):
        if predicate["kind"] == "and":
            return all(
                self._eval_predicate(sub, row)
                for sub in predicate["predicates"]
            )
        # Predicates always address the engine's *full* grouped row, even
        # when a later select stage dropped the referenced column.
        full_names = list(self._engine.dtype.names)
        index = full_names.index(predicate["field"])
        return self._apply_op(predicate["op"], row[index], predicate["value"])

    def iterrows(self):
        # Stages are replayed in declaration order over the engine's full,
        # ordered rows. Every predicate evaluates against the full row even
        # when a later ``select`` drops the referenced column.
        full_names = list(self._engine.dtype.names)
        visible = [name for name, _ in self._columns]
        visible_index = [full_names.index(name) for name in visible]
        limit_state = {}
        for row in self._engine.iterrows():
            keep = True
            for stage_index, stage in enumerate(self._stages):
                if stage[0] == "where":
                    if not self._eval_predicate(stage[1], row):
                        keep = False
                        break
                elif stage[0] == "limit":
                    n, offset = stage[1], stage[2]
                    skipped, taken = limit_state.setdefault(
                        stage_index, [0, 0]
                    )
                    if skipped < offset:
                        limit_state[stage_index][0] += 1
                        keep = False
                        break
                    if taken >= n:
                        keep = False
                        break
                    limit_state[stage_index][1] += 1
            if keep:
                yield tuple(row[i] for i in visible_index)

    def iterdata(self):
        return self.iterrows()

    def __iter__(self):
        return self.iterrows()


# ---------------------------------------------------------------------------
# Sequence construction
# ---------------------------------------------------------------------------


def _build_summary(engine):
    summary = SequenceType("summary")
    for name in engine.dtype.names:
        dtype = engine.dtype[name]
        if dtype.kind == "O":
            dtype = np.dtype("U")
        child = BaseType(name, dtype=dtype, shape=())
        summary[name] = child
    summary._data = engine
    return summary


# ---------------------------------------------------------------------------
# Public server-side functions
# ---------------------------------------------------------------------------


def group_reduce(dataset, sequence, keys_json, aggregates_json, config=None):
    """Group a flat Sequence by key columns and compute aggregates.

    Parameters mirror the task specification: ``keys_json`` and
    ``aggregates_json`` are JSON strings; the returned object is a
    :class:`~pydap.model.DatasetType` containing a single Sequence named
    ``summary``.
    """
    if not isinstance(sequence, SequenceType):
        _fail("group_reduce must be applied to a flat Sequence.")
    if not all(isinstance(child, BaseType) for child in sequence.children()):
        _fail("group_reduce only supports flat Sequences.")

    keys, aggregates = validate_descriptors(sequence, keys_json, aggregates_json)
    if config is None:
        config = GroupReduceConfig()
    source_columns = list(sequence.keys())
    key_dtypes = {key: sequence[key].dtype for key in keys}

    engine = _GroupReduceData(
        source_iter=sequence.iterdata(),
        keys=keys,
        aggregates=aggregates,
        source_columns=source_columns,
        key_dtypes=key_dtypes,
        max_groups=config.group_reduce_max_groups,
        tempdir=config.group_reduce_tempdir,
    )
    summary = _build_summary(engine)

    out = DatasetType("group_reduce")
    out["summary"] = summary
    out._group_reduce_engine = engine
    return out


def _extract_engine(result):
    if isinstance(result, DatasetType) and "summary" in result and hasattr(
        result, "_group_reduce_engine"
    ):
        return result
    if isinstance(result, SequenceType) and hasattr(result, "_data") and isinstance(
        result._data, (_GroupReduceData, _StageData)
    ):
        dataset = DatasetType("stage")
        dataset["summary"] = result
        return dataset
    _fail("where/select/limit must wrap a group_reduce function result.")


def where(dataset, result, expression_json):
    """Lazily filter a ``summary`` result with a JSON comparison expression."""
    container = _extract_engine(result)
    summary = container["summary"]
    data = summary._data
    predicate = _load_json(expression_json, "where expression")
    staged = _StageData(
        data._engine if isinstance(data, _StageData) else data,
        _stage_columns(data),
        _stage_list(data),
    ).with_stage(("where", predicate))
    return _wrap_stage(container, summary, staged)


def select(dataset, result, fields_json):
    """Lazily project a ``summary`` result to the given columns, in order."""
    container = _extract_engine(result)
    summary = container["summary"]
    data = summary._data
    fields = _load_json(fields_json, "select fields")
    if not isinstance(fields, list) or not fields or not all(
        isinstance(name, str) for name in fields
    ):
        _fail("select fields must be a non-empty JSON list of strings.")
    staged = _StageData(
        data._engine if isinstance(data, _StageData) else data,
        _stage_columns(data),
        _stage_list(data),
    ).with_stage(("select", list(fields)))
    return _wrap_stage(container, summary, staged)


def limit(dataset, result, n, offset=0):
    """Lazily take ``n`` rows after skipping ``offset``, in result order."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        _fail("limit n must be a non-negative integer (booleans rejected).")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        _fail("limit offset must be a non-negative integer (booleans rejected).")
    container = _extract_engine(result)
    summary = container["summary"]
    data = summary._data
    staged = _StageData(
        data._engine if isinstance(data, _StageData) else data,
        _stage_columns(data),
        _stage_list(data),
    ).with_stage(("limit", n, offset))
    return _wrap_stage(container, summary, staged)


def _stage_columns(data):
    if isinstance(data, _StageData):
        return data.columns
    return [(name, data.dtype[name]) for name, _ in data.dtype.descr]


def _stage_list(data):
    return data.stages if isinstance(data, _StageData) else []


def _wrap_stage(container, previous_summary, staged_data):
    summary = SequenceType("summary")
    for name, dtype in staged_data.columns:
        if dtype.kind == "O":
            dtype = np.dtype("U")
        summary[name] = BaseType(name, dtype=dtype, shape=())
    summary._data = staged_data
    out = DatasetType(container.name)
    out["summary"] = summary
    out._group_reduce_engine = (
        staged_data._engine
        if isinstance(staged_data, _StageData)
        else staged_data
    )
    return out


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class GroupReduceConfig:
    """Configuration for the grouping engine.

    ``group_reduce_max_groups`` is a positive integer (booleans rejected).
    ``group_reduce_tempdir`` is either ``None`` (system temp directory) or an
    existing directory path.
    """

    def __init__(
        self, group_reduce_max_groups=1024, group_reduce_tempdir=None
    ):
        self.group_reduce_max_groups = self._validate_max_groups(
            group_reduce_max_groups
        )
        self.group_reduce_tempdir = self._validate_tempdir(group_reduce_tempdir)

    @staticmethod
    def _validate_max_groups(value):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                "group_reduce_max_groups must be a positive integer (booleans "
                "are rejected)."
            )
        if value <= 0:
            raise ValueError("group_reduce_max_groups must be positive.")
        return value

    @staticmethod
    def _validate_tempdir(value):
        if value is None:
            return None
        if not isinstance(value, str) or not os.path.isdir(value):
            raise ValueError(
                "group_reduce_tempdir must be None or an existing directory."
            )
        return value


group_reduce.__version__ = "1.0"
where.__version__ = "1.0"
select.__version__ = "1.0"
limit.__version__ = "1.0"
