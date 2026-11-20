"""Benchmark for the memory-bounded group_reduce engine.

The input is generated row-by-row by a generator (no table is ever held in
memory), matching the requirement that neither the input nor the test store
the whole dataset. For each scenario we record:

* source traversal count (must stay 1),
* peak simultaneously resident groups,
* spill-file disk peak,
* process RSS peak and tracemalloc peak,
* first-result latency and total wall time.

Run with::

    python3 benchmark_group_reduce.py --rows 100000
    python3 benchmark_group_reduce.py --rows 1000000
"""

import argparse
import gc
import json
import os
import resource
import time
import tracemalloc

import numpy as np

from pydap.model import BaseType, DatasetType, SequenceType
from pydap.wsgi.group_reduce import GroupReduceConfig, group_reduce

# A fixed, compact column layout so row generation stays O(1) per row.
DTYPE = np.dtype([("site", "U8"), ("g", "i4"), ("v", "f8")])


class StreamingSequence(SequenceType):
    """A Sequence whose ``iterdata`` yields rows from a generator."""

    def __init__(self, name, row_count, distinct_groups, traversal_counter):
        super().__init__(name)
        self["site"] = BaseType("site", dtype=np.dtype("U8"))
        self["g"] = BaseType("g", dtype=np.dtype("i4"))
        self["v"] = BaseType("v", dtype=np.dtype("f8"))
        self._row_count = row_count
        self._distinct = distinct_groups
        self._traversals = traversal_counter
        self._dtype = DTYPE
        self._shape = ()

    def iterdata(self):
        self._traversals["count"] += 1
        rng = np.random.default_rng(20261120)
        for index in range(self._row_count):
            if self._distinct is None:
                # One distinct key per row (near-all-distinct scenario).
                key = index
            else:
                # A fixed small set of repeating keys.
                key = index % self._distinct
            yield ("site%05d" % key, np.int32(key), float(rng.normal()))


def run_scenario(row_count, distinct_groups, max_groups, tempdir):
    traversals = {"count": 0}
    sequence = StreamingSequence("obs", row_count, distinct_groups, traversals)
    dataset = DatasetType("bench")
    dataset["obs"] = sequence

    config = GroupReduceConfig(
        group_reduce_max_groups=max_groups, group_reduce_tempdir=tempdir
    )
    keys = '["g"]'
    aggregates = json.dumps(
        [
            {"name": "n", "op": "count", "field": None},
            {"name": "sum_v", "op": "sum", "field": "v"},
            {"name": "mean_v", "op": "mean", "field": "v"},
        ]
    )

    gc.collect()
    tracemalloc.start()
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t_start = time.perf_counter()

    result = group_reduce(dataset, sequence, keys, aggregates, config=config)
    engine = result._group_reduce_engine
    engine.prepare()
    t_first_ready = time.perf_counter()

    row_count_out = 0
    for _ in engine.iterrows():
        row_count_out += 1
    t_end = time.perf_counter()

    _, malloc_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    return {
        "rows_in": row_count,
        "groups_out": row_count_out,
        "scenario": "few_groups" if distinct_groups is not None else "all_distinct",
        "max_groups_setting": max_groups,
        "source_traversals": traversals["count"],
        "peak_resident_groups": engine.peak_resident_groups,
        "spill_count": engine.spill_count,
        "peak_disk_bytes": engine.peak_disk_bytes,
        "tracemalloc_peak_bytes": malloc_peak,
        "rss_peak_kib_before": rss_before,
        "rss_peak_kib_after": rss_after,
        "first_result_seconds": round(t_first_ready - t_start, 4),
        "total_seconds": round(t_end - t_start, 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument(
        "--out", default="/workspace/dev/benchmark_results.json"
    )
    args = parser.parse_args()

    import tempfile

    tempdir = tempfile.mkdtemp(prefix="group-reduce-bench-")
    scenarios = [
        # fixed few groups (8 repeating keys)
        (8, 32),
        (8, 256),
        # one distinct key per row
        (None, 32),
        (None, 256),
    ]
    report = []
    for distinct, max_groups in scenarios:
        record = run_scenario(
            args.rows, distinct, max_groups, tempdir
        )
        report.append(record)
        print(json.dumps(record, indent=2))
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
