# `group_reduce` engine design and measured resource behaviour

## Organisation alternatives considered

The engine must (a) scan the source `Sequence` exactly once, (b) keep
simultaneously resident group states at or below
`group_reduce_max_groups`, (c) deliver groups in key order after full input
validation, and (d) release every temporary resource deterministically.

Three organisations were compared:

| Organisation | Many repeated keys | Near-all-distinct keys | Disk I/O | First-result latency |
| --- | --- | --- | --- | --- |
| In-memory hash map, sort at the end | O(n) time, tiny state | Unbounded memory — rejected | None | One full scan |
| External sort of all rows, then stream-reduce | Stores every row (and every unneeded column) on disk | O(n log n) merge cost | Highest: all rows written and re-read | One full scan plus full sort |
| **Chosen: bounded hash partition + sorted spill runs + batched k-way merge** | No spills when distinct groups ≤ cap; behaves like the in-memory map | States spill in bounded partitions; post-ingestion merge passes of fixed width | Proportional to distinct *group states*, never to all rows; each state rewritten O(log G / log W) times | One full scan (plus merge only when spilling) |

The chosen design keeps a fixed-capacity ordered dictionary of group states.
When a new key arrives and the cap is reached, the resident states are sorted
and appended as a new spill run. After ingestion, runs are merged in batches
of width `W = max(32, 4 * max_groups)` until at most `W` runs remain, then a
final bounded k-way merge yields the key-sorted output. This bounds both
resident group states (≤ `max_groups`) and open merge streams (≤ `W`),
independently of the number of distinct groups, while never writing raw
input rows.

## Measured results

Generated row-by-row (the whole table is never materialised by the test) by
`dev/benchmark_group_reduce.py` on Python 3.11 / NumPy 2.x. RSS values are
process-wide `ru_maxrss`; tracemalloc values cover only engine allocations
while the benchmark timer is active.

### 100,000 rows

| Scenario | max_groups | traversals | peak resident groups | spill runs | disk peak | tracemalloc peak | first result | total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 repeating keys | 32 | 1 | 8 | 0 | 0 | ~0.92 MB | 1.43 s | 1.43 s |
| 8 repeating keys | 256 | 1 | 8 | 0 | 0 | ~0.12 MB | 1.57 s | 1.57 s |
| all-distinct keys | 32 | 1 | 32 | 3,124 | 13.9 MB | ~1.7 MB | 3.80 s | 5.41 s |
| all-distinct keys | 256 | 1 | 256 | 390 | 13.4 MB | ~2.6 MB | 2.16 s | 3.80 s |

### 1,000,000 rows

| Scenario | max_groups | traversals | peak resident groups | spill runs | disk peak | tracemalloc peak | RSS peak | first result | total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 repeating keys | 32 | 1 | 8 | 0 | 0 | ~0.92 MB | ~53 MB | 14.7 s | 14.7 s |
| 8 repeating keys | 256 | 1 | 8 | 0 | 0 | ~0.12 MB | ~53 MB | 14.8 s | 14.8 s |
| all-distinct keys | 32 | 1 | 32 | 31,249 | 205.5 MB | ~9.0 MB | ~79 MB | 55.0 s | 71.8 s |
| all-distinct keys | 256 | 1 | 256 | 3,906 | 170.2 MB | ~7.1 MB | ~79 MB | 37.9 s | 54.7 s |

Observations:

* The source is traversed exactly once in every scenario.
* Resident group states never exceed the configured cap; tracemalloc peaks
  stay in the single-digit-MB range even for one million distinct groups,
  i.e. engine memory does not grow with the number of input rows.
* Repeating-key workloads never touch disk.
* A larger `max_groups` trades more memory for fewer, larger spill runs and
  substantially lower first-result latency in the all-distinct case.
* Disk volume scales with the number of distinct *groups* (spill states),
  not with the number of input rows.

Reproduce:

```bash
python3 dev/benchmark_group_reduce.py --rows 100000 --out dev/bench_100k.json
python3 dev/benchmark_group_reduce.py --rows 1000000 --out dev/bench_1m.json
```
