# Server-side grouped aggregation (`group_reduce`)

This extension lets a DAP2 client receive *already aggregated* rows: the
server groups a flat `Sequence` by key columns and computes `count`, `sum`,
`mean`, `min` and `max` aggregates, so the client never downloads the raw
observations. Aggregate results support lazy, order-preserving
`where` / `select` / `limit` stages that are also evaluated on the server.

## Python API

```python
from pydap.client import open_url, result_condition, combine_conditions

remote = open_url("http://server/dataset", application=app)  # or a real URL

# JSON descriptors
keys = '["site"]'
aggregates = (
    '['
    '{"name":"n",      "op":"count", "field":null},'
    '{"name":"mean_v", "op":"mean",  "field":"v"}'
    ']'
)

# Creating a view never performs a request.
view = remote.functions.group_reduce(remote.obs, keys, aggregates)

# The first field access executes the whole pipeline in one .dods request.
for row in view["summary"].iterdata():
    print(row)

# Lazy stages, each returning an independent view.
filtered = view.where(result_condition("mean_v", ">", 1.5))
projected = view.select("site", "mean_v")
page = view.limit(10, offset=20)

# AND of several predicates, either explicitly or by chaining where() calls.
predicate = combine_conditions(
    result_condition("mean_v", ">", 1.5),
    result_condition("n", ">=", 3),
)
rows = list(view.where(predicate)["summary"].iterdata())
```

Stage order is meaningful and preserved exactly as written:
`view.limit(5).where(...)` filters the first five groups, while
`view.where(...).limit(5)` filters all groups and then takes five.

A view can also be obtained directly on the server object:

```python
from pydap.wsgi.group_reduce import group_reduce

result = group_reduce(dataset, dataset.obs, keys, aggregates)
for row in result["summary"].iterdata():
    ...
```

## Configuration

The WSGI middleware accepts two dedicated keyword arguments:

```python
ServerSideFunctions(
    handler,
    group_reduce_max_groups=1024,   # positive int; booleans rejected
    group_reduce_tempdir=None,      # None -> system temp directory
)
```

The grouping engine never holds more than `group_reduce_max_groups` group
states simultaneously. Extra groups are written to sorted spill files; after
the single source traversal, bounded batched merge passes (width
`max(32, 4 * max_groups)`) produce the key-sorted result. Source data is
iterated exactly once. Ordinary `kwargs` function registration continues to
work unchanged.

## Replayable wire protocol (DAP2 constraint expression)

Every query is an ordinary DAP2 server-side function call, so a hand-written
HTTP request produces exactly what the client library sends.

```
group_reduce(<sequence>, "<keys_json>", "<aggregates_json>")
where(<function-result>, "<predicate_json>")
select(<function-result>, "<fields_json>")
limit(<function-result>, <n>[, <offset>])
```

* JSON documents are carried as DAP2 double-quoted string literals; embedded
  `"` and `\` are backslash-escaped.
* The full constraint expression is percent-encoded for transport, so commas,
  parentheses, quotes, backslashes, `&` and non-ASCII characters inside
  string literals can never be parsed as function or CE syntax. The server
  tokenises with a quote-aware splitter.
* Predicate JSON:

  ```json
  {"kind": "cmp", "field": "mean_v", "op": ">", "value": 1.5}
  {"kind": "and", "predicates": [ <predicate>, <predicate> ]}
  ```

* Select JSON (order preserved): `["site", "mean_v"]`.

Example raw request (before percent-encoding):

```
GET /dataset.dods?where(group_reduce(obs,"[\"site\"]",
    "[{\"name\":\"n\",\"op\":\"count\",\"field\":null},
      {\"name\":\"mean_v\",\"op\":\"mean\",\"field\":\"v\"}]"),
    "{\"kind\":\"cmp\",\"field\":\"mean_v\",\"op\":\">\",\"value\":1.5}")
```

Pre-aggregation filtering on *original* columns uses the existing legal DAP2
selection syntax and runs before grouping; post-aggregation predicates run on
the complete grouped result:

```
GET /dataset.dods?group_reduce(obs,...)&obs.v>0
```

## Semantics and guarantees

* Rows sharing the complete key tuple form one group; output is ordered by
  the grouping keys (Unicode code point order for strings).
* `count` returns DAP2 `Int32` (overflow raises explicitly); all other
  aggregates return `Float64`.
* Numeric inputs must be finite, real and non-boolean; any invalid row fails
  the whole request with the offending field named. No partial summaries are
  emitted and rows are never silently skipped.
* Every `where` predicate evaluates against the full grouped row even when a
  later `select` drops the referenced column; referencing a column that was
  already removed by `select` is rejected.
* Empty input yields a `summary` Sequence with the complete column
  description and zero rows.
* Temporary spill files and source iterators are released on normal
  completion, input errors, serialisation errors and early client disconnects.
