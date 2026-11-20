"""Tests for server-side grouped aggregation and lazy result stages.

The truth values in these tests are computed independently with plain Python
/ NumPy -- never with the query machinery under test.
"""

import collections
import json
import math
import os
import tempfile
import unittest

import numpy as np
import requests
from webtest import TestApp as App

from pydap.client import (
    LazyFunctionResult,
    combine_conditions,
    open_dods_url,
    open_url,
    result_condition,
)
from pydap.exceptions import ConstraintExpressionError
from pydap.handlers.lib import BaseHandler
from pydap.model import BaseType, DatasetType, SequenceType
from pydap.parsers.group_protocol import (
    compact_json,
    decode_dap_string,
    encode_dap_string,
    parse_call,
    split_ce,
    split_top_level,
)
from pydap.wsgi.group_reduce import (
    GroupReduceConfig,
    group_reduce as group_reduce_func,
)
from pydap.wsgi.ssf import ServerSideFunctions

AGGS = compact_json(
    [
        {"name": "n", "op": "count", "field": None},
        {"name": "sum_v", "op": "sum", "field": "v"},
        {"name": "mean_v", "op": "mean", "field": "v"},
        {"name": "min_v", "op": "min", "field": "v"},
        {"name": "max_v", "op": "max", "field": "v"},
    ]
)
KEYS = compact_json(["site"])


def build_dataset(rows, dtypes=None, name="stations", seq_name="obs"):
    """Build a flat Sequence dataset from python tuples."""
    if dtypes is None:
        dtypes = [("site", "U16"), ("v", "f8")]
    dataset = DatasetType(name)
    sequence = SequenceType(seq_name)
    for column, dtype in dtypes:
        sequence[column] = BaseType(column)
    sequence.data = np.array(rows, dtype=dtypes)
    dataset[seq_name] = sequence
    return dataset


def make_app(dataset, **config):
    return App(ServerSideFunctions(BaseHandler(dataset), **config))


def dap_string(value):
    return encode_dap_string(value)


def group_ce(sequence="obs", keys=KEYS, aggregates=AGGS):
    return "group_reduce(%s,%s,%s)" % (
        sequence,
        dap_string(keys),
        dap_string(aggregates),
    )


def encode_ce(ce):
    return requests.utils.quote(ce, safe="")


class GroupReduceTruth(unittest.TestCase):
    """Hand-computed small table: filtered rows, per-group results, views."""

    ROWS = [
        ("s1", -5.0),
        ("s1", 10.0),
        ("s1", 20.0),
        ("s2", 1.0),
        ("s2", 2.0),
        ("s3", -10.0),
        ("s3", -2.0),
        ("s3", 4.0),
    ]

    def setUp(self):
        self.dataset = build_dataset(self.ROWS)
        self.app = make_app(self.dataset)
        self.remote = open_url(
            "http://localhost/stations",
            application=ServerSideFunctions(BaseHandler(self.dataset)),
        )

    def test_independent_truth_full_table(self):
        # Independent truth computed with plain dictionaries.
        buckets = collections.defaultdict(list)
        for site, value in self.ROWS:
            buckets[site].append(value)
        truth = []
        for site in sorted(buckets):
            values = buckets[site]
            truth.append(
                (
                    site,
                    len(values),
                    sum(values),
                    sum(values) / len(values),
                    min(values),
                    max(values),
                )
            )

        view = self.remote.functions.group_reduce(self.remote.obs, KEYS, AGGS)
        got = [tuple(row) for row in view["summary"].iterdata()]
        self.assertEqual([g[0] for g in got], [t[0] for t in truth])
        for group, expected in zip(got, truth):
            self.assertEqual(group[0], expected[0])
            self.assertEqual(group[1], expected[1])
            np.testing.assert_allclose(
                group[2:], expected[2:], rtol=1e-10, atol=1e-12
            )

    def test_pre_filter_vs_post_filter_differ(self):
        """Filtering the raw column and filtering the mean must differ."""
        view = self.remote.functions.group_reduce(self.remote.obs, KEYS, AGGS)

        # Post-aggregation filter on the grouped mean.
        post = [
            tuple(row)
            for row in view.where(result_condition("mean_v", ">", 5))[
                "summary"
            ].iterdata()
        ]
        # Independent truth for the post case.
        truth_post = []
        for site in sorted({r[0] for r in self.ROWS}):
            values = [v for s, v in self.ROWS if s == site]
            if sum(values) / len(values) > 5:
                truth_post.append(site)
        self.assertEqual([r[0] for r in post], truth_post)

        # Pre-aggregation filter: an ordinary DAP2 selection on the raw row.
        pre_url = view.url + encode_ce("&obs.v>0")
        pre_dataset = open_dods_url(pre_url, application=self.remote.functions.application)
        pre = [tuple(row) for row in pre_dataset["summary"].iterdata()]
        filtered = [(s, v) for s, v in self.ROWS if v > 0]
        truth_pre = []
        for site in sorted({r[0] for r in filtered}):
            values = [v for s, v in filtered if s == site]
            truth_pre.append((site, len(values), sum(values) / len(values)))
        self.assertEqual(len(pre), len(truth_pre))
        for row, (site, count, mean) in zip(pre, truth_pre):
            self.assertEqual(row[0], site)
            self.assertEqual(row[1], count)
            self.assertTrue(math.isclose(row[3], mean, rel_tol=1e-10, abs_tol=1e-12))

        # The two strategies must give visibly different answers.
        self.assertNotEqual(
            [r[0] for r in post], [r[0] for r in pre]
        )

    def test_where_limit_order_matters(self):
        # Hand-picked data so the first ordered group fails the predicate:
        # ordering is s1 (mean -1), s2 (mean 1.5), s3 (mean 8.33...).
        rows = [
            ("s1", -4.0),
            ("s1", 2.0),
            ("s2", 1.0),
            ("s2", 2.0),
            ("s3", 5.0),
            ("s3", 10.0),
            ("s3", 10.0),
        ]
        dataset = build_dataset(rows)
        remote = open_url(
            "http://localhost/",
            application=ServerSideFunctions(BaseHandler(dataset)),
        )
        view = remote.functions.group_reduce(remote.obs, KEYS, AGGS)
        condition = result_condition("mean_v", ">", 0)
        limit_first = [
            tuple(r)
            for r in view.limit(2).where(condition)["summary"].iterdata()
        ]
        where_first = [
            tuple(r)
            for r in view.where(condition).limit(2)["summary"].iterdata()
        ]
        self.assertNotEqual(limit_first, where_first)
        # limit(2) first keeps s1,s2; the post-filter drops s1, leaving s2.
        self.assertEqual([r[0] for r in limit_first], ["s2"])
        # where first leaves s2,s3; limit(2) keeps both.
        self.assertEqual([r[0] for r in where_first], ["s2", "s3"])

    def test_chained_where_is_intersection(self):
        view = self.remote.functions.group_reduce(self.remote.obs, KEYS, AGGS)
        chained = view.where(result_condition("n", "=", 3)).where(
            result_condition("min_v", "<", 0)
        )
        rows = [tuple(r) for r in chained["summary"].iterdata()]
        anded = view.where(
            combine_conditions(
                result_condition("n", "=", 3),
                result_condition("min_v", "<", 0),
            )
        )
        and_rows = [tuple(r) for r in anded["summary"].iterdata()]
        self.assertEqual(rows, and_rows)
        self.assertEqual([r[0] for r in rows], ["s1", "s3"])

    def test_select_order_and_deleted_column(self):
        view = self.remote.functions.group_reduce(self.remote.obs, KEYS, AGGS)
        projected = view.select("mean_v", "site")
        rows = [tuple(r) for r in projected["summary"].iterdata()]
        self.assertEqual(
            list(projected["summary"].keys()), ["mean_v", "site"]
        )
        self.assertEqual(len(rows), 3)

        # Filtering still works on a column that a *later* select drops, and
        # the dropped column never appears in the output.
        mixed = view.where(result_condition("n", ">=", 3)).select("site")
        rows = [tuple(r) for r in mixed["summary"].iterdata()]
        self.assertEqual(list(mixed["summary"].keys()), ["site"])
        self.assertEqual([r[0] for r in rows], ["s1", "s3"])

        # Referencing an already-deleted column is rejected (lazily, before
        # the request is sent).
        with self.assertRaises(ConstraintExpressionError):
            view.select("site").where(result_condition("n", "=", 1))
        with self.assertRaises(ConstraintExpressionError):
            view.select("site").select("n")

    def test_limit_offset_semantics(self):
        view = self.remote.functions.group_reduce(self.remote.obs, KEYS, AGGS)
        rows = [tuple(r) for r in view.limit(2, 1)["summary"].iterdata()]
        self.assertEqual([r[0] for r in rows], ["s2", "s3"])
        zero = list(view.limit(0)["summary"].iterdata())
        self.assertEqual(zero, [])
        for bad in (True, False, -1, 1.5):
            with self.assertRaises(TypeError):
                view.limit(bad)
        with self.assertRaises(TypeError):
            view.limit(1, True)

    def test_operators_and_constants(self):
        view = self.remote.functions.group_reduce(self.remote.obs, KEYS, AGGS)
        for op, expected in [
            ("=", ["s2"]),
            ("!=", ["s1", "s3"]),
            ("<", ["s1"]),
            ("<=", ["s1", "s2"]),
            (">", ["s3"]),
            (">=", ["s2", "s3"]),
        ]:
            rows = list(
                view.where(result_condition("site", op, "s2"))[
                    "summary"
                ].iterdata()
            )
            self.assertEqual([r[0] for r in rows], expected, op)
        with self.assertRaises(TypeError):
            result_condition("site", "=", True)
        with self.assertRaises(ValueError):
            result_condition("mean_v", "~", 1)

    def test_negative_and_signed_keys(self):
        rows = [("k", -7.0), ("k", -3.0), ("k", 4.0)]
        dataset = build_dataset(rows)
        app = ServerSideFunctions(BaseHandler(dataset))
        remote = open_url("http://localhost/", application=app)
        view = remote.functions.group_reduce(remote.obs, KEYS, AGGS)
        row = list(view["summary"].iterdata())[0]
        self.assertEqual(row[1], 3)
        self.assertTrue(math.isclose(row[2], -6.0, rel_tol=1e-10, abs_tol=1e-12))
        self.assertEqual(row[4], -7.0)
        self.assertEqual(row[5], 4.0)


class WireProtocol(unittest.TestCase):
    """The on-the-wire CE must be replayable by hand and fully quoted."""

    def test_quote_aware_tokenizer(self):
        ce = 'where(group_reduce(obs,"[\\"a\\"]","[]"),"{\\"k\\":\\"x,y(z)&w\\"}")'
        name, args = parse_call(ce)
        self.assertEqual(name, "where")
        self.assertEqual(len(args), 2)
        nested, payload = args
        self.assertTrue(nested.startswith("group_reduce("))
        decoded = json.loads(decode_dap_string(payload))
        self.assertEqual(decoded["k"], "x,y(z)&w")

    def test_round_trip_special_characters(self):
        special = 'a,b(c)d"e\\f&g/hé日本'
        literal = encode_dap_string(compact_json({"v": special}))
        self.assertEqual(json.loads(decode_dap_string(literal))["v"], special)
        # The encoded literal contains no raw top-level separators.
        self.assertEqual(split_top_level(literal, ","), [literal])
        # Ampersand inside a literal must not split a CE.
        ce = "f(" + literal + ")"
        self.assertEqual(split_ce(ce + "&g(1)"), [ce, "g(1)"])

    def test_hand_written_http_request_matches_client(self):
        rows = [("a,b", 1.0), ("a,b", 3.0), ("x(y)", 2.0), ("x(y)", 3.0)]
        dataset = build_dataset(rows, dtypes=[("site", "U16"), ("v", "f8")])
        application = ServerSideFunctions(BaseHandler(dataset))
        remote = open_url("http://localhost/", application=application)
        client_view = remote.functions.group_reduce(remote.obs, KEYS, AGGS)
        client_rows = [tuple(r) for r in client_view["summary"].iterdata()]

        # Build exactly the same CE by hand and send it as raw HTTP.
        hand_ce = group_ce()
        hand_dataset = open_dods_url(
            "http://localhost/.dods?" + encode_ce(hand_ce),
            application=application,
        )
        hand_rows = [tuple(r) for r in hand_dataset["summary"].iterdata()]
        self.assertEqual(client_rows, hand_rows)

        # Staged pipeline via raw HTTP.
        predicate = dap_string(
            compact_json(result_condition("mean_v", ">", 2.0))
        )
        staged = "where(%s,%s)" % (hand_ce, predicate)
        staged_rows = [
            tuple(r)
            for r in open_dods_url(
                "http://localhost/.dods?" + encode_ce(staged),
                application=application,
            )["summary"].iterdata()
        ]
        self.assertEqual([r[0] for r in staged_rows], ["x(y)"])

    def test_dds_matches_dods_columns(self):
        dataset = build_dataset([("a", 1.0), ("b", 2.0)])
        app = make_app(dataset)
        ce = group_ce()
        dds = app.get("/.dds?" + encode_ce(ce)).text
        self.assertIn("String site;", dds)
        self.assertIn("Int32 n;", dds)
        self.assertIn("Float64 mean_v;", dds)
        dods_dataset = open_dods_url(
            "http://localhost/.dods?" + encode_ce(ce),
            application=ServerSideFunctions(BaseHandler(dataset)),
        )
        self.assertEqual(
            list(dods_dataset["summary"].keys()),
            ["site", "n", "sum_v", "mean_v", "min_v", "max_v"],
        )


class DescriptorValidation(unittest.TestCase):
    """Errors that can be inferred from the description fail pre-emptively."""

    def setUp(self):
        self.dataset = build_dataset([("a", 1.0)])
        self.app = make_app(self.dataset)

    def _request(self, keys, aggregates):
        ce = "group_reduce(obs,%s,%s)" % (
            dap_string(keys if isinstance(keys, str) else compact_json(keys)),
            dap_string(
                aggregates
                if isinstance(aggregates, str)
                else compact_json(aggregates)
            ),
        )
        return self.app.get(
            "/.dds?" + encode_ce(ce), expect_errors=True
        )

    def test_bad_json(self):
        res = self._request("not json", "[]")
        self.assertEqual(res.status_int, 500)
        self.assertIn("not valid JSON", res.text)

    def test_empty_keys_and_aggregates(self):
        self.assertIn("non-empty", self._request([], []).text)
        self.assertIn(
            "non-empty", self._request(["site"], []).text
        )

    def test_duplicate_and_unknown_fields(self):
        self.assertIn(
            "duplicate keys",
            self._request(["site", "site"], [{"name": "n", "op": "count", "field": None}]).text,
        )
        self.assertIn(
            "unknown grouping field",
            self._request(["missing"], [{"name": "n", "op": "count", "field": None}]).text,
        )
        self.assertIn(
            "unknown field",
            self._request(
                ["site"], [{"name": "m", "op": "mean", "field": "missing"}]
            ).text,
        )

    def test_name_conflicts_and_duplicates(self):
        self.assertIn(
            "must not clash",
            self._request(
                ["site"], [{"name": "site", "op": "count", "field": None}]
            ).text,
        )
        self.assertIn(
            "unique",
            self._request(
                ["site"],
                [
                    {"name": "x", "op": "count", "field": None},
                    {"name": "x", "op": "count", "field": None},
                ],
            ).text,
        )

    def test_unknown_op_and_count_field(self):
        self.assertIn(
            "unknown op",
            self._request(
                ["site"], [{"name": "x", "op": "median", "field": "v"}]
            ).text,
        )
        self.assertIn(
            "field as null",
            self._request(
                ["site"], [{"name": "x", "op": "count", "field": "v"}]
            ).text,
        )

    def test_validation_happens_before_data_read(self):
        # A descriptor error must surface even on a DDS request, where the
        # sequence data would otherwise never be iterated.
        res = self._request(
            ["site"], [{"name": "m", "op": "mean", "field": "nope"}]
        )
        self.assertEqual(res.status_int, 500)


class DataValidationAndResources(unittest.TestCase):
    def _app_with(self, values, dtype="f8"):
        dataset = DatasetType("t")
        sequence = SequenceType("obs")
        sequence["site"] = BaseType("site")
        sequence["v"] = BaseType("v")
        sequence.data = np.rec.fromarrays(
            [
                np.array(["s%02d" % i for i in range(len(values))], dtype="U8"),
                np.array(values, dtype=dtype),
            ],
            names=["site", "v"],
        )
        dataset["obs"] = sequence
        return ServerSideFunctions(BaseHandler(dataset))

    def test_non_finite_and_bool_rejected(self):
        app = App(self._app_with([1.0, np.nan]))
        res = app.get(
            "/.asc?" + encode_ce(group_ce()), expect_errors=True
        )
        self.assertEqual(res.status_int, 500)
        self.assertIn("'v'", res.text)

        app = App(self._app_with([1.0, math.inf]))
        res = app.get(
            "/.asc?" + encode_ce(group_ce()), expect_errors=True
        )
        self.assertEqual(res.status_int, 500)

    def test_partial_failure_never_succeeds(self):
        # The bad row is the last one; no partial summary may be emitted.
        app = App(self._app_with([1.0, 2.0, np.nan]))
        res = app.get(
            "/.dods?" + encode_ce(group_ce()), expect_errors=True
        )
        self.assertEqual(res.status_int, 500)
        self.assertNotIn(b"Data:", res.body)

    def test_temp_files_released_on_error_and_close(self):
        tempdir = tempfile.mkdtemp()
        application = self._app_with(
            [1.0, np.nan] * 50
        )
        application.group_config = GroupReduceConfig(
            group_reduce_max_groups=4, group_reduce_tempdir=tempdir
        )
        app = App(application)
        app.get("/.dods?" + encode_ce(group_ce()), expect_errors=True)
        leftovers = [
            path
            for path in self._walk(tempdir)
            if os.path.isfile(path)
        ]
        self.assertEqual(leftovers, [])

        # Early client close also releases everything.
        good = DatasetType("t")
        seq = SequenceType("obs")
        seq["site"] = BaseType("site"); seq["v"] = BaseType("v")
        seq.data = np.rec.fromarrays(
            [
                np.array(["k%03d" % (i % 40) for i in range(200)], dtype="U8"),
                np.arange(200, dtype="f8"),
            ],
            names=["site", "v"],
        )
        good["obs"] = seq
        good_app = ServerSideFunctions(
            BaseHandler(good),
            group_reduce_max_groups=4,
            group_reduce_tempdir=tempdir,
        )
        remote = open_url("http://localhost/", application=good_app)
        view = remote.functions.group_reduce(remote.obs, KEYS, AGGS)
        iterator = iter(view["summary"].iterdata())
        next(iterator)
        iterator.close()
        leftovers = [
            path for path in self._walk(tempdir) if os.path.isfile(path)
        ]
        self.assertEqual(leftovers, [])

    @staticmethod
    def _walk(directory):
        for root, _, files in os.walk(directory):
            for name in files:
                yield os.path.join(root, name)

    def test_single_source_traversal(self):
        traversals = {"count": 0}

        class CountingSequence(SequenceType):
            def iterdata(self):
                traversals["count"] += 1
                yield from super().iterdata()

        dataset = DatasetType("t")
        sequence = CountingSequence("obs")
        sequence["site"] = BaseType("site"); sequence["v"] = BaseType("v")
        sequence.data = np.rec.fromarrays(
            [
                np.array(["a", "b", "a", "c"], dtype="U8"),
                np.array([1.0, 2.0, 3.0, 4.0]),
            ],
            names=["site", "v"],
        )
        dataset["obs"] = sequence
        application = ServerSideFunctions(
            BaseHandler(dataset), group_reduce_max_groups=1
        )
        remote = open_url("http://localhost/", application=application)
        view = remote.functions.group_reduce(remote.obs, KEYS, AGGS)
        rows = list(view["summary"].iterdata())
        self.assertEqual(len(rows), 3)
        self.assertEqual(traversals["count"], 1)

    def test_empty_input_keeps_full_description(self):
        dataset = DatasetType("empty")
        sequence = SequenceType("obs")
        sequence["site"] = BaseType("site", dtype=np.dtype("U8"))
        sequence["v"] = BaseType("v", dtype=np.dtype("f8"))
        sequence.data = np.rec.fromarrays(
            [np.array([], dtype="U8"), np.array([], dtype="f8")],
            names=["site", "v"],
        )
        dataset["obs"] = sequence
        application = ServerSideFunctions(BaseHandler(dataset))
        app = App(application)
        dds = app.get("/.dds?" + encode_ce(group_ce())).text
        self.assertIn("Sequence {", dds)
        self.assertIn("String site;", dds)
        remote = open_url("http://localhost/", application=application)
        view = remote.functions.group_reduce(remote.obs, KEYS, AGGS)
        self.assertEqual(list(view["summary"].iterdata()), [])


class EngineInternals(unittest.TestCase):
    def test_config_validation(self):
        with self.assertRaises(TypeError):
            GroupReduceConfig(group_reduce_max_groups=True)
        with self.assertRaises(ValueError):
            GroupReduceConfig(group_reduce_max_groups=0)
        with self.assertRaises(ValueError):
            GroupReduceConfig(group_reduce_tempdir="/no/such/dir")
        config = GroupReduceConfig()
        self.assertEqual(config.group_reduce_max_groups, 1024)
        self.assertIsNone(config.group_reduce_tempdir)

    def test_kwargs_registration_still_works(self):
        def double(dataset, var):
            return var

        middleware = ServerSideFunctions(BaseHandler(DatasetType("x")), double=double)
        self.assertIn("double", middleware.functions)

    def test_spill_residency_bounded(self):
        dataset = DatasetType("t")
        sequence = SequenceType("obs")
        sequence["g"] = BaseType("g"); sequence["v"] = BaseType("v")
        sequence.data = np.rec.fromarrays(
            [np.arange(100, dtype="i4"), np.arange(100, dtype="f8")],
            names=["g", "v"],
        )
        dataset["obs"] = sequence
        result = group_reduce_func(
            dataset,
            sequence,
            compact_json(["g"]),
            compact_json(
                [
                    {"name": "n", "op": "count", "field": None},
                    {"name": "s", "op": "sum", "field": "v"},
                ]
            ),
            config=GroupReduceConfig(group_reduce_max_groups=8),
        )
        engine = result._group_reduce_engine
        rows = list(result.summary.iterdata())
        self.assertEqual(len(rows), 100)
        self.assertLessEqual(engine.peak_resident_groups, 8)
        self.assertGreater(engine.spill_count, 0)
        self.assertGreater(engine.peak_disk_bytes, 0)
        self.assertEqual(engine.source_rows, 100)
        # After full consumption the temp directory must be gone.
        self.assertIsNone(engine._temp_dir_obj)

    def test_unsupported_column_type(self):
        dataset = DatasetType("t")
        sequence = SequenceType("obs")
        sequence["flag"] = BaseType("flag", dtype=np.dtype("?"))
        sequence.data = np.array([(True,)], dtype=[("flag", "?")])
        dataset["obs"] = sequence
        application = ServerSideFunctions(BaseHandler(dataset))
        app = App(application)
        ce = "group_reduce(obs,%s,%s)" % (
            dap_string(compact_json(["flag"])),
            dap_string(compact_json([{"name": "n", "op": "count", "field": None}])),
        )
        res = app.get("/.dds?" + encode_ce(ce), expect_errors=True)
        self.assertEqual(res.status_int, 500)


class ClientViewBehaviour(unittest.TestCase):
    def setUp(self):
        dataset = build_dataset(
            [("a", 1.0), ("a", 3.0), ("b", 10.0), ("b", 20.0)]
        )
        self.application = ServerSideFunctions(BaseHandler(dataset))
        self.remote = open_url(
            "http://localhost/", application=self.application
        )
        self.view = self.remote.functions.group_reduce(
            self.remote.obs, KEYS, AGGS
        )

    def test_views_are_lazy_and_independent(self):
        derived = self.view.where(result_condition("mean_v", ">", 5))
        self.assertIsInstance(derived, LazyFunctionResult)
        self.assertIsNone(derived.dataset)
        self.assertIsNot(derived, self.view)
        # Materialising one view must not populate the other.
        list(derived["summary"].iterdata())
        self.assertIsNone(self.view.dataset)

    def test_materialised_view_uses_cache(self):
        first = list(self.view["summary"].iterdata())
        dataset_object = self.view.dataset
        second = list(self.view["summary"].iterdata())
        self.assertEqual(first, second)
        self.assertIs(self.view.dataset, dataset_object)

    def test_two_views_do_not_share_results(self):
        view_a = self.view.select("site", "n")
        view_b = self.view.limit(1)
        list(view_a["summary"].iterdata())
        list(view_b["summary"].iterdata())
        self.assertIsNot(view_a.dataset, view_b.dataset)
        self.assertIsNot(view_a.dataset, self.view.dataset)

    def test_legacy_function_result_field_access(self):
        # Ordinary server functions still expose fields directly.
        from pydap.client import ServerFunctionResult

        result = ServerFunctionResult(
            "http://localhost/x", "mean(x)", application=self.application
        )
        self.assertTrue(hasattr(result, "url"))

    def test_interleaved_reads_of_two_views(self):
        view_a = self.view.select("site", "n")
        view_b = self.view.limit(1)
        iter_a = iter(view_a["summary"].iterdata())
        iter_b = iter(view_b["summary"].iterdata())
        first_a = next(iter_a)
        only_b = next(iter_b)
        second_a = next(iter_a)
        # Neither view's state leaks into the other.
        self.assertEqual(only_b[0], "a")
        self.assertEqual(first_a[0], "a")
        # view_a carries every group, so its second row is the next key.
        self.assertEqual(second_a[0], "b")
        with self.assertRaises(StopIteration):
            next(iter_b)
        self.assertIsNot(view_a.dataset, view_b.dataset)


class StagedWireResponses(unittest.TestCase):
    def setUp(self):
        rows = [
            ("a", 1.0), ("a", 4.0),
            ("b", 2.0), ("b", 8.0),
            ("c", -3.0),
        ]
        self.dataset = build_dataset(rows)
        self.application = ServerSideFunctions(BaseHandler(self.dataset))
        self.app = App(self.application)

    def test_dds_and_dods_share_stage_result(self):
        ce = "select(where(%s,%s),%s)" % (
            group_ce(),
            dap_string(
                compact_json(result_condition("mean_v", ">=", 2.5))
            ),
            dap_string(compact_json(["site", "n"])),
        )
        encoded = encode_ce(ce)
        dds = self.app.get("/.dds?" + encoded).text
        # The description only carries the projected columns.
        self.assertIn("String site;", dds)
        self.assertIn("Int32 n;", dds)
        self.assertNotIn("Float64 mean_v;", dds)
        dods = open_dods_url(
            "http://localhost/.dods?" + encoded, application=self.application
        )
        self.assertEqual(list(dods["summary"].keys()), ["site", "n"])
        rows = [tuple(r) for r in dods["summary"].iterdata()]
        # Independent truth: means are a=2.5, b=5.0, c=-3.0; >=2.5 keeps a,b.
        self.assertEqual([r[0] for r in rows], ["a", "b"])

    def test_limit_over_raw_http(self):
        ce = "limit(%s,2,1)" % group_ce()
        dods = open_dods_url(
            "http://localhost/.dods?" + encode_ce(ce),
            application=self.application,
        )
        rows = [tuple(r) for r in dods["summary"].iterdata()]
        self.assertEqual([r[0] for r in rows], ["b", "c"])

    def test_multi_key_raw_http(self):
        rows = [
            ("north", "x", 1.0), ("north", "x", 3.0),
            ("south", "x", 5.0), ("north", "y", 7.0),
        ]
        dataset = DatasetType("t")
        sequence = SequenceType("obs")
        for name in ("region", "site", "v"):
            sequence[name] = BaseType(name)
        sequence.data = np.array(
            rows, dtype=[("region", "U16"), ("site", "U8"), ("v", "f8")]
        )
        dataset["obs"] = sequence
        application = ServerSideFunctions(BaseHandler(dataset))
        keys = compact_json(["region", "site"])
        aggregates = compact_json(
            [{"name": "n", "op": "count", "field": None}]
        )
        ce = "group_reduce(obs,%s,%s)" % (
            dap_string(keys),
            dap_string(aggregates),
        )
        dods = open_dods_url(
            "http://localhost/.dods?" + encode_ce(ce), application=application
        )
        got = [tuple(r) for r in dods["summary"].iterdata()]
        truth = sorted({(r, s) for r, s, _ in rows})
        self.assertEqual([(g[0], g[1]) for g in got], truth)
        self.assertEqual([g[2] for g in got], [2, 1, 1])

    def test_type_mismatch_rejected(self):
        ce = "where(%s,%s)" % (
            group_ce(),
            dap_string(
                compact_json(result_condition("n", "=", "not-a-number"))
            ),
        )
        res = self.app.get(
            "/.dds?" + encode_ce(ce), expect_errors=True
        )
        self.assertEqual(res.status_int, 500)


if __name__ == "__main__":
    unittest.main()
