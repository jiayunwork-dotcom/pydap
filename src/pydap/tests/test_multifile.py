"""Tests for the cross-file DAP4 view (``open_mfdataset``).

All sources are served by the in-process :mod:`pydap.tests.dap4_test_server`
protocol double, so these tests never touch a public OPeNDAP service. The
server records every request, allowing assertions on laziness (no full
granule downloads) and exact constraint ranges.
"""

import os
import tempfile

import numpy as np
import pytest
from netCDF4 import Dataset

from pydap.client import open_mfdataset, to_netcdf
from pydap.exceptions import (
    BufferLimitExceededError,
    DAP4ChecksumError,
    DAP4TruncatedResponseError,
)
from pydap.multifile import MultiFileProxy
from pydap.tests.dap4_test_server import DAP4TestServer, Granule

# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------


def granule(name, times, temp_values, dims=("/time", "/x"), coord_x=None, **attrs):
    times = np.asarray(times, dtype=np.int64)
    n = times.size
    if coord_x is None:
        coord_x = np.array([10, 20, 30], dtype=np.float64)
    nx = coord_x.size
    temp = np.asarray(temp_values, dtype=np.float32).reshape(n, nx)
    if dims == ("/x", "/time"):
        temp = np.transpose(temp, (1, 0)).copy()
    return Granule(
        name=name,
        dimensions={"/time": n, "/x": nx},
        variables={
            "/time": (times, ("/time",), {"units": "seconds"}),
            "/x": (coord_x, ("/x",), {"units": "m"}),
            "/temp": (temp, dims, {"units": "K", **attrs}),
        },
        attributes={"/": {"collection": "synthetic"}},
    )


@pytest.fixture
def two_granules():
    g1 = granule("g1", [1, 2, 3], [[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    g2 = granule("g2", [4, 5], [[100, 101, 102], [103, 104, 105]])
    with DAP4TestServer([g1, g2]) as server:
        yield server, [g1, g2]


def _urls(server, granules):
    return [server.url_for(g.name) for g in granules]


def _full(granules, path):
    return np.concatenate([g.array_for(path) for g in granules], axis=0)


# ---------------------------------------------------------------------------
# Offline validation (no network access must occur)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(urls=[], concat_dim="/time", keep_variables=["/temp"]),
        dict(
            urls=["http://h/a", "http://h/a"],
            concat_dim="/time",
            keep_variables=["/temp"],
        ),
        dict(
            urls=["http://h/a?dap4.ce=/x"], concat_dim="/time", keep_variables=["/temp"]
        ),
        dict(urls=["ftp://h/a"], concat_dim="/time", keep_variables=["/temp"]),
        dict(urls=["http://h/a#f"], concat_dim="/time", keep_variables=["/temp"]),
        dict(urls=["http://h/a"], concat_dim="time", keep_variables=["/temp"]),
        dict(urls=["http://h/a"], concat_dim="/time", keep_variables=[]),
        dict(
            urls=["http://h/a"], concat_dim="/time", keep_variables=["/temp", "/temp"]
        ),
        dict(urls=["http://h/a"], concat_dim="/time", keep_variables=["temp"]),
    ],
)
def test_offline_validation(kwargs):
    with pytest.raises((ValueError, TypeError)):
        open_mfdataset(**kwargs)


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "8"])
def test_buffer_bytes_validation(value):
    with pytest.raises((ValueError, TypeError)):
        open_mfdataset(
            ["http://h/a"],
            concat_dim="/time",
            keep_variables=["/temp"],
            max_buffer_bytes=value,
        )


def test_overlap_validation():
    with pytest.raises(ValueError):
        open_mfdataset(
            ["http://h/a"],
            concat_dim="/time",
            keep_variables=["/temp"],
            overlap="middle",
        )


# ---------------------------------------------------------------------------
# Basic view semantics
# ---------------------------------------------------------------------------


def test_open_view_metadata(two_granules):
    server, granules = two_granules
    view = open_mfdataset(
        _urls(server, granules), concat_dim="/time", keep_variables=["/temp"]
    )
    assert view.dimensions == {"time": 5, "x": 3}
    var = view["/temp"]
    assert var.dtype == np.float32
    assert var.shape == (5, 3)
    assert var.dims == ["/time", "/x"]
    assert isinstance(var.data, MultiFileProxy)
    # Only DMR + coordinate/static fetches happened; no dynamic /temp request.
    assert not any("/temp" in p for p in server.dap_paths)
    view.close()
    with pytest.raises(RuntimeError):
        view["/temp"][:].data


def test_view_context_manager_and_idempotent_close(two_granules):
    server, granules = two_granules
    with open_mfdataset(
        _urls(server, granules), concat_dim="/time", keep_variables=["/temp"]
    ) as view:
        assert view["/temp"].shape == (5, 3)
    view.close()  # idempotent
    with pytest.raises(RuntimeError):
        view["/temp"][:].data


def test_slices_match_full_merge(two_granules):
    server, granules = two_granules
    view = open_mfdataset(
        _urls(server, granules), concat_dim="/time", keep_variables=["/temp", "/x"]
    )
    full = _full(granules, "/temp")
    checks = [
        (slice(None), slice(None)),
        (slice(1, 4), slice(0, 2)),
        (slice(0, 5, 2), slice(None)),
        (slice(None, None, -1), slice(None)),
        (3, slice(None)),
        (-1, -1),
        (slice(-3, -1), slice(1, 3)),
        (slice(0, 0), slice(None)),
        (slice(None, None, 2), slice(0, 3, 2)),
    ]
    for key in checks:
        got = np.asarray(view["/temp"][key].data)
        np.testing.assert_array_equal(got, full[key])
    view.close()


def test_step_zero_rejected(two_granules):
    server, granules = two_granules
    view = open_mfdataset(
        _urls(server, granules), concat_dim="/time", keep_variables=["/temp"]
    )
    with pytest.raises(ValueError):
        view["/temp"][::0]
    view.close()


def test_advanced_index_rejected(two_granules):
    server, granules = two_granules
    view = open_mfdataset(
        _urls(server, granules), concat_dim="/time", keep_variables=["/temp"]
    )
    with pytest.raises(IndexError):
        view["/temp"][[0, 2]]
    with pytest.raises(IndexError):
        view["/temp"][np.array([True, False, True, False, True])]
    view.close()


def test_out_of_bounds_integer(two_granules):
    server, granules = two_granules
    view = open_mfdataset(
        _urls(server, granules), concat_dim="/time", keep_variables=["/temp"]
    )
    with pytest.raises(IndexError):
        view["/temp"][5]
    view.close()


def test_lazy_request_ranges(two_granules):
    server, granules = two_granules
    view = open_mfdataset(
        _urls(server, granules), concat_dim="/time", keep_variables=["/temp"]
    )
    server.reset_log()
    np.asarray(view["/temp"][1:4].data)
    # Exactly one constrained request per contributing granule; never the
    # full variable (full time extent would be [0:1:2] for g1).
    dap = server.dap_paths
    assert len(dap) == 2
    assert any("g1" in p and "temp%5B1:1:2%5D" in p for p in dap)
    assert any("g2" in p and "temp%5B0:1:0%5D" in p for p in dap)
    assert not any("temp%5B0:1:2%5D" in p for p in dap)
    view.close()


def test_url_order_shuffled():
    g1 = granule("g1", [1, 2], [[0, 1, 2], [3, 4, 5]])
    g2 = granule("g2", [3, 4], [[100, 101, 102], [103, 104, 105]])
    with DAP4TestServer([g1, g2]) as server:
        # Pass URLs in reverse order; coordinate merge still ascending and
        # ownership follows URL order.
        view = open_mfdataset(
            [server.url_for("g2"), server.url_for("g1")],
            concat_dim="/time",
            keep_variables=["/temp"],
        )
        np.testing.assert_array_equal(view["/time"].data, [1, 2, 3, 4])
        np.testing.assert_array_equal(
            np.asarray(view["/temp"][:].data),
            np.concatenate([g1.array_for("/temp"), g2.array_for("/temp")]),
        )
        view.close()


def test_descending_coordinates():
    g1 = granule("g1", [30, 20, 10], [[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    g2 = granule("g2", [9, 8], [[4, 4, 4], [5, 5, 5]])
    with DAP4TestServer([g1, g2]) as server:
        view = open_mfdataset(
            _urls(server, [g1, g2]), concat_dim="/time", keep_variables=["/temp"]
        )
        np.testing.assert_array_equal(view["/time"].data, [8, 9, 10, 20, 30])
        # Values follow their owning coordinate, not granule order.
        got = np.asarray(view["/temp"][:].data)[:, 0]
        np.testing.assert_array_equal(got, [5, 4, 3, 2, 1])
        view.close()


# ---------------------------------------------------------------------------
# Overlap policies
# ---------------------------------------------------------------------------


def _overlap_granules(server_cls=None):
    g1 = granule("g1", [1, 2, 3], [[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    g2 = granule("g2", [3, 4], [[20, 20, 20], [30, 30, 30]])
    return g1, g2


def test_overlap_error():
    g1, g2 = _overlap_granules()
    with DAP4TestServer([g1, g2]) as server:
        with pytest.raises(ValueError, match="Duplicate concat coordinate"):
            open_mfdataset(
                _urls(server, [g1, g2]),
                concat_dim="/time",
                keep_variables=["/temp"],
            )


@pytest.mark.parametrize("strategy,expected_row3", [("first", 2.0), ("last", 20.0)])
def test_overlap_resolution_is_whole_variable(strategy, expected_row3):
    g1, g2 = _overlap_granules()
    with DAP4TestServer([g1, g2]) as server:
        view = open_mfdataset(
            _urls(server, [g1, g2]),
            concat_dim="/time",
            keep_variables=["/temp"],
            overlap=strategy,
        )
        np.testing.assert_array_equal(view["/time"].data, [1, 2, 3, 4])
        data = np.asarray(view["/temp"][:].data)
        # The chosen source decides the whole row (all kept variables).
        np.testing.assert_array_equal(data[2], np.full(3, expected_row3))
        view.close()


# ---------------------------------------------------------------------------
# Big integers, empty granules/slices, missing values
# ---------------------------------------------------------------------------


def test_big_integers_distinguished():
    big = 2**53 + 5
    g1 = Granule(
        "g1",
        dimensions={"/time": 1},
        variables={
            "/time": (np.array([big], dtype=np.int64), ("/time",), {}),
            "/v": (np.array([[1.0]], dtype=np.float32), ("/time",), {}),
        },
    )
    g2 = Granule(
        "g2",
        dimensions={"/time": 2},
        variables={
            # big+1 is not representably distinct from big once cast to float
            "/time": (np.array([big + 1, big + 2], dtype=np.int64), ("/time",), {}),
            "/v": (np.array([[2.0], [3.0]], dtype=np.float32), ("/time",), {}),
        },
    )
    with DAP4TestServer([g1, g2]) as server:
        view = open_mfdataset(
            _urls(server, [g1, g2]), concat_dim="/time", keep_variables=["/v"]
        )
        coords = view["/time"].data
        assert coords[0] == big and coords[1] == big + 1 and coords[2] == big + 2
        view.close()


def test_empty_granule_and_empty_slice():
    g1 = granule("g1", [], np.zeros((0, 3), dtype=np.float32))
    g2 = granule("g2", [1, 2], [[7, 8, 9], [10, 11, 12]])
    with DAP4TestServer([g1, g2]) as server:
        view = open_mfdataset(
            _urls(server, [g1, g2]), concat_dim="/time", keep_variables=["/temp"]
        )
        assert view["/temp"].shape == (2, 3)
        empty = np.asarray(view["/temp"][0:0].data)
        assert empty.shape == (0, 3)
        np.testing.assert_array_equal(
            np.asarray(view["/temp"][:].data), g2.array_for("/temp")
        )
        view.close()


def test_fill_value_preserved():
    fill = np.float32(-999.0)
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[fill, 6.0], [7.0, fill]], dtype=np.float32)
    g1 = Granule(
        "g1",
        dimensions={"/time": 2, "/x": 2},
        variables={
            "/time": (np.array([1, 2], dtype=np.int64), ("/time",), {}),
            "/x": (np.array([0, 1], dtype=np.int64), ("/x",), {}),
            "/v": (a, ("/time", "/x"), {"_FillValue": fill}),
        },
    )
    g2 = Granule(
        "g2",
        dimensions={"/time": 2, "/x": 2},
        variables={
            "/time": (np.array([3, 4], dtype=np.int64), ("/time",), {}),
            "/x": (np.array([0, 1], dtype=np.int64), ("/x",), {}),
            "/v": (b, ("/time", "/x"), {"_FillValue": fill}),
        },
    )
    with DAP4TestServer([g1, g2]) as server:
        view = open_mfdataset(
            _urls(server, [g1, g2]), concat_dim="/time", keep_variables=["/v"]
        )
        assert view["/v"].attributes["_FillValue"] == fill
        got = np.asarray(view["/v"][:].data)
        # Raw numeric semantics: the fill marker value is preserved untouched
        # (no CF decoding into NaN).
        np.testing.assert_array_equal(got, np.concatenate([a, b]))
        view.close()


# ---------------------------------------------------------------------------
# Groups, axis reorder, static variables, conflicts
# ---------------------------------------------------------------------------


def _group_granule(name, times, base, order):
    n = len(times)
    temp = np.arange(n * 2 * 2, dtype=np.float32).reshape(n, 2, 2) + base
    if order == ("/g/y", "/time", "/g/x"):
        temp = np.transpose(temp, (1, 0, 2)).copy()
    return Granule(
        name=name,
        dimensions={"/time": n, "/g/x": 2, "/g/y": 2},
        variables={
            "/time": (np.asarray(times, dtype=np.int64), ("/time",), {}),
            "/g/x": (np.array([10, 11], dtype=np.float64), ("/g/x",), {}),
            "/g/y": (np.array([20, 21], dtype=np.float64), ("/g/y",), {}),
            "/g/temp": (temp, order, {"units": "K"}),
            "/g/static": (np.array([42], dtype=np.int32), (), {"tag": "same"}),
        },
    )


def test_group_axis_reorder_and_static():
    g1 = _group_granule("g1", [1, 2], 0, ("/time", "/g/y", "/g/x"))
    g2 = _group_granule("g2", [3, 4], 100, ("/g/y", "/time", "/g/x"))
    with DAP4TestServer([g1, g2]) as server:
        view = open_mfdataset(
            _urls(server, [g1, g2]),
            concat_dim="/time",
            keep_variables=["/g/temp", "/g/static"],
        )
        # Output axis order follows first URL.
        assert view["/g/temp"].dims == ["/time", "/g/y", "/g/x"]
        assert view["/g/temp"].shape == (4, 2, 2)
        full = np.concatenate(
            [g1.array_for("/g/temp"), np.transpose(g2.array_for("/g/temp"), (1, 0, 2))],
            axis=0,
        )
        got = np.asarray(view["/g/temp"][1:3, 0:2, 1:2].data)
        np.testing.assert_array_equal(got, full[1:3, 0:2, 1:2])
        np.testing.assert_array_equal(
            np.asarray(view["/g/static"].data), np.array([42], dtype=np.int32)
        )
        view.close()


def test_group_same_named_dimensions_not_mixed():
    # Two groups each declare a dimension 'x' with a different length.
    def build(name, nx_a, nx_b):
        return Granule(
            name=name,
            dimensions={"/time": 1, "/a/x": nx_a, "/b/x": nx_b},
            variables={
                "/time": (np.array([1], dtype=np.int64), ("/time",), {}),
                "/a/x": (np.arange(nx_a, dtype=np.float64), ("/a/x",), {}),
                "/b/x": (np.arange(nx_b, dtype=np.float64), ("/b/x",), {}),
                "/a/v": (np.ones((1, nx_a), dtype=np.float32), ("/time", "/a/x"), {}),
                "/b/v": (np.zeros((1, nx_b), dtype=np.float32), ("/time", "/b/x"), {}),
            },
        )

    g1 = build("g1", 2, 3)
    g2 = build("g2", 2, 4)  # /b/x length mismatch
    with DAP4TestServer([g1, g2]) as server:
        with pytest.raises(ValueError, match="/b/x"):
            open_mfdataset(
                _urls(server, [g1, g2]),
                concat_dim="/time",
                keep_variables=["/a/v", "/b/v"],
            )


def test_spatial_coordinate_conflict():
    g1 = granule("g1", [1], [[0, 1, 2]], coord_x=np.array([10.0, 20.0, 30.0]))
    g2 = granule("g2", [2], [[3, 4, 5]], coord_x=np.array([10.0, 20.0, 99.0]))
    with DAP4TestServer([g1, g2]) as server:
        with pytest.raises(ValueError, match="Coordinate values"):
            open_mfdataset(
                _urls(server, [g1, g2]),
                concat_dim="/time",
                keep_variables=["/temp"],
            )


def test_static_value_conflict():
    g1 = _group_granule("g1", [1], 0, ("/time", "/g/y", "/g/x"))
    g2 = _group_granule("g2", [2], 100, ("/time", "/g/y", "/g/x"))
    g2.variables["/g/static"] = (
        np.array([7], dtype=np.int32),
        (),
        {"tag": "different"},
    )
    with DAP4TestServer([g1, g2]) as server:
        with pytest.raises(ValueError, match="Static variable"):
            open_mfdataset(
                _urls(server, [g1, g2]),
                concat_dim="/time",
                keep_variables=["/g/temp", "/g/static"],
            )


def test_missing_variable_rejected_at_open():
    g1 = granule("g1", [1], [[0, 1, 2]])
    g2 = Granule(
        "g2",
        dimensions={"/time": 1},
        variables={
            "/time": (np.array([2], dtype=np.int64), ("/time",), {"units": "seconds"}),
            # /temp deliberately absent
        },
    )
    with DAP4TestServer([g1, g2]) as server:
        with pytest.raises(ValueError, match="missing from granule"):
            open_mfdataset(
                _urls(server, [g1, g2]),
                concat_dim="/time",
                keep_variables=["/temp"],
            )


def test_dtype_conflict_rejected():
    g1 = granule("g1", [1], [[0, 1, 2]])
    g2 = Granule(
        "g2",
        dimensions={"/time": 1, "/x": 3},
        variables={
            "/time": (np.array([2], dtype=np.int64), ("/time",), {"units": "seconds"}),
            "/x": (np.array([10, 20, 30], dtype=np.float64), ("/x",), {}),
            "/temp": (
                np.array([[3, 4, 5]], dtype=np.float64),
                ("/time", "/x"),
                {"units": "K"},
            ),
        },
    )
    with DAP4TestServer([g1, g2]) as server:
        with pytest.raises(ValueError, match="dtype mismatch"):
            open_mfdataset(
                _urls(server, [g1, g2]),
                concat_dim="/time",
                keep_variables=["/temp"],
            )


# ---------------------------------------------------------------------------
# Buffer accounting
# ---------------------------------------------------------------------------


def test_buffer_budget_enforced_and_bounded():
    # 40 time rows x 4 columns of float32 = 640 bytes total dynamic data.
    n = 20
    data1 = np.arange(n * 4, dtype=np.float32).reshape(n, 4)
    data2 = data1 + 1000
    g1 = Granule(
        "g1",
        dimensions={"/time": n, "/x": 4},
        variables={
            "/time": (np.arange(1, n + 1, dtype=np.int64), ("/time",), {}),
            "/x": (np.arange(4, dtype=np.int64), ("/x",), {}),
            "/temp": (data1, ("/time", "/x"), {}),
        },
    )
    g2 = Granule(
        "g2",
        dimensions={"/time": n, "/x": 4},
        variables={
            "/time": (np.arange(n + 1, 2 * n + 1, dtype=np.int64), ("/time",), {}),
            "/x": (np.arange(4, dtype=np.int64), ("/x",), {}),
            "/temp": (data2, ("/time", "/x"), {}),
        },
    )
    with DAP4TestServer([g1, g2]) as server, tempfile.TemporaryDirectory() as td:
        budget = 128  # 8 rows of float32 x 4 cols; total data is 5x larger
        view = open_mfdataset(
            _urls(server, [g1, g2]),
            concat_dim="/time",
            keep_variables=["/temp"],
            max_buffer_bytes=budget,
        )
        out = os.path.join(td, "merged.nc4")
        to_netcdf(view, output_file=out)
        # Peak live payload must stay bounded regardless of total size.
        assert view.buffer_budget.peak <= budget
        with Dataset(out) as nc:
            np.testing.assert_array_equal(
                nc.variables["temp"][:], np.concatenate([data1, data2])
            )
        view.close()


def test_single_oversized_slice_rejected():
    g1 = granule(
        "g1", list(range(100)), np.arange(300, dtype=np.float32).reshape(100, 3)
    )
    with DAP4TestServer([g1]) as server:
        view = open_mfdataset(
            [server.url_for("g1")],
            concat_dim="/time",
            keep_variables=["/temp"],
            max_buffer_bytes=64,
        )
        with pytest.raises(BufferLimitExceededError):
            view["/temp"][:].data
        view.close()


# ---------------------------------------------------------------------------
# NetCDF round trip and failure atomicity
# ---------------------------------------------------------------------------


def test_netcdf_roundtrip(two_granules):
    server, granules = two_granules
    view = open_mfdataset(
        _urls(server, granules),
        concat_dim="/time",
        keep_variables=["/temp", "/x"],
    )
    full = _full(granules, "/temp")
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "merged.nc4")
        to_netcdf(view, output_file=out)
        with Dataset(out) as nc:
            assert nc.dimensions["time"].size == 5
            np.testing.assert_array_equal(nc.variables["time"][:], [1, 2, 3, 4, 5])
            np.testing.assert_array_equal(nc.variables["x"][:], [10.0, 20.0, 30.0])
            np.testing.assert_array_equal(nc.variables["temp"][:], full)
            assert nc.variables["temp"].units == "K"
            assert nc.collection == "synthetic"
        # View slice and NetCDF readback of the same region agree.
        region = np.asarray(view["/temp"][1:4, 1:3].data)
        with Dataset(out) as nc:
            np.testing.assert_array_equal(region, nc.variables["temp"][1:4, 1:3])
    view.close()


def _write_sentinel(path, payload=b"ORIGINAL-BYTES"):
    with open(path, "wb") as fh:
        fh.write(payload)
    return payload


def test_truncation_during_export_leaves_existing_file_unchanged():
    g1 = granule("g1", [1, 2, 3], [[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    g2 = granule("g2", [4, 5], [[100, 101, 102], [103, 104, 105]])
    with DAP4TestServer([g1, g2]) as server:
        view = open_mfdataset(
            _urls(server, [g1, g2]),
            concat_dim="/time",
            keep_variables=["/temp"],
        )
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "merged.nc4")
            sentinel = _write_sentinel(out)
            server.set_truncate(12)
            with pytest.raises((DAP4TruncatedResponseError, ConnectionError, OSError)):
                to_netcdf(view, output_file=out)
            with open(out, "rb") as fh:
                assert fh.read() == sentinel
        server.clear_faults()
        view.close()


def test_truncated_response_rejected():
    g1 = granule("g1", [1, 2, 3], [[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    g2 = granule("g2", [4, 5], [[100, 101, 102], [103, 104, 105]])
    with DAP4TestServer([g1, g2]) as server:
        view = open_mfdataset(
            _urls(server, [g1, g2]),
            concat_dim="/time",
            keep_variables=["/temp"],
        )
        server.set_truncate(12)
        with pytest.raises((DAP4TruncatedResponseError, ConnectionError, OSError)):
            view["/temp"][:].data
        server.clear_faults()
        view.close()


def test_bad_checksum_rejected():
    g1 = granule("g1", [1, 2, 3], [[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    g2 = granule("g2", [4, 5], [[100, 101, 102], [103, 104, 105]])
    with DAP4TestServer([g1, g2]) as server:
        view = open_mfdataset(
            _urls(server, [g1, g2]),
            concat_dim="/time",
            keep_variables=["/temp"],
        )
        server.set_corrupt_checksum(True)
        with pytest.raises(DAP4ChecksumError):
            view["/temp"][:].data
        server.clear_faults()
        view.close()


def test_http_failure_rejected_and_existing_file_kept():
    g1 = granule("g1", [1, 2, 3], [[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    g2 = granule("g2", [4, 5], [[100, 101, 102], [103, 104, 105]])
    with DAP4TestServer([g1, g2]) as server:
        view = open_mfdataset(
            _urls(server, [g1, g2]),
            concat_dim="/time",
            keep_variables=["/temp"],
        )
        server.set_http_failure("g2.dap", status=500)
        import requests as _requests

        # Retries on 500 surface as a requests exception (RetryError /
        # ConnectionError); either way the read must fail loudly.
        with pytest.raises(_requests.exceptions.RequestException):
            view["/temp"][3:5].data
        server.clear_faults()
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "merged.nc4")
            sentinel = _write_sentinel(out)
            server.set_http_failure("g2.dap", status=500)
            with pytest.raises(_requests.exceptions.RequestException):
                to_netcdf(view, output_file=out)
            with open(out, "rb") as fh:
                assert fh.read() == sentinel
        view.close()


def test_local_write_failure_keeps_old_file(monkeypatch):
    import netCDF4

    g1 = granule("g1", [1, 2], [[0, 1, 2], [3, 4, 5]])
    with DAP4TestServer([g1]) as server:
        view = open_mfdataset(
            [server.url_for("g1")],
            concat_dim="/time",
            keep_variables=["/temp"],
        )
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "merged.nc4")
            sentinel = _write_sentinel(out)

            def boom(*a, **k):
                raise OSError("simulated write failure")

            monkeypatch.setattr(netCDF4.Dataset, "sync", boom)
            with pytest.raises(OSError):
                to_netcdf(view, output_file=out)
            with open(out, "rb") as fh:
                assert fh.read() == sentinel
        view.close()


# ---------------------------------------------------------------------------
# Session ownership and view isolation
# ---------------------------------------------------------------------------


def test_caller_session_not_closed(two_granules):
    server, granules = two_granules
    from pydap.net import create_session

    session = create_session()
    view = open_mfdataset(
        _urls(server, granules),
        concat_dim="/time",
        keep_variables=["/temp"],
        session=session,
    )
    view.close()
    # Caller-owned session remains usable.
    response = session.get(server.url_for(granules[0].name) + ".dmr")
    assert response.status_code == 200
    session.close()


def test_two_views_are_independent(two_granules):
    server, granules = two_granules
    v1 = open_mfdataset(
        _urls(server, granules), concat_dim="/time", keep_variables=["/temp"]
    )
    v2 = open_mfdataset(
        _urls(server, granules),
        concat_dim="/time",
        keep_variables=["/temp"],
        overlap="error",
    )
    np.testing.assert_array_equal(
        np.asarray(v1["/temp"][0:2].data),
        np.asarray(v2["/temp"][0:2].data),
    )
    v1.close()
    with pytest.raises(RuntimeError):
        v1["/temp"][:].data
    # Second view keeps working after the first is closed.
    np.testing.assert_array_equal(
        np.asarray(v2["/temp"][0:2].data),
        _full(granules, "/temp")[0:2],
    )
    v2.close()
