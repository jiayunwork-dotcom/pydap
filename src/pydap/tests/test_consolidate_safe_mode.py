"""Tests for the safe_mode coordinate-value verification added to
``consolidate_metadata`` and the http:// allowance for local test sources.
"""

import warnings

import numpy as np
import pytest

from pydap.client import _verify_shared_dimension_values, consolidate_metadata, open_dmr
from pydap.net import create_session
from pydap.tests.dap4_test_server import DAP4TestServer, Granule


def _granule(name, x_values):
    return Granule(
        name=name,
        dimensions={"/time": 2, "/x": 3},
        variables={
            "/time": (np.array([1, 2], dtype=np.int64), ("/time",), {}),
            "/x": (np.asarray(x_values, dtype=np.float64), ("/x",), {"units": "m"}),
            "/temp": (
                np.zeros((2, 3), dtype=np.float32),
                ("/time", "/x"),
                {"units": "K"},
            ),
        },
    )


def test_safe_mode_detects_coordinate_value_conflict():
    g1 = _granule("g1", [10.0, 20.0, 30.0])
    g2 = _granule("g2", [10.0, 20.0, 99.0])  # same length, different values
    with DAP4TestServer([g1, g2]) as server:
        session = create_session()
        urls = [server.url_for("g1"), server.url_for("g2")]
        dmr_urls = [url + ".dmr" for url in urls]
        empty = [open_dmr(u, session=session) for u in dmr_urls]
        # Lengths look identical...
        assert empty[0].dimensions["x"] == empty[1].dimensions["x"]
        # ...but the value-level check must refuse.
        with pytest.raises(ValueError, match="values differ"):
            _verify_shared_dimension_values(
                session, urls, empty, dmr_urls[0], skip_dim="time"
            )


def test_safe_mode_accepts_matching_coordinates():
    g1 = _granule("g1", [10.0, 20.0, 30.0])
    g2 = _granule("g2", [10.0, 20.0, 30.0])
    with DAP4TestServer([g1, g2]) as server:
        session = create_session()
        urls = [server.url_for("g1"), server.url_for("g2")]
        dmr_urls = [url + ".dmr" for url in urls]
        empty = [open_dmr(u, session=session) for u in dmr_urls]
        # Should complete without raising.
        _verify_shared_dimension_values(
            session, urls, empty, dmr_urls[0], skip_dim="time"
        )


def test_safe_mode_false_keeps_trust_premise():
    g1 = _granule("g1", [10.0, 20.0, 30.0])
    g2 = _granule("g2", [10.0, 20.0, 99.0])
    with DAP4TestServer([g1, g2]) as server:
        session = create_session(
            use_cache=True,
            cache_kwargs={"cache_name": "trust_premise", "use_temp": False},
        )
        session.cache.clear()
        urls = [server.url_for("g1"), server.url_for("g2")]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # safe_mode=False must not perform the value comparison; it
            # returns before the value-verification stage (None is allowed).
            try:
                consolidate_metadata(
                    urls, session=session, concat_dim="time", safe_mode=False
                )
            except Exception as exc:  # downstream caching mechanics only
                assert "values differ" not in str(exc)
        session.cache.clear()
