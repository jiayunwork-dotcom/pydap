"""End-to-end example over a real HTTP (waitress) server.

Starts a pydap WSGI application on a socket, then drives the complete
open_url -> functions -> lazy view -> summary round trip, plus an equivalent
hand-written HTTP request.
"""

import json
import socket
import subprocess
import sys
import time
import urllib.request

import numpy as np

from pydap.client import open_url, result_condition
from pydap.handlers.lib import BaseHandler
from pydap.model import BaseType, DatasetType, SequenceType
from pydap.wsgi.ssf import ServerSideFunctions


def build_dataset():
    rows = [
        ("alpha", 1.0), ("alpha", 3.0),
        ("beta", 10.0), ("beta", 20.0),
        ("gamma", -4.0),
    ]
    dataset = DatasetType("stations")
    sequence = SequenceType("obs")
    sequence["site"] = BaseType("site")
    sequence["v"] = BaseType("v")
    sequence.data = np.array(rows, dtype=[("site", "U16"), ("v", "f8")])
    dataset["obs"] = sequence
    return dataset


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main():
    port = free_port()
    code = (
        "from waitress import serve\n"
        "from pydap.handlers.lib import BaseHandler\n"
        "from pydap.wsgi.ssf import ServerSideFunctions\n"
        "import example_http_server_ds\n"
        "serve(ServerSideFunctions(BaseHandler(example_http_server_ds.build_dataset())), "
        "host='127.0.0.1', port=%d, threads=2, _quiet=True)\n" % port
    )
    server = subprocess.Popen(
        [sys.executable, "-c", code],
        cwd="/workspace/dev",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    base = "http://127.0.0.1:%d/stations" % port
    try:
        for _ in range(50):
            try:
                urllib.request.urlopen(base + ".dds", timeout=1)
                break
            except Exception:
                time.sleep(0.2)

        remote = open_url(base)
        keys = json.dumps(["site"])
        aggs = json.dumps(
            [
                {"name": "n", "op": "count", "field": None},
                {"name": "mean_v", "op": "mean", "field": "v"},
            ]
        )
        view = remote.functions.group_reduce(remote.obs, keys, aggs)
        print("client rows:")
        for row in view["summary"].iterdata():
            print(" ", tuple(row))

        filtered = view.where(result_condition("mean_v", ">", 2.0)).select(
            "site", "mean_v"
        )
        print("filtered + projected rows:")
        for row in filtered["summary"].iterdata():
            print(" ", tuple(row))

        # Hand-written equivalent HTTP request using urllib.
        def quoted(value):
            return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'

        ce = "group_reduce(obs,%s,%s)" % (quoted(keys), quoted(aggs))
        import requests

        url = base + ".dods?" + requests.utils.quote(ce, safe="")
        from pydap.client import open_dods_url

        hand = open_dods_url(url)
        print("raw-HTTP rows:")
        for row in hand["summary"].iterdata():
            print(" ", tuple(row))
    finally:
        server.terminate()
        server.wait(timeout=10)


if __name__ == "__main__":
    main()
