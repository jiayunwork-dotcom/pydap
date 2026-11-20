"""Tiny dataset module used by example_http_server.py."""

import numpy as np

from pydap.model import BaseType, DatasetType, SequenceType


def build_dataset():
    rows = [
        ("alpha", 1.0),
        ("alpha", 3.0),
        ("beta", 10.0),
        ("beta", 20.0),
        ("gamma", -4.0),
    ]
    dataset = DatasetType("stations")
    sequence = SequenceType("obs")
    sequence["site"] = BaseType("site")
    sequence["v"] = BaseType("v")
    sequence.data = np.array(rows, dtype=[("site", "U16"), ("v", "f8")])
    dataset["obs"] = sequence
    return dataset
