"""Replayable wire protocol for server-side grouped aggregation.

The pydap DAP2 client encodes the lazy query built through
``dataset.functions.group_reduce(...).where(...).select(...).limit(...)`` as
ordinary DAP2 server-side function calls in the constraint expression (CE),
so any hand-written HTTP request can reproduce exactly what the client does.

Grammar (all arguments are positional; nested calls are allowed)::

    group_reduce(<sequence>, "<keys_json>", "<aggregates_json>")
    where(<function-result>, "<predicate_json>")
    select(<function-result>, "<fields_json>")
    limit(<function-result>, <n>[, <offset>])

Encoding rules
--------------
* Every JSON document is passed as a *DAP2 string literal*: a double-quoted
  string where the characters ``"`` and ``\\`` are backslash-escaped. The
  JSON itself is compact (``json.dumps(..., separators=(",", ":"))``) and
  therefore never contains raw commas/parentheses outside of the quoted
  literal.
* The complete CE is percent-encoded for transport (``urllib`` style), so a
  comma, parenthesis, quote, backslash, ``&`` or non-ASCII byte inside a
  string literal can never be mistaken for function syntax or a CE separator.
* The middleware splits function calls with a quote-aware tokenizer:
  parentheses and commas occurring inside a double-quoted literal are
  ignored while tokenising, and backslash escapes are honoured.
* Predicate JSON::

      {"kind": "cmp", "field": "<summary column>", "op": "=|!=|<|<=|>|>=",
       "value": <string or finite number>}
      {"kind": "and", "predicates": [<predicate>, ...]}

* Select JSON: ``["col1", "col2"]`` (order preserved).
* ``limit`` arguments are non-negative integers; booleans are rejected.

Stage semantics are order preserving: the functions nest in the exact call
order used by the client, e.g.
``select(where(limit(group_reduce(...), 2), "..."), '["n"]')`` means
limit first, then filter, then project.
"""

from __future__ import annotations

import ast
import json
import re

from pydap.exceptions import ConstraintExpressionError

FUNCTION_RE = re.compile(r"([^(]*)\((.*)\)")


def split_top_level(text, separator):
    """Split *text* on *separator* outside parentheses and quoted literals."""
    tokens = []
    start = 0
    depth = 0
    in_string = False
    escaped = False
    for pos, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == separator and depth == 0:
            tokens.append(text[start:pos])
            start = pos + 1
    tokens.append(text[start:])
    return tokens


def split_ce(query_string):
    """Split a DAP2 CE into projection/selection tokens, quote-aware.

    Mirrors :func:`pydap.parsers.parse_ce` splitting on ``&`` but never breaks
    inside a double-quoted function argument (which may itself contain ``&``).
    """
    return [token for token in split_top_level(query_string, "&") if token]


def parse_call(expression):
    """Split ``name(arg, ...)`` into ``(name, [raw argument strings])``."""
    match = FUNCTION_RE.match(expression)
    if not match:
        raise ConstraintExpressionError(
            "Invalid server-side function call: %r" % expression
        )
    name, args = match.groups()
    name = name.strip()
    if not args:
        return name, []
    return name, split_top_level(args, ",")


def decode_dap_string(token):
    """Decode a DAP2 double-quoted string literal into a python ``str``."""
    token = token.strip()
    if len(token) < 2 or token[0] != '"' or token[-1] != '"':
        raise ConstraintExpressionError(
            "Expected a quoted string argument, got %r." % token
        )
    try:
        value = ast.literal_eval(token)
    except (ValueError, SyntaxError):
        raise ConstraintExpressionError("Malformed quoted string: %r." % token)
    if not isinstance(value, str):
        raise ConstraintExpressionError("Quoted argument must be a string.")
    return value


def encode_dap_string(value):
    """Encode a python string as a DAP2 double-quoted literal."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return '"%s"' % escaped


def compact_json(payload):
    """Compact JSON used inside a DAP2 string literal."""
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
