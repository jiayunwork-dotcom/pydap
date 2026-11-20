"""An implementation of server-side functions.

pydap implements DAP server-side functions throught a custom WSGI middleware.
This simplifies writing custom handlers, since they don't need to parse and
apply the function calls themselves.

"""

import ast
import operator
import re
import sys
from functools import reduce
from importlib.metadata import entry_points

import numpy as np
from webob import Request

from ..exceptions import ServerError
from ..handlers.lib import BaseHandler, apply_projection
from ..lib import fix_shorthand, unquote, walk
from ..model import DatasetType, SequenceType
from ..parsers import parse_ce
from ..parsers.group_protocol import (
    decode_dap_string,
    parse_call,
    split_ce,
)
from ..wsgi.group_reduce import (
    GroupReduceConfig,
    group_reduce as group_reduce_func,
    limit as limit_func,
    select as select_func,
    where as where_func,
)

FUNCTION = re.compile(r"([^(]*)\((.*)\)")
RELOP = re.compile(r"(<=|<|>=|>|=~|=|!=)")

# Functions that implement the staged grouped-aggregation protocol. Their
# JSON/string arguments require quote-aware tokenisation, and their results
# are composed lazily.
GROUP_PROTOCOL_FUNCS = ("group_reduce", "where", "select", "limit")


def load_functions():
    """Load all available functions from the system, returning a dictionary."""
    # Relative import of functions:
    eps = entry_points(group="pydap.function")
    Rs = [r for r in eps if r.module[:5] == "pydap"]
    nRs = [r for r in eps if r.module[:5] != "pydap"]
    base_dict = dict((r.name, r.load()) for r in Rs)

    opts_dict = dict((r.name, r.load()) for r in nRs)
    base_dict.update(opts_dict)
    return base_dict


class _ClosingResponse:
    """Wrap a WSGI response iterator so grouped engines are released.

    pydap streams Sequence data through a generator. If the client goes away
    mid-stream (or any exception is raised while serialising) the generator's
    ``finally`` blocks only run when it is closed. WSGI servers call
    ``close()`` on the app_iter when they stop iterating, which propagates
    into every nested iterator here.

    Grouping engines are fully *prepared* (input scan, validation, spill
    finalisation) before this wrapper is returned, so by the time the status
    line is emitted every data-detectable failure has already been converted
    into a regular error response.
    """

    def __init__(self, app_iter, closer):
        self._app_iter = iter(app_iter)
        self._closer = closer
        self._closed = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._app_iter)

    def close(self):
        if self._closed:
            return
        self._closed = True
        close = getattr(self._app_iter, "close", None)
        if close is not None:
            close()
        self._closer()


class ServerSideFunctions(object):
    """A WebOb based middleware for handling server-side function calls.

    The middleware works by removing function calls from the request,
    forwarding the request to pydap, and then applying the functions calls to
    the returned dataset.

    """

    def __init__(self, app, **kwargs):
        self.app = app

        self.functions = load_functions()
        # Pull the grouped-aggregation configuration out of kwargs before the
        # remaining entries are registered as ordinary server-side functions
        # (the historical kwargs-based registration keeps working).
        config_kwargs = {}
        for key in ("group_reduce_max_groups", "group_reduce_tempdir"):
            if key in kwargs:
                config_kwargs[key] = kwargs.pop(key)
        self.group_config = GroupReduceConfig(**config_kwargs)
        self.functions.update(kwargs)

    def __call__(self, environ, start_response):
        # specify that we want the parsed dataset
        environ["x-wsgiorg.want_parsed_response"] = True
        req = Request(environ)
        # ``req.query_string`` arrives percent-encoded; decode once with the
        # DAP-aware unquoting rules and split with the quote-aware tokenizer
        # so ampersands/commas inside JSON string literals are preserved.
        raw_query = unquote(req.query_string)
        tokens = split_ce(raw_query)
        projection, selection = parse_ce(raw_query)

        # check if there are any functions calls in the request
        called = any(
            FUNCTION.match(s) and not _is_group_stage(s) for s in selection
        ) or any(p for p in projection if isinstance(p, str)) or any(
            _is_group_stage(token) for token in tokens
        )

        # ignore DAS requests and requests without functions
        path, response = req.path.rsplit(".", 1)
        if response == "das" or not called:
            return self.app(environ, start_response)

        group_tokens = [token for token in tokens if _is_group_stage(token)]
        group_only = bool(group_tokens) and not _has_legacy_function(
            tokens, projection
        )

        engines = []

        def release_engines():
            for engine in engines:
                engine.close()

        if group_only:
            # Grouped-query path: run the complete input scan and validation
            # (phase 1) before any response byte is produced so data failures
            # become regular error responses, then serialise (phase 2).
            try:
                dataset = self._evaluate(
                    req, tokens, projection, selection, engines
                )
                for engine in engines:
                    engine.prepare()
            except BaseException:
                release_engines()
                if environ.get("x-wsgiorg.throw_errors"):
                    raise
                from ..responses.error import ErrorResponse

                return ErrorResponse(info=sys.exc_info())(
                    environ, start_response
                )
        else:
            # Historical function path: keep the original exception
            # propagation semantics untouched.
            dataset = self._evaluate(req, tokens, projection, selection, engines)

        try:
            res = BaseHandler.responses[response](dataset)
            app_iter = res(environ, start_response)
        except BaseException:
            release_engines()
            raise
        return _ClosingResponse(app_iter, release_engines)

    # ------------------------------------------------------------------
    # Request evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, req, tokens, projection, selection, engines):
        # Requests that only use the staged grouped-aggregation protocol are
        # handled on a dedicated, quote-aware path; everything else (including
        # mixed legacy function calls) goes through the historical machinery.
        group_tokens = [token for token in tokens if _is_group_stage(token)]
        if group_tokens and not _has_legacy_function(tokens, projection):
            return self._evaluate_group_request(
                req, group_tokens, projection, selection, engines
            )
        return self._evaluate_legacy(req, projection, selection)

    def _evaluate_group_request(
        self, req, group_tokens, projection, selection, engines
    ):
        # Non-function selections are ordinary DAP2 row constraints on the
        # source sequence and are applied by the handler before grouping.
        plain_selection = [
            token
            for token in selection
            if not FUNCTION.match(token) and not _is_group_stage(token)
        ]
        req.query_string = "&".join(plain_selection)
        res = req.get_response(self.app)
        method = getattr(res.app_iter, "x_wsgiorg_parsed_response", False)
        if not method:
            raise ServerError("Unable to call server-side function!")
        dataset = method(DatasetType)

        # Evaluate the (possibly nested) grouped-query calls in token order.
        # Each top-level token is an independent projection expression; the
        # final result of each is a Dataset containing a ``summary`` Sequence.
        produced = None
        for token in group_tokens:
            produced = eval_group_call(
                token, dataset, self.functions, self.group_config, engines
            )

        # A grouped query always yields the summary sequence. Projection may
        # additionally include ordinary source variables (e.g. a parallel
        # raw column); merge them into the returned dataset.
        if produced is None:  # pragma: no cover - guarded by ``called``
            raise ServerError("No group_reduce call found in request.")

        base_projection = [p for p in projection if not isinstance(p, str)]
        if base_projection:
            base_projection = fix_shorthand(base_projection, dataset)
            merged = apply_projection(base_projection, dataset)
            if "summary" not in merged.keys():
                merged["summary"] = produced["summary"]
            return merged
        return produced

    def _evaluate_legacy(self, req, projection, selection):
        # apply selection without any function calls
        req.query_string = "&".join(s for s in selection if not FUNCTION.match(s))
        res = req.get_response(self.app)

        # get the dataset
        method = getattr(res.app_iter, "x_wsgiorg_parsed_response", False)
        if not method:
            raise ServerError("Unable to call server-side function!")
        dataset = method(DatasetType)

        # apply selection containing server-side functions
        selection = (s for s in selection if FUNCTION.match(s))
        for expr in selection:
            if RELOP.search(expr):
                call, op, other = RELOP.split(expr)
                op = {
                    "<": operator.lt,
                    ">": operator.gt,
                    "!=": operator.ne,
                    "=": operator.eq,
                    ">=": operator.ge,
                    "<=": operator.le,
                    "=~": lambda a, b: re.match(b, a),
                }[op]
                other = ast.literal_eval(other)
            else:
                call, op, other = expr, operator.eq, 1

            # evaluate the function call
            sequence = eval_function(dataset, call, self.functions)

            # is this an inplace call?
            for var in walk(dataset, SequenceType):
                if sequence is var:
                    break
            else:
                # get the data from the resulting variable, and use it to
                # constrain the original dataset
                child = list(sequence.children())[0]
                data = np.fromiter(child.data, child.dtype)
                if data.dtype.char == "S":
                    valid = np.array(
                        list(map(lambda v: op(str(v), str(other)), data)), bool
                    )
                else:
                    valid = op(data, other)

                for sequence in walk(dataset, SequenceType):
                    sequence.data = np.rec.fromrecords(
                        [tuple(row) for row in sequence.iterdata()],
                        names=list(sequence.keys()),
                    )[valid]

        # now apply projection
        if projection:
            projection = fix_shorthand(projection, dataset)
            base = [p for p in projection if not isinstance(p, str)]
            func = [p for p in projection if isinstance(p, str)]

            # apply non-function projection
            out = apply_projection(base, dataset)

            # apply function projection
            for call in func:
                var = eval_function(dataset, call, self.functions)
                for child in walk(var):
                    parent = reduce(operator.getitem, [out] + child.id.split(".")[:-1])
                    if child.name not in parent.keys():
                        parent[child.name] = child
                        break
            dataset = out

        return dataset


def _is_group_stage(token):
    match = FUNCTION.match(token)
    return bool(match) and match.group(1).strip() in GROUP_PROTOCOL_FUNCS


def _has_legacy_function(tokens, projection):
    for token in tokens:
        match = FUNCTION.match(token)
        if match and not _is_group_stage(token):
            return True
    return any(
        isinstance(p, str) and not _is_group_stage(p) for p in projection
    )


def eval_group_call(expression, dataset, functions, config, engines):
    """Quote-aware evaluation of a grouped-query function call expression."""
    name, raw_args = parse_call(expression)

    def resolve(token):
        token = token.strip()
        nested = FUNCTION.match(token)
        if nested and nested.group(1).strip() in GROUP_PROTOCOL_FUNCS:
            return eval_group_call(token, dataset, functions, config, engines)
        # Quoted string literal (JSON payloads). Check this before the
        # generic function-call test, since a literal may itself contain
        # parentheses or commas.
        if token[:1] == '"':
            return decode_dap_string(token)
        if FUNCTION.match(token):
            # A legacy function nested inside the grouped call: evaluate it
            # through the historical machinery.
            return eval_function(dataset, token, functions)
        # Bare variable path.
        try:
            names = re.sub(r"\[.*?\]", "", str(token)).split(".")
            return reduce(operator.getitem, [dataset] + names)
        except Exception:
            try:
                return ast.literal_eval(token)
            except Exception:
                return token

    args = [resolve(token) for token in raw_args]

    if name == "group_reduce":
        # Inject the per-middleware configuration; the registered entry-point
        # function signature is (dataset, sequence, keys_json, aggregates_json)
        # and remains directly callable with four arguments.
        result = group_reduce_func(dataset, *args, config=config)
        engine = result._group_reduce_engine
        if engine not in engines:
            engines.append(engine)
        return result
    if name in ("where", "select", "limit"):
        func = where_func if name == "where" else (
            select_func if name == "select" else limit_func
        )
        result = func(dataset, *args)
        return result
    raise ServerError("Unknown grouped-query function %r" % name)


def eval_function(dataset, function, functions):
    """Evaluate a given function on a dataset.

    This function parses and evaluates a (possibly nested) function call,
    returning its result.

    """
    name, args = FUNCTION.match(function).groups()

    def tokenize(input):
        start = pos = count = 0
        for char in input:
            if char == "(":
                count += 1
            elif char == ")":
                count -= 1
            elif char == "," and count == 0:
                yield input[start:pos]
                start = pos + 1
            pos += 1
        yield input[start:]

    def parse(token):
        if FUNCTION.match(token):
            return eval_function(dataset, token, functions)
        else:
            try:
                names = re.sub(r"\[.*?\]", "", str(token)).split(".")
                return reduce(operator.getitem, [dataset] + names)
            except Exception:
                try:
                    return ast.literal_eval(token)
                except Exception:
                    return token

    args = map(parse, tokenize(args))
    func = functions[name]

    return func(dataset, *args)
