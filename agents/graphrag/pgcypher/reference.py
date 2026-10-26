"""A slow, brute force matcher for the same subset, used only as a test oracle.

It enumerates every binding by backtracking over an in memory graph, then applies the rules the
compiler documents: jsonb style equality, three valued logic, code point string order, edge
uniqueness across the whole MATCH. It shares the parser with the compiler and nothing else; the
hand computed golden tests guard the parser itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .parser import (
    Agg,
    BoolOp,
    Cmp,
    CypherError,
    InList,
    IsNull,
    Lit,
    Not,
    Param,
    Prop,
    Query,
    StrOp,
    Var,
    parse,
)


@dataclass(frozen=True)
class Edge:
    """An edge of graph.edges. Its key (src, dst, rel) is its identity, as in the table."""

    src: str
    dst: str
    rel: str
    props: dict = field(default_factory=dict, hash=False, compare=False)

    @property
    def key(self) -> tuple[str, str, str]:
        return self.src, self.dst, self.rel


@dataclass
class Graph:
    vertices: dict[
        str, tuple[str, dict]
    ]  # node_id -> (node_type, props); props["name"] is the name column
    edges: list[Edge]

    def __post_init__(self) -> None:
        self.out: dict[str, list[Edge]] = {v: [] for v in self.vertices}
        self.inc: dict[str, list[Edge]] = {v: [] for v in self.vertices}
        for e in self.edges:
            self.out[e.src].append(e)
            self.inc[e.dst].append(e)

    def steps(self, v: str, direction: str):
        if direction in ("out", "both"):
            for e in self.out[v]:
                yield e, e.dst
        if direction in ("in", "both"):
            for e in self.inc[v]:
                if direction == "in" or e.src != e.dst:  # a self loop is seen once when undirected
                    yield e, e.src


# ------------------------------------------------------------------------------- values
def kind(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, (int, float)):
        return "number"
    if isinstance(v, str):
        return "string"
    return "array" if isinstance(v, list) else "object"


def same(a, b) -> bool:
    """Equality as jsonb has it: same kind, then same value, element by element for lists and
    maps, so [1] and [true] differ while 1 and 1.0 do not."""
    if kind(a) != kind(b):
        return False
    if isinstance(a, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    return a == b


def eq(a, b):
    if a is None or b is None:
        return None
    return same(a, b)


def order(op: str, a, b):
    if a is None or b is None or kind(a) != kind(b) or kind(a) in ("array", "object"):
        return None
    return {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b}[op]


def and3(a, b):
    if a is False or b is False:
        return False
    return None if a is None or b is None else True


def or3(a, b):
    if a is True or b is True:
        return True
    return None if a is None or b is None else False


def _numbers(v):
    """1 and 1.0 are the same value to Cypher and to jsonb, so they must group the same here."""
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, list | tuple):
        return [_numbers(x) for x in v]
    if isinstance(v, dict):
        return {k: _numbers(x) for k, x in v.items()}
    return v


def canon(v) -> str:
    return json.dumps(_numbers(v), sort_keys=True)


# ------------------------------------------------------------------------------ matcher
class _Ref:
    def __init__(self, q: Query, g: Graph, params: dict) -> None:
        self.q, self.g, self.params = q, g, params

    def val(self, e, env):
        if isinstance(e, Lit):
            return e.value
        if isinstance(e, Param):
            if e.name not in self.params:
                raise CypherError(f"missing parameter ${e.name}")
            return self.params[e.name]
        if isinstance(e, Var):
            kind_, x = self.bound(e.name, env)
            return (
                x
                if kind_ == "node"
                else list(x.key)
                if kind_ == "rel"
                else [list(r.key) for r in x]
            )
        if isinstance(e, Prop):
            kind_, x = self.bound(e.var, env)
            if kind_ == "path":
                raise CypherError(f"{e.var} is a list of relationships and has no properties")
            return (self.g.vertices[x][1] if kind_ == "node" else x.props).get(e.key)
        if isinstance(e, Agg):
            raise CypherError("aggregate outside RETURN")
        return self.cond(e, env)

    @staticmethod
    def bound(name, env):
        if name not in env:
            raise CypherError(f"variable {name} is not bound by the MATCH")
        return env[name]

    def cond(self, e, env):
        if isinstance(e, Lit):
            if e.value is None or isinstance(e.value, bool):
                return e.value
            raise CypherError("a condition must be boolean")
        if isinstance(e, (Var, Prop, Param)):
            return eq(self.val(e, env), True)
        if isinstance(e, BoolOp):
            a, b = self.cond(e.left, env), self.cond(e.right, env)
            return and3(a, b) if e.op == "AND" else or3(a, b)
        if isinstance(e, Not):
            a = self.cond(e.expr, env)
            return None if a is None else not a
        if isinstance(e, IsNull):
            return (self.val(e.expr, env) is None) != e.negated
        if isinstance(e, Cmp):
            a, b = self.val(e.left, env), self.val(e.right, env)
            if e.op == "=":
                return eq(a, b)
            if e.op == "<>":
                r = eq(a, b)
                return None if r is None else not r
            return order(e.op, a, b)
        if isinstance(e, StrOp):
            a, b = self.val(e.left, env), self.val(e.right, env)
            if not (isinstance(a, str) and isinstance(b, str)):
                return None
            return {"STARTS WITH": a.startswith(b), "ENDS WITH": a.endswith(b), "CONTAINS": b in a}[
                e.op
            ]
        if isinstance(e, InList):
            x = self.val(e.expr, env)
            results = [eq(x, self.val(i, env)) for i in e.items]
            return True if True in results else None if None in results else False
        raise CypherError(f"unsupported expression {e!r}")

    # pattern matching
    def node_ok(self, n, v, env) -> bool:
        label, props = self.g.vertices[v]
        if n.label and label != n.label:
            return False
        if n.var and n.var in env and env[n.var] != ("node", v):
            return False
        return all(eq(props.get(k), self.val(x, env)) is True for k, x in n.props)

    def edge_ok(self, r, e, env) -> bool:
        if r.types and e.rel not in r.types:
            return False
        return all(eq(e.props.get(k), self.val(x, env)) is True for k, x in r.props)

    @staticmethod
    def bind(env, var, value):
        if not var:
            return env
        return {**env, var: value}

    def trails(self, start, r, env, used):
        stack = [(start, [])]
        while stack:
            v, path = stack.pop()
            if r.lo <= len(path):
                yield v, path
            if len(path) < r.hi:
                for e, other in self.g.steps(v, r.direction):
                    if e.key not in used and e not in path and self.edge_ok(r, e, env):
                        stack.append((other, [*path, e]))

    def matches(self):
        def patterns(pi, env, used):
            if pi == len(self.q.patterns):
                yield env
                return
            p = self.q.patterns[pi]
            first = p.nodes[0]
            starts = (
                [env[first.var][1]] if first.var and first.var in env else list(self.g.vertices)
            )
            for v in starts:
                if self.node_ok(first, v, env):
                    yield from hops(p, 0, v, self.bind(env, first.var, ("node", v)), used, pi)

        def hops(p, ri, cur, env, used, pi):
            if ri == len(p.rels):
                yield from patterns(pi + 1, env, used)
                return
            r, n = p.rels[ri], p.nodes[ri + 1]
            if r.var and (r.var in env or r.var == n.var):
                raise CypherError(f"variable {r.var} is bound twice")
            if n.var and n.var in env and env[n.var][0] != "node":
                raise CypherError(f"{n.var} is a relationship, not a node")
            if r.var_length:
                options = [(end, path) for end, path in self.trails(cur, r, env, used)]
            else:
                options = [
                    (other, [e])
                    for e, other in self.g.steps(cur, r.direction)
                    if e.key not in used and self.edge_ok(r, e, env)
                ]
            for end, path in options:
                if not self.node_ok(n, end, env):
                    continue
                value = ("path", path) if r.var_length else ("rel", path[0])
                env2 = self.bind(self.bind(env, r.var, value), n.var, ("node", end))
                yield from hops(p, ri + 1, end, env2, used | {e.key for e in path}, pi)

        yield from patterns(0, {}, frozenset())

    def run(self) -> list[tuple]:
        q = self.q
        envs = [env for env in self.matches() if q.where is None or self.cond(q.where, env) is True]
        keys = [i for i, it in enumerate(q.items) if not isinstance(it.expr, Agg)]
        if len(keys) == len(q.items):
            rows = [tuple(self.val(it.expr, env) for it in q.items) for env in envs]
        else:
            groups: dict[str, tuple[tuple, list]] = {}
            for env in envs:
                kv = tuple(self.val(q.items[i].expr, env) for i in keys)
                groups.setdefault(canon(kv), (kv, []))[1].append(env)
            if not groups and not keys:
                groups[canon(())] = ((), [])
            rows = []
            for kv, members in groups.values():
                it_kv = iter(kv)
                rows.append(
                    tuple(
                        self.agg(it.expr, members) if isinstance(it.expr, Agg) else next(it_kv)
                        for it in q.items
                    )
                )
        if q.distinct:
            seen, unique = set(), []
            for r in rows:
                if canon(r) not in seen:
                    seen.add(canon(r))
                    unique.append(r)
            rows = unique
        for e, desc in reversed(q.order):
            col = self.order_column(e)
            rows.sort(key=lambda r, c=col: sort_key(r[c]), reverse=desc)
        if q.skip:
            rows = rows[q.skip :]
        if q.limit is not None:
            rows = rows[: q.limit]
        return rows

    def order_column(self, e) -> int:
        for i, it in enumerate(self.q.items):  # an alias wins over a variable of the same name
            if isinstance(e, Var) and e.name == it.name:
                return i
        for i, it in enumerate(self.q.items):
            if e == it.expr:
                return i
        raise CypherError("ORDER BY must name a RETURN item")

    def agg(self, a: Agg, envs):
        if a.func == "count" and a.arg is None:
            return len(envs)
        values = [v for v in (self.val(a.arg, env) for env in envs) if v is not None]
        if a.distinct:
            values = list({canon(v): v for v in values}.values())
        return len(values) if a.func == "count" else values


def _jsonb_key(v):
    """Postgres's jsonb btree order, used within lists: null < string < number < boolean < list
    < map, lists by length and then element by element."""
    k = kind(v)
    if k is None:
        return (0,)
    if k in ("array", "object"):
        items = v if k == "array" else [x for _, x in sorted(v.items())]
        return (4 if k == "array" else 5, len(v), tuple(_jsonb_key(x) for x in items))
    return ({"string": 1, "number": 2, "boolean": 3}[k], v)


def sort_key(v):
    rank = {"object": 0, "array": 1, "string": 2, "boolean": 3, "number": 4}.get(kind(v) or "", 5)
    if rank < 2:  # maps and lists order as Postgres orders jsonb
        return (rank, _jsonb_key(v))
    return (rank, (v if rank < 5 else 0,))


def run_reference(text: str, graph: Graph, params: dict | None = None) -> list[tuple]:
    return _Ref(parse(text), graph, params or {}).run()
