"""Compile the parsed openCypher subset into one native Postgres query over the recipe's graph.

The graph is graph.nodes and graph.edges from sql/schema.sql, plus the graph.edges_both view from
sql/graph_functions.sql. A node's label is its node_type, its name property is the name column
and every other property lives in props. An edge's type is rel and it is identified by its key
(src_id, dst_id, rel).

Values travel as jsonb, because properties live in jsonb, so each returned row is a list of JSON
values: a node variable returns the node_id, a relationship variable its key as
[src_id, dst_id, rel], and a variable length relationship the list of those keys along its path.
Literals, parameters, labels and property keys are always bind parameters; only validated integers
(hop bounds, SKIP, LIMIT) and generated aliases are written into the SQL text.

The variable length relationships in one MATCH may total at most max_hops (default 6) hops,
because the number of trails grows exponentially with hops on a graph with hubs; pass a larger
max_hops deliberately. Each walk is a recursive CTE whose Postgres CYCLE clause, keyed on the
edge's key, keeps it a trail.

The semantics that matter for correctness follow Cypher:
  no edge is bound twice within the MATCH, including inside a variable length path (relationship
  uniqueness), which is why an undirected three hop match around a triangle returns 6 rows, not 24;
  a missing property is null, and so is one stored as JSON null (Cypher cannot store a null
  property), and comparisons involving null are null (three valued logic);
  < <= > >= across different types are null, and strings compare by code point (COLLATE "C");
  < <= > >= between two lists or two maps are null too: a deliberate gap, not Cypher's
  element by element order;
  a node, relationship or path may be compared only with another of the same kind (= or <>),
  and is never a condition by itself: anything else is rejected rather than approximated;
  count(x) and collect(x) skip nulls, and the non aggregated RETURN items are the grouping keys.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal

from .parser import (
    Agg,
    BoolOp,
    Cmp,
    CypherError,
    InList,
    IsNull,
    Lit,
    NodePat,
    Not,
    Param,
    Pattern,
    Prop,
    Query,
    RelPat,
    StrOp,
    Var,
    parse,
)

MAX_HOPS = 6


def _param_json(value) -> str:
    """JSON for a bind value. Cypher has integers and floats but no decimals, so a Decimal (what
    psycopg returns for numeric columns) becomes an integer when it is integral and a float
    otherwise. Anything jsonb cannot hold (NaN, infinity, a NUL character, other types) is a
    CypherError."""

    def default(o):
        if isinstance(o, Decimal) and o.is_finite():
            return int(o) if o == o.to_integral_value() else float(o)
        raise TypeError(type(o).__name__)

    def has_nul(v) -> bool:
        if isinstance(v, str):
            return "\x00" in v
        if isinstance(v, list | tuple):
            return any(has_nul(x) for x in v)
        if isinstance(v, dict):
            return any(has_nul(k) or has_nul(x) for k, x in v.items())
        return False

    if has_nul(value):
        raise CypherError("a string cannot contain a NUL character")
    try:
        return json.dumps(value, default=default, allow_nan=False)
    except (TypeError, ValueError) as err:
        raise CypherError(f"parameter value is not JSON: {err}") from None


@dataclass
class Compiled:
    sql: str
    params: dict
    columns: list[str]


class _Compiler:
    def __init__(self, params: dict | None, max_hops: int) -> None:
        self.max_hops = max_hops
        self.hops = 0  # upper bounds of the variable length relationships so far
        self.user_params = params or {}
        self.params: dict = {}
        self.from_: list[str] = []
        self.conds: list[str] = []
        self.vars: dict[str, tuple[str, str]] = {}  # name -> (node | rel | path, alias)
        self.keycols: dict[str, tuple[str, str]] = {}  # edge alias -> its key's src and dst columns
        self.edges: list[tuple[str, str]] = []  # (one | many, sql) for relationship uniqueness
        self.n = 0

    # ------------------------------------------------------------------------ helpers
    def bind(self, value) -> str:
        name = f"p{len(self.params)}"
        self.params[name] = value
        return f"%({name})s"

    def json(self, value) -> str:
        return f"{self.bind(_param_json(value))}::jsonb"

    def alias(self, prefix: str) -> str:
        self.n += 1
        return f"{prefix}{self.n}"

    def lookup(self, name: str) -> tuple[str, str]:
        if name not in self.vars:
            raise CypherError(f"variable {name} is not bound by the MATCH")
        return self.vars[name]

    # ------------------------------------------------------------------------ patterns
    def pattern(self, p: Pattern) -> None:
        left = self.node(p.nodes[0])
        for r, n in zip(p.rels, p.nodes[1:], strict=True):
            right = self.node(n)
            self.rel(r, left, right)
            left = right

    def node(self, n: NodePat) -> str:
        if n.var and n.var in self.vars:
            kind, alias = self.vars[n.var]
            if kind != "node":
                raise CypherError(f"{n.var} is a relationship, not a node")
        else:
            alias = self.alias("v")
            self.from_.append(f"graph.nodes AS {alias}")
            if n.var:
                self.vars[n.var] = ("node", alias)
        if n.label:
            self.conds.append(f"{alias}.node_type = {self.bind(n.label)}")
        for key, e in n.props:
            self.conds.append(self.node_match(alias, key, e))
        return alias

    def prop(self, alias: str, key: str) -> str:
        """A property from props, with a stored JSON null read as missing (SQL NULL)."""
        return f"nullif({alias}.props -> {self.bind(key)}, 'null'::jsonb)"

    def node_prop(self, alias: str, key: str) -> str:
        """name is a column of graph.nodes; every other property lives in props."""
        if key == "name":
            return f"to_jsonb({alias}.name)"
        return self.prop(alias, key)

    def name_equals(self, alias: str, e) -> str:
        """name = e as text equality on the name column, so the btree on name serves it for any
        e, with Cypher's result when e is not a string (false) or is null (null)."""
        v = self.val(e)
        return f"({alias}.name = ({v} #>> '{{}}') AND jsonb_typeof({v}) = 'string')"

    def node_match(self, alias: str, key: str, e) -> str:
        if key == "name":
            return self.name_equals(alias, e)
        return f"{self.node_prop(alias, key)} = {self.val(e)}"

    def entity(self, e) -> str | None:
        """node, rel or path when e is a bare variable, which is compared by identity only."""
        return self.lookup(e.name)[0] if isinstance(e, Var) else None

    def edge_key(self, a: str) -> tuple[str, str]:
        """The edge's key as a row, for comparing two single edges, and as text, for checking it
        against a variable length path's list of keys."""
        ks, kd = self.keycols[a]
        return (
            f"({a}.{ks}, {a}.{kd}, {a}.rel)",
            f"jsonb_build_array({a}.{ks}, {a}.{kd}, {a}.rel)::text",
        )

    def edge_filters(self, a: str, r: RelPat) -> list[str]:
        out = []
        if r.types:
            out.append(f"{a}.rel = ANY ({self.bind(list(r.types))}::text[])")
        for key, e in r.props:
            out.append(f"{self.prop(a, key)} = {self.val(e)}")
        return out

    def rel(self, r: RelPat, left: str, right: str) -> None:
        if r.var and r.var in self.vars:
            raise CypherError(f"variable {r.var} is bound twice")
        if r.var_length:
            self.hops += r.hi
            if self.hops > self.max_hops:
                raise CypherError(
                    f"variable length relationships total {self.hops} hops, above the cap of"
                    f" {self.max_hops}; pass max_hops to raise it"
                )
        # edges_both holds every edge in both directions of travel and keeps its stored key in
        # key_src/key_dst, so an undirected walk sees one edge, not two, whichever way it crosses it
        table, ks, kd = (
            ("edges_both", "key_src", "key_dst")
            if r.direction == "both"
            else ("edges", "src_id", "dst_id")
        )
        near, far = ("dst_id", "src_id") if r.direction == "in" else ("src_id", "dst_id")
        if not r.var_length:
            a = self.alias("e")
            self.from_.append(f"graph.{table} AS {a}")
            self.keycols[a] = (ks, kd)
            self.conds += [
                f"{a}.{near} = {left}.node_id",
                f"{a}.{far} = {right}.node_id",
                *self.edge_filters(a, r),
            ]
            self.edges.append(("one", a))
            if r.var:
                self.vars[r.var] = ("rel", a)
            return
        w = self.alias("w")
        step = "".join(f" AND {c}" for c in self.edge_filters("e", r))
        key = f"jsonb_build_array(e.{ks}, e.{kd}, e.rel)"
        # The Postgres CYCLE clause, keyed on the edge's key, keeps the walk a trail. Each row also
        # carries keys, its edge keys as text, for uniqueness against the rest of the MATCH; a
        # named path carries edges too, the same keys as jsonb, which is what it returns.
        named = r.var is not None
        edges_col = ", edges" if named else ""
        edges_start = ", '[]'::jsonb" if named else ""
        edges_step = f", walk.edges || jsonb_build_array({key})" if named else ""
        self.from_.append(
            "LATERAL (\n"
            f"  WITH RECURSIVE walk (node_id, hops, key_src, key_dst, rel, keys{edges_col}) AS (\n"
            f"    SELECT {left}.node_id, 0, NULL::text, NULL::text, NULL::text, ARRAY[]::text[]"
            f"{edges_start}\n"
            "    UNION ALL\n"
            f"    SELECT e.{far}, walk.hops + 1, e.{ks}, e.{kd}, e.rel,\n"
            f"           walk.keys || {key}::text{edges_step}\n"
            f"    FROM walk JOIN graph.{table} AS e ON e.{near} = walk.node_id{step}\n"
            f"    WHERE walk.hops < {int(r.hi)}\n"
            "  ) CYCLE key_src, key_dst, rel SET reused USING trail\n"
            f"  SELECT node_id, keys{edges_col} FROM walk\n"
            f"  WHERE hops >= {int(r.lo)} AND NOT reused\n"
            f") AS {w}"
        )
        self.conds.append(f"{w}.node_id = {right}.node_id")
        self.edges.append(("many", f"{w}.keys"))
        if r.var:
            self.vars[r.var] = ("path", w)

    def uniqueness(self) -> None:
        for i, (ka, a) in enumerate(self.edges):
            for kb, b in self.edges[i + 1 :]:
                if ka == kb == "one":
                    self.conds.append(f"{self.edge_key(a)[0]} <> {self.edge_key(b)[0]}")
                elif ka == "one":
                    self.conds.append(f"NOT {self.edge_key(a)[1]} = ANY ({b})")
                elif kb == "one":
                    self.conds.append(f"NOT {self.edge_key(b)[1]} = ANY ({a})")
                else:
                    self.conds.append(f"NOT {a} && {b}")

    # --------------------------------------------------------------------- expressions
    def val(self, e) -> str:
        """SQL for the expression as a jsonb value."""
        if isinstance(e, Lit):
            return "NULL::jsonb" if e.value is None else self.json(e.value)
        if isinstance(e, Param):
            if e.name not in self.user_params:
                raise CypherError(f"missing parameter ${e.name}")
            v = self.user_params[e.name]
            return "NULL::jsonb" if v is None else self.json(v)
        if isinstance(e, Var):
            kind, a = self.lookup(e.name)
            if kind == "path":
                return f"{a}.edges"
            if kind == "rel":
                ks, kd = self.keycols[a]
                return f"jsonb_build_array({a}.{ks}, {a}.{kd}, {a}.rel)"
            return f"to_jsonb({a}.node_id)"
        if isinstance(e, Prop):
            kind, a = self.lookup(e.var)
            if kind == "path":
                raise CypherError(f"{e.var} is a list of relationships and has no properties")
            if kind == "node":
                return self.node_prop(a, e.key)
            return self.prop(a, e.key)
        if isinstance(e, Agg):
            raise CypherError("count() and collect() are only allowed as RETURN items")
        return f"to_jsonb({self.cond(e)})"

    def cond(self, e) -> str:
        """SQL for the expression as a boolean, with SQL three valued logic."""
        if isinstance(e, Lit):
            if e.value is None:
                return "NULL::boolean"
            if isinstance(e.value, bool):
                return "TRUE" if e.value else "FALSE"
            raise CypherError("a condition must be boolean")
        if isinstance(e, (Var, Prop, Param)):
            if kind := self.entity(e):
                raise CypherError(f"{e.name} is a {kind}, not a condition")
            return f"({self.val(e)} = 'true'::jsonb)"
        if isinstance(e, BoolOp):
            return f"({self.cond(e.left)} {e.op} {self.cond(e.right)})"
        if isinstance(e, Not):
            return f"(NOT {self.cond(e.expr)})"
        if isinstance(e, IsNull):
            return f"({self.val(e.expr)} IS {'NOT ' if e.negated else ''}NULL)"
        if isinstance(e, Cmp):
            kinds = {self.entity(e.left), self.entity(e.right)}
            if kinds != {None} and (len(kinds) > 1 or e.op not in ("=", "<>")):
                raise CypherError(
                    "a node, relationship or path compares only with = or <> to another of its kind"
                )
            if e.op == "=" and (fast := self.name_equality(e)):
                return fast
            left, right = self.val(e.left), self.val(e.right)
            if e.op in ("=", "<>"):
                return f"({left} {e.op} {right})"
            return (
                f"(CASE WHEN jsonb_typeof({left}) <> jsonb_typeof({right}) THEN NULL"
                f" WHEN jsonb_typeof({left}) IN ('array', 'object') THEN NULL"
                f" WHEN jsonb_typeof({left}) = 'string'"
                f" THEN ({left} #>> '{{}}') COLLATE \"C\" {e.op} ({right} #>> '{{}}')"
                f" ELSE {left} {e.op} {right} END)"
            )
        if isinstance(e, StrOp | InList):
            for x in (e.left, e.right) if isinstance(e, StrOp) else (e.expr, *e.items):
                if kind := self.entity(x):
                    raise CypherError(f"{x.name} is a {kind}; use one of its properties")
        if isinstance(e, StrOp):
            left, right = self.val(e.left), self.val(e.right)
            ls, rs = f"({left} #>> '{{}}')", f"({right} #>> '{{}}')"
            test = {
                "STARTS WITH": f"starts_with({ls}, {rs})",
                "ENDS WITH": f"right({ls}, length({rs})) = {rs}",
                "CONTAINS": f"strpos({ls}, {rs}) > 0",
            }[e.op]
            strings = f"jsonb_typeof({left}) = 'string' AND jsonb_typeof({right}) = 'string'"
            return f"(CASE WHEN {strings} THEN {test} END)"
        if isinstance(e, InList):
            if isinstance(e.expr, Prop) and e.expr.key == "name":
                kind, alias = self.lookup(e.expr.var)
                if kind == "node":  # one text equality per item, so the btree on name serves it
                    tests = [self.name_equals(alias, i) for i in e.items]
                    return f"({' OR '.join(tests)})" if tests else "FALSE"
            items = ", ".join(self.val(i) for i in e.items)
            return f"({self.val(e.expr)} = ANY (ARRAY[{items}]::jsonb[]))"
        if isinstance(e, Agg):
            raise CypherError("count() and collect() are only allowed as RETURN items")
        raise CypherError(f"unsupported expression {e!r}")

    def name_equality(self, e: Cmp) -> str | None:
        """WHERE n.name = x (either way round) on a node, as text equality on the name column."""
        for prop, other in ((e.left, e.right), (e.right, e.left)):
            if isinstance(prop, Prop) and prop.key == "name":
                kind, alias = self.lookup(prop.var)
                if kind == "node":
                    return self.name_equals(alias, other)
        return None

    def agg(self, a: Agg) -> str:
        distinct = "DISTINCT " if a.distinct else ""
        if a.func == "count":
            return (
                "to_jsonb(count(*))"
                if a.arg is None
                else f"to_jsonb(count({distinct}{self.val(a.arg)}))"
            )
        v = self.val(a.arg)
        return f"coalesce(jsonb_agg({distinct}{v}) FILTER (WHERE {v} IS NOT NULL), '[]'::jsonb)"

    # ------------------------------------------------------------------------- query
    @staticmethod
    def order_column(q: Query, e) -> str:
        """A RETURN alias wins over a variable of the same name, as in Cypher."""
        for i, it in enumerate(q.items):
            if isinstance(e, Var) and e.name == it.name:
                return f"c{i}"
        for i, it in enumerate(q.items):
            if e == it.expr:
                return f"c{i}"
        raise CypherError("ORDER BY must name a RETURN item, by alias or by the same expression")

    @staticmethod
    def sort_keys(c: str, desc: bool) -> list[str]:
        """Order by type as Cypher does (maps, lists, strings, booleans, numbers, then nulls last
        ascending and first descending), then numerically, by code point, or false before true.
        Maps and lists, which include relationship keys and paths, then order as Postgres orders
        jsonb (by size, then element by element), so the order is total and repeatable, though
        not exactly Cypher's order within lists."""
        d = " DESC" if desc else ""
        return [
            f"(CASE jsonb_typeof({c}) WHEN 'object' THEN 0 WHEN 'array' THEN 1 WHEN 'string' THEN 2"
            f" WHEN 'boolean' THEN 3 WHEN 'number' THEN 4 ELSE 5 END){d}",
            f"(CASE WHEN jsonb_typeof({c}) = 'number' THEN ({c})::numeric END){d}",
            f"(CASE WHEN jsonb_typeof({c}) = 'string' THEN {c} #>> '{{}}' END) COLLATE \"C\"{d}",
            f"(CASE WHEN jsonb_typeof({c}) = 'boolean' THEN ({c})::boolean END){d}",
            f"(CASE WHEN jsonb_typeof({c}) IN ('object', 'array') THEN {c} END){d}",
        ]

    def compile(self, q: Query) -> Compiled:
        for p in q.patterns:
            self.pattern(p)
        self.uniqueness()
        if q.where is not None:
            self.conds.append(self.cond(q.where))
        cols, group, has_agg = [], [], False
        for i, it in enumerate(q.items):
            if isinstance(it.expr, Agg):
                has_agg = True
                cols.append(f"{self.agg(it.expr)} AS c{i}")
            else:
                cols.append(f"{self.val(it.expr)} AS c{i}")
                group.append(str(i + 1))
        sql = "SELECT " + ("DISTINCT " if q.distinct else "") + ",\n       ".join(cols)
        sql += "\nFROM " + ",\n     ".join(self.from_)
        if self.conds:
            sql += "\nWHERE " + "\n  AND ".join(self.conds)
        if has_agg and group:
            sql += "\nGROUP BY " + ", ".join(group)
        if q.order or q.skip is not None or q.limit is not None:
            keys = [k for e, desc in q.order for k in self.sort_keys(self.order_column(q, e), desc)]
            outer = (
                f"SELECT {', '.join(f'c{i}' for i in range(len(q.items)))}\nFROM (\n{sql}\n) AS q"
            )
            if keys:
                outer += "\nORDER BY " + ",\n         ".join(keys)
            if q.skip is not None:
                outer += f"\nOFFSET {int(q.skip)}"
            if q.limit is not None:
                outer += f"\nLIMIT {int(q.limit)}"
            sql = outer
        return Compiled(sql, self.params, [it.name for it in q.items])


def compile_cypher(text: str, params: dict | None = None, max_hops: int = MAX_HOPS) -> Compiled:
    """Compile against the graph schema that sql/schema.sql and sql/graph_functions.sql create."""
    return _Compiler(params, max_hops).compile(parse(text))


def run(conn, text: str, params: dict | None = None, max_hops: int = MAX_HOPS) -> list[tuple]:
    """Compile and execute on a psycopg connection; returns rows of Python JSON values, whatever
    row factory the connection was opened with."""
    from psycopg.rows import tuple_row

    c = compile_cypher(text, params, max_hops)
    with conn.cursor(row_factory=tuple_row) as cur:
        return cur.execute(c.sql, c.params).fetchall()
