"""Parse a small openCypher read subset into plain dataclasses.

Supported: one MATCH clause with comma separated patterns; node variables, one label, inline
property maps; relationship variables, types (:A|B), direction (->, <-, undirected), variable
length (*2, *1..3, *..3, *0..2) with an upper bound; WHERE; RETURN [DISTINCT] with AS aliases,
count(*), count([DISTINCT] x), collect([DISTINCT] x); ORDER BY ... [ASC|DESC]; SKIP; LIMIT.
Expressions: property access, string/number/boolean/null literals, $parameters, = <> < <= > >=,
AND, OR, NOT, IS [NOT] NULL, STARTS WITH, ENDS WITH, CONTAINS, IN [list].

Anything else (OPTIONAL MATCH, WITH, UNWIND, writes, several labels, path variables, unbounded
variable length, arithmetic) raises CypherError rather than being guessed at.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


class CypherError(ValueError):
    pass


# --------------------------------------------------------------------------------------- AST
@dataclass(frozen=True)
class Lit:
    value: object


@dataclass(frozen=True)
class Param:
    name: str


@dataclass(frozen=True)
class Var:
    name: str


@dataclass(frozen=True)
class Prop:
    var: str
    key: str


@dataclass(frozen=True)
class Cmp:
    op: str  # = <> < <= > >=
    left: object
    right: object


@dataclass(frozen=True)
class StrOp:
    op: str  # STARTS WITH | ENDS WITH | CONTAINS
    left: object
    right: object


@dataclass(frozen=True)
class InList:
    expr: object
    items: tuple


@dataclass(frozen=True)
class IsNull:
    expr: object
    negated: bool


@dataclass(frozen=True)
class BoolOp:
    op: str  # AND | OR
    left: object
    right: object


@dataclass(frozen=True)
class Not:
    expr: object


@dataclass(frozen=True)
class Agg:
    func: str  # count | collect
    arg: object  # None for count(*)
    distinct: bool


@dataclass(frozen=True)
class NodePat:
    var: str | None
    label: str | None
    props: tuple  # ((key, expr), ...)


@dataclass(frozen=True)
class RelPat:
    var: str | None
    types: tuple
    direction: str  # out | in | both
    var_length: bool
    lo: int
    hi: int
    props: tuple


@dataclass(frozen=True)
class Pattern:
    nodes: tuple
    rels: tuple  # len(rels) == len(nodes) - 1


@dataclass(frozen=True)
class ReturnItem:
    expr: object
    name: str


@dataclass(frozen=True)
class Query:
    patterns: tuple
    where: object | None
    distinct: bool
    items: tuple
    order: tuple  # ((expr, descending), ...)
    skip: int | None
    limit: int | None


# ----------------------------------------------------------------------------------- lexer
_TOKEN = re.compile(
    r"""
    (?P<ws>\s+)
  | (?P<string>'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")
  | (?P<number>\d+\.\d+|\d+)
  | (?P<param>\$[A-Za-z_][A-Za-z0-9_]*)
  | (?P<ident>[A-Za-z_][A-Za-z0-9_]*|`[^`]+`)
  | (?P<op><>|<=|>=|\.\.|[-<>=()\[\]{}:,.|*])
""",
    re.VERBOSE,
)

_KEYWORDS = {
    "MATCH",
    "WHERE",
    "RETURN",
    "DISTINCT",
    "AS",
    "ORDER",
    "BY",
    "ASC",
    "ASCENDING",
    "DESC",
    "DESCENDING",
    "SKIP",
    "LIMIT",
    "AND",
    "OR",
    "NOT",
    "IS",
    "NULL",
    "TRUE",
    "FALSE",
    "STARTS",
    "ENDS",
    "WITH",
    "CONTAINS",
    "IN",
    "OPTIONAL",
    "UNWIND",
    "CREATE",
    "MERGE",
    "SET",
    "DELETE",
    "DETACH",
    "REMOVE",
    "CALL",
    "UNION",
    "XOR",
}
_UNSUPPORTED = {
    "OPTIONAL",
    "UNWIND",
    "CREATE",
    "MERGE",
    "SET",
    "DELETE",
    "DETACH",
    "REMOVE",
    "CALL",
    "UNION",
    "XOR",
}


@dataclass(frozen=True)
class Tok:
    kind: str  # string number param ident kw op end
    value: str
    start: int
    end: int


def tokenize(text: str) -> list[Tok]:
    out, pos = [], 0
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m:
            raise CypherError(f"unexpected character {text[pos]!r} at {pos}")
        kind, value = m.lastgroup or "", m.group()
        if kind == "ident":
            if value.startswith("`"):
                value = value[1:-1]
            elif value.upper() in _KEYWORDS:
                kind, value = "kw", value.upper()
        if kind != "ws":
            out.append(Tok(kind, value, m.start(), m.end()))
        pos = m.end()
    out.append(Tok("end", "", len(text), len(text)))
    return out


def _unquote(s: str) -> str:
    return re.sub(r"\\(.)", lambda m: {"n": "\n", "t": "\t"}.get(m.group(1), m.group(1)), s[1:-1])


# ---------------------------------------------------------------------------------- parser
class _Parser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.toks = tokenize(text)
        self.i = 0

    # token helpers
    def peek(self, k: int = 0) -> Tok:
        return self.toks[min(self.i + k, len(self.toks) - 1)]

    def next(self) -> Tok:
        t = self.peek()
        self.i += 1
        return t

    def at(self, value: str, k: int = 0) -> bool:
        t = self.peek(k)
        return t.kind in ("op", "kw") and t.value == value

    def accept(self, value: str) -> bool:
        if self.at(value):
            self.i += 1
            return True
        return False

    def expect(self, value: str) -> Tok:
        if not self.at(value):
            t = self.peek()
            raise CypherError(
                f"expected {value!r} at {t.start}, found {t.value or 'end of query'!r}"
            )
        return self.next()

    def ident(self) -> str:
        t = self.next()
        if t.kind != "ident":
            raise CypherError(f"expected a name at {t.start}, found {t.value or 'end of query'!r}")
        return t.value

    def integer(self) -> int:
        t = self.next()
        if t.kind != "number" or "." in t.value:
            raise CypherError(f"expected an integer at {t.start}")
        return int(t.value)

    # clauses
    def query(self) -> Query:
        for t in self.toks:
            if t.kind == "kw" and t.value in _UNSUPPORTED:
                raise CypherError(f"{t.value} is not supported")
        self.expect("MATCH")
        patterns = [self.pattern()]
        while self.accept(","):
            patterns.append(self.pattern())
        if self.at("MATCH"):
            raise CypherError("only one MATCH clause is supported")
        where = self.expr() if self.accept("WHERE") else None
        self.expect("RETURN")
        distinct = self.accept("DISTINCT")
        items = [self.return_item()]
        while self.accept(","):
            items.append(self.return_item())
        order = []
        if self.accept("ORDER"):
            self.expect("BY")
            while True:
                e = self.expr()
                desc = False
                if self.accept("DESC") or self.accept("DESCENDING"):
                    desc = True
                elif self.accept("ASC") or self.accept("ASCENDING"):
                    pass
                order.append((e, desc))
                if not self.accept(","):
                    break
        skip = self.integer() if self.accept("SKIP") else None
        limit = self.integer() if self.accept("LIMIT") else None
        if self.peek().kind != "end":
            t = self.peek()
            raise CypherError(f"unexpected {t.value!r} at {t.start}")
        names = [it.name for it in items]
        if len(set(names)) != len(names):
            raise CypherError("RETURN column names must be unique; use AS")
        return Query(tuple(patterns), where, distinct, tuple(items), tuple(order), skip, limit)

    def return_item(self) -> ReturnItem:
        start = self.peek().start
        e = self.expr()
        end = self.toks[self.i - 1].end
        name = self.ident() if self.accept("AS") else self.text[start:end]
        return ReturnItem(e, name)

    # patterns
    def pattern(self) -> Pattern:
        nodes, rels = [self.node()], []
        while self.at("-") or self.at("<"):
            rels.append(self.rel())
            nodes.append(self.node())
        return Pattern(tuple(nodes), tuple(rels))

    def node(self) -> NodePat:
        self.expect("(")
        var = self.ident() if self.peek().kind == "ident" else None
        label = None
        if self.accept(":"):
            label = self.ident()
            if self.at(":") or self.at("|"):
                raise CypherError("a node may have one label")
        props = self.prop_map() if self.at("{") else ()
        self.expect(")")
        return NodePat(var, label, props)

    def rel(self) -> RelPat:
        left = self.accept("<")
        self.expect("-")
        var, types, var_length, lo, hi, props = None, [], False, 1, 1, ()
        if self.accept("["):
            if self.peek().kind == "ident":
                var = self.ident()
            if self.accept(":"):
                types.append(self.ident())
                while self.accept("|"):
                    self.accept(":")
                    types.append(self.ident())
            if self.accept("*"):
                var_length, lo, hi = True, 1, None
                if self.peek().kind == "number":
                    lo = hi = self.integer()
                if self.accept(".."):
                    hi = self.integer() if self.peek().kind == "number" else None
                if hi is None:
                    raise CypherError(
                        "variable length relationships need an upper bound, e.g. *1..3"
                    )
                if lo > hi:
                    raise CypherError(f"empty range *{lo}..{hi}")
            if self.at("{"):
                props = self.prop_map()
            self.expect("]")
        self.expect("-")
        right = self.accept(">")
        if left and right:
            raise CypherError("a relationship cannot point both ways")
        direction = "in" if left else "out" if right else "both"
        return RelPat(var, tuple(types), direction, var_length, lo, hi, props)

    def prop_map(self) -> tuple:
        self.expect("{")
        pairs = []
        if not self.at("}"):
            while True:
                key = self.ident()
                self.expect(":")
                pairs.append((key, self.atom()))
                if not self.accept(","):
                    break
        self.expect("}")
        return tuple(pairs)

    # expressions, lowest precedence first
    def expr(self):
        e = self.and_expr()
        while self.accept("OR"):
            e = BoolOp("OR", e, self.and_expr())
        return e

    def and_expr(self):
        e = self.not_expr()
        while self.accept("AND"):
            e = BoolOp("AND", e, self.not_expr())
        return e

    def not_expr(self):
        if self.accept("NOT"):
            return Not(self.not_expr())
        return self.comparison()

    def comparison(self):
        left = self.atom()
        t = self.peek()
        if t.kind == "op" and t.value in ("=", "<>", "<", "<=", ">", ">="):
            self.next()
            return Cmp(t.value, left, self.atom())
        if self.accept("IS"):
            negated = self.accept("NOT")
            self.expect("NULL")
            return IsNull(left, negated)
        if self.accept("STARTS"):
            self.expect("WITH")
            return StrOp("STARTS WITH", left, self.atom())
        if self.accept("ENDS"):
            self.expect("WITH")
            return StrOp("ENDS WITH", left, self.atom())
        if self.accept("CONTAINS"):
            return StrOp("CONTAINS", left, self.atom())
        if self.accept("IN"):
            self.expect("[")
            items = []
            if not self.at("]"):
                while True:
                    items.append(self.atom())
                    if not self.accept(","):
                        break
            self.expect("]")
            return InList(left, tuple(items))
        return left

    def atom(self):
        t = self.peek()
        if self.accept("("):
            e = self.expr()
            self.expect(")")
            return e
        if t.kind == "string":
            self.next()
            return Lit(_unquote(t.value))
        if t.kind == "number" or (self.at("-") and self.peek(1).kind == "number"):
            sign = -1 if self.accept("-") else 1
            n = self.next().value
            return Lit(sign * (float(n) if "." in n else int(n)))
        if t.kind == "param":
            self.next()
            return Param(t.value[1:])
        if t.kind == "kw" and t.value in ("TRUE", "FALSE", "NULL"):
            self.next()
            return Lit({"TRUE": True, "FALSE": False, "NULL": None}[t.value])
        if t.kind == "ident":
            name = self.ident()
            if self.at("("):
                return self.call(name)
            if self.accept("."):
                return Prop(name, self.ident())
            return Var(name)
        raise CypherError(f"unexpected {t.value or 'end of query'!r} at {t.start}")

    def call(self, name: str) -> Agg:
        func = name.lower()
        if func not in ("count", "collect"):
            raise CypherError(f"function {name}() is not supported")
        self.expect("(")
        if func == "count" and self.accept("*"):
            self.expect(")")
            return Agg("count", None, False)
        distinct = self.accept("DISTINCT")
        arg = self.expr()
        self.expect(")")
        return Agg(func, arg, distinct)


def parse(text: str) -> Query:
    return _Parser(text).query()
