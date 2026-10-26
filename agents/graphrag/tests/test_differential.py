"""Random graphs and random queries: the compiled SQL must return exactly what the reference
matcher returns. The graphs include self loops, edges of different types between the same two
nodes and missing properties; the queries mix directions, variable length, rebound variables, nulls
and type mismatches."""

from __future__ import annotations

import json
import os
import random

import pytest
from conftest import load

from pgcypher.compiler import run
from pgcypher.reference import Edge, Graph, canon, run_reference

LABELS, TYPES = ["A", "B", "C"], ["R", "S", "T"]
NAMES = ["al", "alba", "bo", "bob", "cy", "Cy"]
SEEDS = range(int(os.environ.get("PGRAPH_CASES", "300")))
TRAIL = os.environ.get("PGRAPH_TRAIL", "cycle")
QUERIES_PER_GRAPH = 4


def random_graph(rng: random.Random) -> Graph:
    n = rng.randint(4, 9)
    vertices = {}
    for i in range(n):
        props: dict = {}
        if rng.random() < 0.8:
            props["x"] = rng.randint(0, 5)
        props["name"] = rng.choice(NAMES)  # a NOT NULL column in graph.nodes
        if rng.random() < 0.5:
            props["flag"] = rng.random() < 0.5
        vertices[f"n{i}"] = (rng.choice(LABELS), props)
    ids = list(vertices)
    edges: dict[tuple, Edge] = {}  # graph.edges is keyed on (src_id, dst_id, rel)
    for _ in range(rng.randint(n, 3 * n)):
        e = Edge(
            rng.choice(ids),
            rng.choice(ids),
            rng.choice(TYPES),
            {"w": rng.randint(0, 3)} if rng.random() < 0.7 else {},
        )
        edges.setdefault(e.key, e)
    return Graph(vertices, list(edges.values()))


class QueryGen:
    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.names = iter("abcdefghijk")
        self.nodes: list[str] = []
        self.rels: list[str] = []  # single relationship variables (have properties)
        self.paths: list[str] = []  # variable length relationship variables (lists)
        self.var_length = 0

    def node(self) -> str:
        r = self.rng
        if self.nodes and r.random() < 0.15:
            v = r.choice(self.nodes)
        elif r.random() < 0.8:
            v = next(self.names)
            self.nodes.append(v)
        else:
            v = ""
        label = f":{r.choice(LABELS)}" if r.random() < 0.25 else ""
        roll = r.random()
        props = (
            f" {{x: {r.randint(0, 5)}}}"
            if roll < 0.08
            else f" {{name: '{r.choice(NAMES)}'}}"
            if roll < 0.14
            else ""
        )
        return f"({v}{label}{props})"

    def rel(self) -> str:
        r = self.rng
        v = next(self.names) if r.random() < 0.35 else ""
        types = ":" + "|".join(r.sample(TYPES, r.randint(1, 2))) if r.random() < 0.45 else ""
        length = ""
        if self.var_length < 2 and r.random() < 0.3:
            self.var_length += 1
            length = r.choice(["*1..2", "*2", "*0..2", "*1..3", "*..2"])
        props = f" {{w: {r.randint(0, 3)}}}" if r.random() < 0.05 else ""
        if v:
            (self.paths if length else self.rels).append(v)
        body = f"[{v}{types}{length}{props}]"
        return r.choice([f"-{body}->", f"<-{body}-", f"-{body}-"])

    def pattern(self) -> str:
        parts = [self.node()]
        for _ in range(self.rng.randint(0, 2)):
            parts += [self.rel(), self.node()]
        return "".join(parts)

    def pred(self) -> str:
        r = self.rng
        v = r.choice(self.nodes)
        choice = r.randrange(9)
        if choice == 0 and self.rels:
            op = r.choice(["=", "<>", "<", ">=", "<="])
            return f"{r.choice(self.rels)}.w {op} {r.randint(0, 3)}"
        if choice == 1:
            return f"{v}.x {r.choice(['=', '<>', '<', '<=', '>', '>='])} {r.randint(0, 5)}"
        if choice == 2:
            return f"{v}.name {r.choice(['=', '<', '>='])} '{r.choice(NAMES)}'"
        if choice == 3:
            return f"{v}.x {r.choice(['<', '='])} {r.choice(self.nodes)}.x"
        if choice == 4:
            return f"{v}.{r.choice(['flag', 'x'])} IS {r.choice(['', 'NOT '])}NULL"
        if choice == 5:
            op = r.choice(["STARTS WITH", "ENDS WITH", "CONTAINS"])
            return f"{v}.name {op} '{r.choice(['a', 'b', 'l', 'C'])}'"
        if choice == 6:
            items = [str(r.randint(0, 5)) for _ in range(r.randint(0, 3))] + (
                ["null"] if r.random() < 0.3 else []
            )
            return f"{v}.x IN [{', '.join(items)}]"
        if choice == 7:
            return f"{v}.flag"
        return f"{v}.name > {r.randint(0, 3)}"  # type mismatch: always null

    def where(self, depth: int = 2) -> str:
        if depth == 0 or self.rng.random() < 0.5:
            p = self.pred()
            return f"NOT ({p})" if self.rng.random() < 0.15 else p
        return f"({self.where(depth - 1)} {self.rng.choice(['AND', 'OR'])} {self.where(depth - 1)})"

    def query(self) -> tuple[str, list[str], bool]:
        """Returns the query text, each column's kind (scalar, path or collect) and whether it is
        ordered."""
        r = self.rng
        patterns = [self.pattern() for _ in range(r.randint(1, 2))]
        if (
            not self.nodes
        ):  # every node came out anonymous; add a named one so WHERE and RETURN have a variable
            self.nodes.append("z")
            patterns.append("(z)")
        text = "MATCH " + ", ".join(patterns)
        if r.random() < 0.6:
            text += " WHERE " + self.where()
        scalars = [f"{v}{p}" for v in self.nodes for p in ("", ".x", ".name")] + [
            f"{v}.w" for v in self.rels
        ]
        if r.random() < 0.35:
            keys = r.sample(scalars, min(len(scalars), r.randint(0, 2)))
            agg = r.choice(
                [
                    "count(*)",
                    f"count({r.choice(scalars)})",
                    f"count(DISTINCT {r.choice(scalars)})",
                    f"collect({r.choice(scalars)})",
                    f"collect(DISTINCT {r.choice(scalars)})",
                ]
            )
            exprs = [*keys, agg]
            kinds = ["scalar"] * len(keys) + ["collect" if agg.startswith("collect") else "scalar"]
        else:
            pool = scalars + self.paths
            exprs = r.sample(pool, min(len(pool), r.randint(1, 3)))
            kinds = ["path" if e in self.paths else "scalar" for e in exprs]
        distinct = "DISTINCT " if r.random() < 0.3 else ""
        text += " RETURN " + distinct + ", ".join(f"{e} AS c{i}" for i, e in enumerate(exprs))
        ordered = all(k == "scalar" for k in kinds) and r.random() < 0.4
        if ordered:  # order by every column, so the expected order is total
            text += " ORDER BY " + ", ".join(
                f"c{i}{r.choice(['', ' DESC'])}" for i in range(len(exprs))
            )
            if r.random() < 0.5:
                text += f" SKIP {r.randint(0, 2)} LIMIT {r.randint(1, 4)}"
        return text, kinds, ordered


def normalize(rows, kinds) -> list[str]:
    """collect() order is unspecified, so compare its lists sorted; path lists keep their order."""
    return [
        canon(
            [sorted(v, key=canon) if k == "collect" else v for v, k in zip(row, kinds, strict=True)]
        )
        for row in rows
    ]


@pytest.mark.parametrize("seed", SEEDS)
def test_compiled_sql_matches_reference(conn, seed):
    rng = random.Random(seed)
    g = random_graph(rng)
    load(conn, g)
    for _ in range(QUERIES_PER_GRAPH):
        gen = QueryGen(rng)
        text, kinds, ordered = gen.query()
        expected = normalize(run_reference(text, g), kinds)
        got = normalize(run(conn, text, trail=TRAIL), kinds)
        if not ordered:
            expected, got = sorted(expected), sorted(got)
        assert got == expected, (
            f"seed {seed}\n{text}\ngraph {json.dumps([g.vertices, [e.__dict__ for e in g.edges]])}"
        )
