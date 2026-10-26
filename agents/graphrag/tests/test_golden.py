"""Hand computed answers on the FILMS graph. Each query must give the expected rows both from the
compiled SQL and from the reference matcher, so a parser bug shared by the two still fails here."""

from __future__ import annotations

import os

import pytest
from conftest import FILMS

from pgcypher.compiler import compile_cypher, run
from pgcypher.parser import CypherError
from pgcypher.reference import run_reference

TRAIL = os.environ.get("PGRAPH_TRAIL", "cycle")
KNOWS_AB = ["p:alice", "p:bob", "KNOWS"]
KNOWS_BC = ["p:bob", "p:cara", "KNOWS"]
KNOWS_CA = ["p:cara", "p:alice", "KNOWS"]

CASES = [
    # multi hop, the 2Wiki shape: the mother of the director of Up
    (
        "MATCH (f:Film {title: 'Up'})-[:DIRECTOR]->(d)-[:MOTHER]->(m) RETURN m.name",
        None,
        [("Cara",)],
    ),
    # the blog's query: name is a column, matched by plain text equality against the bind parameter
    (
        "MATCH (f {name: $film})-[:DIRECTOR]->(d)-[:MOTHER]->(m) RETURN DISTINCT m.name",
        {"film": "Up"},
        [("Cara",)],
    ),
    (
        "MATCH (f {name: 2009}) RETURN f.title",
        None,  # a number never equals the text column
        [],
    ),
    ("MATCH (f:Film) WHERE f.name = f.title RETURN count(*) AS n", None, [(2,)]),
    (
        "MATCH (f:Film)-[:DIRECTOR]->(d) RETURN f.title, d.name ORDER BY f.title",
        None,
        [("Heat", "Alice"), ("Up", "Bob")],
    ),
    # variable length, KNOWS is the cycle alice -> bob -> cara -> alice
    (
        "MATCH (a:Person)-[:KNOWS*1..2]->(b) RETURN a.name, b.name ORDER BY a.name, b.name",
        None,
        [
            ("Alice", "Bob"),
            ("Alice", "Cara"),
            ("Bob", "Alice"),
            ("Bob", "Cara"),
            ("Cara", "Alice"),
            ("Cara", "Bob"),
        ],
    ),
    (
        "MATCH (a:Person)-[:KNOWS*3]->(b) RETURN a.name, b.name ORDER BY a.name",
        None,
        [("Alice", "Alice"), ("Bob", "Bob"), ("Cara", "Cara")],
    ),
    (
        "MATCH (a {name: 'Alice'})-[:MOTHER*0..1]->(b) RETURN b.name ORDER BY b.name",
        None,
        [("Alice",), ("Dan",)],
    ),
    (
        "MATCH (a {name: 'Alice'})-[p:KNOWS*2]->(b) RETURN b.name, p",
        None,
        [("Cara", [KNOWS_AB, KNOWS_BC])],
    ),
    # a variable length path never reuses an edge: no bouncing back to Bob, no fourth lap
    (
        "MATCH (a {name: 'Bob'})-[:KNOWS*2]-(b) RETURN b.name ORDER BY b.name",
        None,
        [("Alice",), ("Cara",)],
    ),
    ("MATCH (a {name: 'Alice'})-[:KNOWS*4]->(b) RETURN b.name", None, []),
    # direction
    (
        "MATCH (a:Person {name: 'Bob'})-[:KNOWS]-(b) RETURN b.name ORDER BY b.name",
        None,
        [("Alice",), ("Cara",)],
    ),
    (
        "MATCH (m)<-[:MOTHER]-(c) RETURN m.name, c.name ORDER BY m.name",
        None,
        [("Cara", "Bob"), ("Dan", "Alice")],
    ),
    # relationship uniqueness: around the undirected triangle, 6 trails of length 3 (24 walks
    # without it)
    ("MATCH (a:Person)-[:KNOWS]-(b)-[:KNOWS]-(c)-[:KNOWS]-(d) RETURN count(*) AS n", None, [(6,)]),
    (
        "MATCH (a {name: 'Bob'})-[:KNOWS]-(b)-[:KNOWS]-(c) RETURN c.name ORDER BY c.name",
        None,
        [("Alice",), ("Cara",)],
    ),
    # WHERE, nulls and types
    (
        "MATCH (p:Person) WHERE p.born > 1980 RETURN p.name ORDER BY p.name",
        None,
        [("Bob",), ("Cara",)],
    ),
    (
        "MATCH (p:Person) WHERE NOT (p.born > 1980) RETURN p.name ORDER BY p.name",
        None,
        [("Alice",)],
    ),
    ("MATCH (p:Person) WHERE p.born IS NULL RETURN p.name", None, [("Dan",)]),
    ("MATCH (p:Person) WHERE p.name > 5 RETURN p.name", None, []),
    (
        "MATCH (p:Person) WHERE p.name STARTS WITH 'C' OR p.name IN ['Dan']"
        " RETURN p.name ORDER BY p.name",
        None,
        [("Cara",), ("Dan",)],
    ),
    (
        "MATCH (p:Person) WHERE p.name < 'b' RETURN p.name ORDER BY p.name",
        None,  # code point order
        [("Alice",), ("Bob",), ("Cara",), ("Dan",)],
    ),
    ("MATCH (f:Film) WHERE f.year >= $y RETURN f.title", {"y": 2000}, [("Up",)]),
    (
        "MATCH (a)-[k:KNOWS]->(b) WHERE k.since >= 2010 RETURN a.name, b.name, k ORDER BY a.name",
        None,
        [("Bob", "Cara", KNOWS_BC), ("Cara", "Alice", KNOWS_CA)],
    ),
    # RETURN shapes
    (
        "MATCH (p:Person)-[r]->() RETURN p.name, count(r) AS out_degree ORDER BY p.name",
        None,
        [("Alice", 2), ("Bob", 2), ("Cara", 1)],
    ),
    (
        "MATCH (a:Person)-[:KNOWS*1..2]->(b) RETURN DISTINCT b.name ORDER BY b.name",
        None,
        [("Alice",), ("Bob",), ("Cara",)],
    ),
    (
        "MATCH (d)<-[:DIRECTOR]-(f) RETURN d.name, collect(f.title) AS films ORDER BY d.name",
        None,
        [("Alice", ["Heat"]), ("Bob", ["Up"])],
    ),
    ("MATCH (p:Person) WHERE p.born > 3000 RETURN count(*) AS n", None, [(0,)]),
    (
        "MATCH (p:Person) RETURN p.name ORDER BY p.name DESC SKIP 1 LIMIT 2",
        None,
        [("Cara",), ("Bob",)],
    ),
    (
        "MATCH (p:Person) RETURN p.name, p.born ORDER BY p.born",
        None,
        [("Alice", 1970), ("Bob", 1985), ("Cara", 1990), ("Dan", None)],
    ),
    (
        "MATCH (p:Person) RETURN p.name, p.born ORDER BY p.born DESC",
        None,
        [("Dan", None), ("Cara", 1990), ("Bob", 1985), ("Alice", 1970)],
    ),
]


@pytest.mark.parametrize(("query", "params", "expected"), CASES, ids=[c[0][:60] for c in CASES])
def test_sql_matches_hand_answer(films, query, params, expected):
    assert run(films, query, params, trail=TRAIL) == expected


@pytest.mark.parametrize(("query", "params", "expected"), CASES, ids=[c[0][:60] for c in CASES])
def test_reference_matches_hand_answer(query, params, expected):
    assert run_reference(query, FILMS, params) == expected


def test_without_uniqueness_the_triangle_has_24_walks(films):
    """The control for the uniqueness case above: the same three hops as plain joins, no edge
    check."""
    n = films.execute("""
        SELECT count(*)
        FROM graph.nodes a, graph.edges_both e1, graph.edges_both e2, graph.edges_both e3
        WHERE a.node_type = 'Person' AND e1.src_id = a.node_id AND e2.src_id = e1.dst_id
          AND e3.src_id = e2.dst_id AND e1.rel = 'KNOWS' AND e2.rel = 'KNOWS' AND e3.rel = 'KNOWS'
        """).fetchone()[0]
    assert n == 24


@pytest.mark.parametrize(
    "query",
    [
        "MATCH (a)-[*]->(b) RETURN a",
        "MATCH (a)-[*2..]->(b) RETURN a",
        "OPTIONAL MATCH (a) RETURN a",
        "MATCH (a:Person:Film) RETURN a",
        "MATCH (a) WITH a RETURN a",
        "MATCH (a) RETURN b",
        "MATCH (a)-[r]->(b), (b)-[r]->(c) RETURN a",
        "MATCH (a) RETURN a.x, a.x",
        "MATCH (a) WHERE a.x = $missing RETURN a",
    ],
)
def test_unsupported_or_invalid_queries_raise(query):
    with pytest.raises(CypherError):
        compile_cypher(query)


def test_values_are_bind_parameters_never_sql_text():
    c = compile_cypher('MATCH (p:Person {name: "x\'); DROP TABLE graph.nodes; --"}) RETURN p')
    assert "DROP TABLE" not in c.sql
    assert any("DROP TABLE" in str(v) for v in c.params.values())
