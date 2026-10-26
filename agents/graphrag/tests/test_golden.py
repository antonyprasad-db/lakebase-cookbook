"""Hand computed answers on the FILMS graph. Each query must give the expected rows both from the
compiled SQL and from the reference matcher, so a parser bug shared by the two still fails here."""

from __future__ import annotations

import datetime
import os
from decimal import Decimal

import psycopg
import pytest
from conftest import FILMS, install
from psycopg.rows import dict_row

from pgcypher.compiler import compile_cypher, run
from pgcypher.parser import CypherError, Lit, parse
from pgcypher.reference import run_reference

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
    (  # lists order repeatably, so SKIP and LIMIT over them are stable
        "MATCH (d)<-[:DIRECTOR]-(f) RETURN d.name, collect(f.title) AS films ORDER BY films DESC",
        None,
        [("Bob", ["Up"]), ("Alice", ["Heat"])],
    ),
    # lists: equal element by element; < between two lists is null (a documented gap)
    ("MATCH (f:Film) WHERE f.genre = $g RETURN f.title", {"g": ["crime"]}, [("Heat",)]),
    ("MATCH (f:Film) WHERE f.genre < $g RETURN f.title", {"g": ["a", "b"]}, []),
    # Alice's lucky is [1] and Bob's is [true]: different lists, as in jsonb and Cypher
    (
        "MATCH (a:Person), (b:Person) WHERE a.lucky = b.lucky"
        " RETURN a.name, b.name ORDER BY a.name",
        None,
        [("Alice", "Alice"), ("Bob", "Bob")],
    ),
    # nodes compare with nodes, by identity
    ("MATCH (a:Person), (b:Person) WHERE a = b RETURN count(*) AS n", None, [(4,)]),
    (
        "MATCH (a:Person)-[:KNOWS]->(b), (b)-[:KNOWS]->(c) WHERE a <> c RETURN count(*) AS n",
        None,
        [(3,)],
    ),
    # Cypher orders lists before strings, whatever the list holds
    (
        "MATCH (f:Film) RETURN f.title, f.genre ORDER BY f.genre",
        None,
        [("Heat", ["crime"]), ("Up", "animation")],
    ),
    # Up stores rating as JSON null and Heat has none: both are missing, as in Cypher
    ("MATCH (f:Film) WHERE f.rating IS NULL RETURN count(*) AS n", None, [(2,)]),
    ("MATCH (f:Film) RETURN count(f.rating) AS n, collect(f.rating) AS r", None, [(0, [])]),
    # WHERE n.name = ... takes the same text-equality path as {name: ...}
    ("MATCH (f)-[:DIRECTOR]->(d) WHERE f.name = $film RETURN d.name", {"film": "Up"}, [("Bob",)]),
    ("MATCH (f)-[:DIRECTOR]->(d) WHERE 'Heat' = f.name RETURN d.name", None, [("Alice",)]),
    (
        "MATCH (p) WHERE p.name IN ['Bob', 2, 'Dan'] RETURN p.name ORDER BY p.name",
        None,
        [("Bob",), ("Dan",)],
    ),
    ("MATCH (p) WHERE p.name IN [] RETURN p.name", None, []),
    ("MATCH (p:Person) WHERE NOT (p.name IN ['Bob', null]) RETURN p.name", None, []),
    (  # an ORDER BY name is the RETURN alias, even when a variable has the same name
        "MATCH (a:Person), (b:Person) WHERE a.born < b.born RETURN a AS b, b AS a"
        " ORDER BY a DESC, b",
        None,
        [("p:alice", "p:cara"), ("p:bob", "p:cara"), ("p:alice", "p:bob")],
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
    assert run(films, query, params) == expected


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
        "MATCH (a) SET a.x = 1 RETURN a",
        "MATCH (a) WHERE a.x = 1 XOR a.y = 2 RETURN a",
        "MATCH (a) RETURN a UNION MATCH (b) RETURN b",
        "MATCH (a) OPTIONAL MATCH (b) RETURN a",
        "MATCH (a)-[*1..7]->(b) RETURN a",  # above the default hop cap
        "MATCH (a)-[*1..3]->(b)-[*1..4]->(c) RETURN a",  # the cap is per MATCH
        "MATCH (a) WHERE a = 'p:alice' RETURN a",  # a node is not a string
        "MATCH (a) WHERE a RETURN a",
        "MATCH (a)-[r]->(b) WHERE r = b RETURN a",
        "MATCH (a)-[r]->(b) WHERE a < b RETURN a",
        "MATCH (a) WHERE a IN ['p:alice'] RETURN a",
        "MATCH (a) WHERE a STARTS WITH 'p' RETURN a",
        "MATCH (a) WHERE a.name = 'x\\q' RETURN a",  # an escape Cypher does not have
        "MATCH (a), (b {ref: a}) RETURN b",  # map values are literals or parameters, as in Cypher
        "MATCH (a)-[:KNOWS {since: b.born}]->(b) RETURN a",
        "MATCH (a)-[b]->(b) RETURN a",
        "MATCH (a) WHERE a.name = 'x\\u0000' RETURN a",  # jsonb cannot hold NUL
        "MATCH (a) RETURN a LIMIT 99999999999999999999",
    ],
)
def test_unsupported_or_invalid_queries_raise(query):
    with pytest.raises(CypherError):
        compile_cypher(query)


def test_hop_cap_can_be_raised_deliberately():
    assert "hops < 7" in compile_cypher("MATCH (a)-[*1..7]->(b) RETURN a", max_hops=7).sql


@pytest.mark.parametrize(
    "query",
    [
        "MATCH (f {name: $n}) RETURN f",
        "MATCH (f) WHERE f.name = $n RETURN f",
        "MATCH (f) WHERE f.name IN [$n, 'x'] RETURN f",
    ],
)
def test_name_equality_is_text_so_the_name_index_serves_it(query):
    """Both forms must compare the name column as text; to_jsonb(name) would bypass the btree."""
    sql = compile_cypher(query, {"n": "Up"}).sql
    assert "v1.name = (" in sql and "to_jsonb(v1.name) =" not in sql
    assert compile_cypher(query, {"n": 5}).sql == sql  # the SQL never depends on the value


@pytest.mark.parametrize(
    ("literal", "value"),
    [
        (r"'\u00e9'", "\u00e9"),
        (r"'a\rb'", "a\rb"),
        (r"'\U0001F600'", "\U0001f600"),
        (r"'it\'s'", "it's"),
        (r'"say \"hi\""', 'say "hi"'),
        (r"'back\\slash'", "back\\slash"),
    ],
)
def test_string_escapes(literal, value):
    assert parse(f"MATCH (a) WHERE a.x = {literal} RETURN a").where.right == Lit(value)


def test_keywords_are_fine_as_labels_types_and_keys():
    c = compile_cypher(
        "MATCH (o:Order)-[:CONTAINS]->(p:Set {limit: 1}) WHERE o.desc = 'x' RETURN o.order"
    )
    bound = [v for v in c.params.values()]
    assert {"Order", "Set", "limit", "desc", "order"} <= {v for v in bound if isinstance(v, str)}
    assert ["CONTAINS"] in bound


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), datetime.date(2024, 1, 1), {1, 2}, "x\x00y", ["a\x00"]]
)
def test_parameters_json_cannot_hold_raise(value):
    with pytest.raises(CypherError):
        compile_cypher("MATCH (f) WHERE f.year > $y RETURN f", {"y": value})


def test_decimal_parameters_are_numbers(films):
    rows = run(films, "MATCH (f:Film) WHERE f.year >= $y RETURN f.title", {"y": Decimal("2000.0")})
    assert rows == [("Up",)]


def test_a_graph_schema_holding_data_is_never_dropped(films):
    with pytest.raises(pytest.exit.Exception, match="refusing to drop"):
        install(films)
    assert films.execute("SELECT count(*) FROM graph.nodes").fetchone()[0] == len(FILMS.vertices)


def test_rows_are_values_whatever_the_row_factory(films):
    with psycopg.connect(os.environ["PGRAPH_DSN"], row_factory=dict_row) as conn:
        assert run(conn, "MATCH (f:Film {name: 'Up'}) RETURN f.title") == [("Up",)]


def test_values_are_bind_parameters_never_sql_text():
    c = compile_cypher('MATCH (p:Person {name: "x\'); DROP TABLE graph.nodes; --"}) RETURN p')
    assert "DROP TABLE" not in c.sql
    assert any("DROP TABLE" in str(v) for v in c.params.values())
