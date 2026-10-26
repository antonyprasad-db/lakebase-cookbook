"""Shared fixtures. Point PGRAPH_DSN at a Postgres database the tests may overwrite: its graph
schema is dropped and rebuilt from sql/schema.sql and sql/graph_functions.sql, so the tests run on
the recipe's own tables. It needs the extensions schema.sql creates (vector, ltree, pg_trgm), which
Lakebase has."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import psycopg
import pytest
from psycopg.types.json import Jsonb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pgcypher.reference import Edge, Graph  # noqa: E402


def install(conn) -> None:
    conn.execute("DROP SCHEMA IF EXISTS graph CASCADE")
    conn.execute((ROOT / "sql/schema.sql").read_text())
    conn.execute((ROOT / "sql/graph_functions.sql").read_text())


def load(conn, g: Graph) -> None:
    """name is a column of graph.nodes, so it comes out of the reference graph's props."""
    conn.execute("TRUNCATE graph.edges, graph.node_embeddings, graph.nodes")
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO graph.nodes (node_id, node_type, name, props) VALUES (%s, %s, %s, %s)",
            [
                (v, label, props["name"], Jsonb({k: x for k, x in props.items() if k != "name"}))
                for v, (label, props) in g.vertices.items()
            ],
        )
        cur.executemany(
            "INSERT INTO graph.edges (src_id, dst_id, rel, props) VALUES (%s, %s, %s, %s)",
            [(e.src, e.dst, e.rel, Jsonb(e.props)) for e in g.edges],
        )


@pytest.fixture(scope="session")
def conn():
    dsn = os.environ.get("PGRAPH_DSN")
    if not dsn:
        pytest.skip("set PGRAPH_DSN to a Postgres database the tests may overwrite")
    with psycopg.connect(dsn, autocommit=True) as c:
        install(c)
        yield c


# A films and people graph small enough to work every answer out by hand.
FILMS = Graph(
    vertices={
        "f:heat": ("Film", {"name": "Heat", "title": "Heat", "year": 1995}),
        "f:up": ("Film", {"name": "Up", "title": "Up", "year": 2009}),
        "p:alice": ("Person", {"name": "Alice", "born": 1970}),
        "p:bob": ("Person", {"name": "Bob", "born": 1985}),
        "p:cara": ("Person", {"name": "Cara", "born": 1990}),
        "p:dan": ("Person", {"name": "Dan"}),
    },
    edges=[
        Edge("f:heat", "p:alice", "DIRECTOR"),
        Edge("f:up", "p:bob", "DIRECTOR"),
        Edge("p:bob", "p:cara", "MOTHER"),
        Edge("p:alice", "p:dan", "MOTHER"),
        Edge("p:alice", "p:bob", "KNOWS", {"since": 2001}),
        Edge("p:bob", "p:cara", "KNOWS", {"since": 2010}),  # same two people as the MOTHER edge
        Edge("p:cara", "p:alice", "KNOWS", {"since": 2015}),
    ],
)


@pytest.fixture
def films(conn):
    load(conn, FILMS)
    return conn
