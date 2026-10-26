"""Cross check against Postgres's own SQL/PGQ, as it stood at 6168c65, the last commit before the
2026-09-07 revert. Set PGRAPH_PGQ_DSN to a throwaway database on a server built from that commit
(with pgvector, ltree and pg_trgm installed, as schema.sql needs them) to run; it is skipped
otherwise, since no released Postgres has SQL/PGQ.

That build supports a single fixed length path pattern and lets one edge match twice in a path, so
the comparison is on that subset with our relationship uniqueness switched off: same rows, row for
row. The second test measures how often Cypher's uniqueness rule changes the answer."""

from __future__ import annotations

import os
import random

import psycopg
import pytest
from conftest import install, load
from test_differential import LABELS, NAMES, TYPES, random_graph

from pgcypher import compiler
from pgcypher.reference import canon

SEEDS = range(int(os.environ.get("PGRAPH_PGQ_CASES", "200")))


@pytest.fixture(scope="module")
def pgq():
    dsn = os.environ.get("PGRAPH_PGQ_DSN")
    if not dsn:
        pytest.skip("set PGRAPH_PGQ_DSN to a Postgres built at 6168c65, before the SQL/PGQ revert")
    with psycopg.connect(dsn, autocommit=True) as c:
        install(c)
        # SQL/PGQ maps a graph onto one table per label, so each node type and each
        # (type, source type, destination type) combination of edges becomes a view
        for lab in LABELS:
            c.execute(
                f"CREATE VIEW graph.v_{lab.lower()} AS SELECT node_id AS id, "
                f"(props->>'x')::int AS x, name, (props->>'flag')::boolean AS flag "
                f"FROM graph.nodes WHERE node_type = '{lab}'"
            )
        edge_tables = []
        for t in TYPES:
            for s in LABELS:
                for d in LABELS:
                    name = f"e_{t}_{s}_{d}".lower()
                    c.execute(
                        f"CREATE VIEW graph.{name} AS SELECT e.src_id AS src, e.dst_id AS dst, "
                        f"(e.props->>'w')::int AS w FROM graph.edges e "
                        f"JOIN graph.nodes vs ON vs.node_id = e.src_id "
                        f"JOIN graph.nodes vd ON vd.node_id = e.dst_id "
                        f"WHERE e.rel = '{t}' AND vs.node_type = '{s}' AND vd.node_type = '{d}'"
                    )
                    # one rel per view, so (src, dst) is the edge key within it
                    edge_tables.append(
                        f"graph.{name} KEY (src, dst) "
                        f"SOURCE KEY (src) REFERENCES v_{s.lower()} (id) "
                        f"DESTINATION KEY (dst) REFERENCES v_{d.lower()} (id) LABEL {t.lower()}"
                    )
        vertex_tables = ", ".join(
            f"graph.v_{lab.lower()} KEY (id) LABEL {lab.lower()}" for lab in LABELS
        )
        c.execute(
            f"CREATE PROPERTY GRAPH graph.g VERTEX TABLES ({vertex_tables}) "
            f"EDGE TABLES ({', '.join(edge_tables)})"
        )
        yield c


def random_pair(rng: random.Random) -> tuple[str, str]:
    """One fixed length path pattern written twice: as openCypher and as SQL/PGQ."""
    hops = rng.randint(
        1, 2
    )  # unlabeled three hop patterns take seconds each to plan in the PGQ build
    names = [f"n{i}" for i in range(hops + 1)]
    cy, pg = [], []
    for i, v in enumerate(names):
        lab = rng.choice(LABELS) if rng.random() < 0.35 else None
        cy.append(f"({v}{':' + lab if lab else ''})")
        pg.append(f"({v}{' IS ' + lab.lower() if lab else ''})")
        if i < hops:
            t = rng.choice(TYPES) if rng.random() < 0.6 else None
            d = rng.choice(["out", "in", "both"])
            cb, pb = f"[{':' + t if t else ''}]", f"[{'IS ' + t.lower() if t else ''}]"
            cy.append({"out": f"-{cb}->", "in": f"<-{cb}-", "both": f"-{cb}-"}[d])
            pg.append({"out": f"-{pb}->", "in": f"<-{pb}-", "both": f"-{pb}-"}[d])
    where = []
    for _ in range(rng.randint(0, 2)):
        v = rng.choice(names)
        where.append(
            rng.choice(
                [
                    f"{v}.x {rng.choice(['=', '<', '>='])} {rng.randint(0, 5)}",
                    f"{v}.name = '{rng.choice(NAMES)}'",
                    f"{v}.flag IS NULL",
                    f"{v}.x < {rng.choice(names)}.x",
                ]
            )
        )
    cond = f" WHERE {' AND '.join(where)}" if where else ""
    cols = ", ".join(f"{v} AS {v}" for v in names)
    cypher = f"MATCH {''.join(cy)}{cond} RETURN {cols}"
    sql = (
        f"SELECT * FROM GRAPH_TABLE (graph.g MATCH {''.join(pg)}{cond} "
        f"COLUMNS ({', '.join(f'{v}.id AS {v}' for v in names)}))"
    )
    return cypher, sql


def rows(result) -> list[str]:
    return sorted(canon(list(r)) for r in result)


@pytest.mark.parametrize("seed", SEEDS)
def test_matches_pgq_without_uniqueness(pgq, seed, monkeypatch):
    rng = random.Random(20_000 + seed)
    load(pgq, random_graph(rng))
    monkeypatch.setattr(compiler._Compiler, "uniqueness", lambda self: None)
    for _ in range(4):
        cypher, sql = random_pair(rng)
        assert rows(compiler.run(pgq, cypher)) == rows(pgq.execute(sql).fetchall()), (
            f"{cypher}\n{sql}"
        )


def test_how_often_uniqueness_changes_the_answer(pgq):
    rng, differ, total = random.Random(30_000), 0, 0
    for _ in range(100):
        load(pgq, random_graph(rng))
        for _ in range(4):
            cypher, sql = random_pair(rng)
            total += 1
            differ += rows(compiler.run(pgq, cypher)) != rows(pgq.execute(sql).fetchall())
    print(f"\nCypher uniqueness changed the answer on {differ} of {total} random queries")
    assert differ > 0
