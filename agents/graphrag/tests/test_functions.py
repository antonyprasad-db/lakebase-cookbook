"""graph.neighbors, graph.shortest_path and graph.follow: hand answers on FILMS, then random graphs
checked against a plain Python breadth first search."""

from __future__ import annotations

import random
from collections import deque
from itertools import pairwise

import pytest
from conftest import load
from test_differential import TYPES, random_graph


def neighbors(conn, start, hops, types=None, direction="out"):
    rows = conn.execute(
        "SELECT node_id, hops FROM graph.neighbors(%s, %s, %s, %s)", (start, hops, types, direction)
    ).fetchall()
    return {v: h for v, h in rows}


def shortest(conn, src, dst, hops=6, types=None, direction="both"):
    return conn.execute(
        "SELECT graph.shortest_path(%s, %s, %s, %s, %s)", (src, dst, hops, types, direction)
    ).fetchone()[0]


def test_neighbors_by_hand(films):
    assert neighbors(films, "p:alice", 2) == {"p:dan": 1, "p:bob": 1, "p:cara": 2}
    assert neighbors(films, "p:alice", 1, ["KNOWS"], "in") == {"p:cara": 1}
    assert neighbors(films, "p:dan", 3) == {}


def test_shortest_path_by_hand(films):
    assert shortest(films, "f:heat", "p:cara") == ["f:heat", "p:alice", "p:cara"]
    assert shortest(films, "f:heat", "f:up") == ["f:heat", "p:alice", "p:bob", "f:up"]
    assert shortest(films, "f:heat", "p:cara", direction="out") == [
        "f:heat",
        "p:alice",
        "p:bob",
        "p:cara",
    ]
    assert shortest(films, "p:dan", "f:up", hops=1) is None
    assert shortest(films, "p:bob", "p:bob") == ["p:bob"]


def test_follow_by_hand(films):
    rows = films.execute(
        "SELECT node_id, path FROM graph.follow('f:up', ARRAY['DIRECTOR', 'MOTHER'])"
    ).fetchall()
    assert rows == [("p:cara", ["f:up", "p:bob", "p:cara"])]


def test_bad_direction_raises(films):
    with pytest.raises(Exception, match="direction must be"):
        neighbors(films, "p:alice", 1, None, "sideways")


def bfs(g, start, hops, types, direction):
    dist, todo = {start: 0}, deque([start])
    while todo:
        v = todo.popleft()
        if dist[v] == hops:
            continue
        for e, other in g.steps(v, direction):
            if (types is None or e.rel in types) and other not in dist:
                dist[other] = dist[v] + 1
                todo.append(other)
    return dist


def connected(g, a, b, types, direction) -> bool:
    return any(
        other == b and (types is None or e.rel in types) for e, other in g.steps(a, direction)
    )


@pytest.mark.parametrize("seed", range(150))
def test_functions_match_python_bfs(conn, seed):
    rng = random.Random(10_000 + seed)
    g = random_graph(rng)
    load(conn, g)
    ids = list(g.vertices)
    for _ in range(4):
        start, goal = rng.choice(ids), rng.choice(ids)
        hops = rng.randint(1, 4)
        types = rng.sample(TYPES, rng.randint(1, 2)) if rng.random() < 0.5 else None
        direction = rng.choice(["out", "in", "both"])
        dist = bfs(g, start, hops, types, direction)
        assert neighbors(conn, start, hops, types, direction) == {
            v: d for v, d in dist.items() if v != start
        }
        path = shortest(conn, start, goal, hops, types, direction)
        if goal not in dist:
            assert path is None
        else:
            assert len(path) - 1 == dist[goal] and path[0] == start and path[-1] == goal
            assert all(connected(g, a, b, types, direction) for a, b in pairwise(path))
        seq = [rng.choice(TYPES) for _ in range(rng.randint(1, 3))]
        got = sorted(
            conn.execute("SELECT node_id, path FROM graph.follow(%s, %s)", (start, seq)).fetchall()
        )
        want, frontier = [], [[start]]
        for t in seq:
            frontier = [[*p, e.dst] for p in frontier for e in g.out[p[-1]] if e.rel == t]
        want = sorted((p[-1], p) for p in frontier)
        assert got == want
