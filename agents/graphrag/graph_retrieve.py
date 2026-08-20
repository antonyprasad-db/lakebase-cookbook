"""Pure GraphRAG retrieval logic — the in-memory twin of sql/graphrag_retrieval.sql.

Kept dependency-free so the agent's retrieval step is unit-testable offline and so
mock mode needs no Lakebase. The LIVE path (see graphrag_agent.py) runs the SQL against
Lakebase via pgvector HNSW + recursive CTE; this module reproduces the SAME algorithm
(cosine seed -> k-hop expansion both directions -> blended seed_sim * 0.5^hop score) so
behavior matches and is verifiable without infra.
"""
from __future__ import annotations

import math


def cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def retrieve(query_embedding, nodes, edges, embeddings, seed_k=2, max_hops=2, limit=25,
             seed_floor=-1.0, authority_boost=1.0):
    """Mirror of graphrag_retrieval.sql.

    Args:
      query_embedding: list[float]
      nodes: dict node_id -> {"node_type","name","props",...}
      edges: iterable of (src_id, dst_id, rel)
      embeddings: dict node_id -> list[float]
      seed_floor: min cosine similarity a seed must clear. Cosine ranges [-1, 1], so the
        default -1.0 keeps ALL candidates (identical to the pre-floor behavior). Raise it
        to drop weak seeds that drag unrelated subgraphs into the ranking. Calibrate it to
        your embedding model's observed similarity range, not to a fixed number — the useful
        band is model-specific (measured on live Lakebase with databricks-gte-large-en, the
        example graph's similarities span 0.38-0.71, so a floor of 0.3 is a no-op and ~0.55
        is where the distractors drop out).
      authority_boost: multiplier applied to a node whose props carry
        source_method='uc_certified', so a curated definition wins the tie against an
        equally-relevant inferred one. A multiplier rather than a sort key on purpose:
        ordering by (authority, score) would let a barely-relevant certified node outrank a
        highly relevant inferred one. The default 1.0 is an exact no-op — every weight is
        then 1.0 and the ordering matches the pre-authority behaviour, certified rows
        present or not — so this is opt-in like seed_floor.
    Returns: list of dicts {node_id,node_type,name,nearest_hop,rels,score,source_class,
      authority_score}, ordered by the UNROUNDED authority-weighted score then node_id.
      `authority_score` is the rounded value for display; ordering deliberately does not use
      it (see the note in the assemble step).
    """
    # 1) semantic seed — top-k by cosine similarity, above the seed_floor.
    #    `any(emb)` skips a degenerate all-zero embedding: it has no direction, so its
    #    similarity is undefined. The SQL twin excludes the same case by comparing on
    #    cosine DISTANCE, which pgvector returns as NaN for a zero-norm vector (and NaN
    #    would otherwise pass any floor, since NaN > every float in Postgres).
    sims = sorted(
        (
            (nid, sim)
            for nid, emb in embeddings.items()
            if any(emb) and (sim := cosine(query_embedding, emb)) >= seed_floor
        ),
        key=lambda x: x[1], reverse=True,
    )[:seed_k]

    # undirected adjacency (both directions), mirrors the `adj` CTE
    adj: dict = {}
    for s, d, r in edges:
        adj.setdefault(s, []).append((d, r))
        adj.setdefault(d, []).append((s, r))

    # 2) bounded k-hop walk with per-path cycle guard
    best: dict = {}          # node_id -> best graph_score
    nearest: dict = {}       # node_id -> min hop
    rels: dict = {}          # node_id -> set of via rels
    frontier: list = [(nid, 0, sim, (nid,)) for nid, sim in sims]
    while frontier:
        nid, hop, seed_sim, visited = frontier.pop()
        score = seed_sim * (0.5 ** hop)
        if nid not in best or score > best[nid]:
            best[nid] = score
        if nid not in nearest or hop < nearest[nid]:
            nearest[nid] = hop
        if hop >= max_hops:
            continue
        for nxt, rel in adj.get(nid, []):
            if nxt in visited:
                continue
            rels.setdefault(nxt, set()).add(rel)
            frontier.append((nxt, hop + 1, seed_sim, visited + (nxt,)))

    # 3) assemble + rank, then let source authority break the tie between an
    #    equally-relevant certified node and an inferred one.
    #
    #    Every line below has a counterpart in the SQL; keep them in step. In particular:
    #      * `is None` rather than `or`, because COALESCE catches only NULL. A node carrying
    #        {"source_method": ""} must read as "" in BOTH halves, not "" in SQL and
    #        "inferred" here.
    #      * `score > 0` guards the boost. seed_similarity is 1 - cosine_distance so it spans
    #        [-1, 1], and seed_floor's -1.0 default admits negatives; multiplying a negative
    #        score makes it more negative, demoting the certified node the boost exists to
    #        promote (-0.10 * 2.0 = -0.20, below an inferred -0.15).
    #      * a None boost coalesces to 1.0, matching the SQL's COALESCE, rather than raising.
    #      * the sort uses the UNROUNDED key; ordering by the rounded output would collapse
    #        scores differing beyond 4 dp into ties (0.175024 / 0.175006 -> 0.1750) and let
    #        the tiebreak reorder them, which is not a no-op.
    # Clamped to >= 1.0, mirroring GREATEST(:authority_boost, 1.0) in the SQL. `score > 0`
    # below stops a NEGATIVE score being made worse; this stops a boost BELOW 1.0 doing the
    # same thing to a positive one. Together they guarantee a boost can never rank a certified
    # node lower than it would have sat unboosted, whatever the caller passes.
    boost = 1.0 if authority_boost is None else max(authority_boost, 1.0)
    out = []
    for nid, score in best.items():
        if nid not in nodes:
            continue
        n = nodes[nid]
        raw = (n.get("props") or {}).get("source_method")
        source_class = "inferred" if raw is None else raw
        weight = boost if (source_class == "uc_certified" and score > 0) else 1.0
        rank_key = score * weight
        out.append({
            "node_id": nid,
            "node_type": n["node_type"],
            "name": n["name"],
            "nearest_hop": nearest[nid],
            "rels": sorted(rels.get(nid, [])),
            "score": round(score, 4),
            "source_class": source_class,
            "authority_score": round(rank_key, 4),
            "_rank_key": rank_key,
        })
    # Tie broken on node_id for determinism; a boost makes exact ties likelier than the raw
    # scores did. NOTE Postgres orders TEXT by database collation while this compares Unicode
    # codepoints, so the halves agree only under C collation — matters for ids differing in
    # case or punctuation.
    out.sort(key=lambda r: (-r["_rank_key"], r["node_id"]))
    for r in out:
        del r["_rank_key"]
    return out[:limit]
