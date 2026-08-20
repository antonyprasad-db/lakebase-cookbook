-- GraphRAG hybrid retrieval on Lakebase (real Postgres / pgvector)
-- ---------------------------------------------------------------------------
-- The core of GraphRAG: seed with SEMANTIC similarity (pgvector), then EXPAND
-- along relationships (recursive CTE) to pull in graph-connected context that
-- pure vector search would miss. Returns a ranked context set for the agent
-- to synthesize an answer from.
--
-- Parameters (bind at call time from the agent):
--   :query_embedding  VECTOR(1024)  -- embedding of the user question (from your embeddings endpoint)
--   :seed_k           INT           -- how many semantic seed nodes (e.g. 5)
--   :max_hops         INT           -- traversal depth bound (e.g. 2)
--   :seed_floor       FLOAT         -- min cosine similarity a seed must clear. Cosine ranges
--                                      [-1, 1], so -1.0 keeps ALL seeds (same as no floor).
--                                      Raise it to guard against distractor
--                                      pollution: under heavy off-topic content, weak seeds
--                                      drag unrelated subgraphs in and dilute the ranking.
--                                      REQUIRED BIND: Postgres has no server-side default for a
--                                      named parameter, so you must pass this even to keep the
--                                      old behavior — bind -1.0 for "no floor". Omitting it
--                                      raises "bind message supplies 2 parameters, but prepared
--                                      statement requires 3".
--                                      Calibrate it to your embedding model's observed range,
--                                      not to a fixed constant (see README).
--
-- Empty-seed caveat: if the floor exceeds every node's similarity, the seed CTE is
-- empty and this statement returns ZERO rows. Have the caller detect that and either
-- retry at -1.0 or decline to answer — never synthesize from empty context.
--
-- Distance: cosine. pgvector '<=>' returns cosine DISTANCE (0=identical),
-- so similarity = 1 - distance. HNSW index (vector_cosine_ops) backs the ORDER BY.
-- ---------------------------------------------------------------------------

WITH RECURSIVE
-- 1) SEMANTIC SEED — nearest nodes to the question by embedding.
seed AS (
    SELECT
        e.node_id,
        1 - (e.embedding <=> :query_embedding) AS similarity
    FROM graph.node_embeddings e
    -- Drop weak seeds (distractor guard; -1.0 = keep all). Expressed on cosine
    -- DISTANCE rather than similarity — equivalent for real vectors
    -- (sim >= floor  <=>  dist <= 1 - floor) but it also excludes NaN, which
    -- pgvector returns for a zero-norm embedding. NaN compares GREATER than every
    -- float in Postgres, so a similarity form (`1 - dist >= :seed_floor`) admits
    -- such a row at any floor; NaN then propagates through the score and sorts
    -- FIRST under ORDER BY ... DESC, putting a garbage node at rank 1.
    WHERE (e.embedding <=> :query_embedding) <= (1 - :seed_floor)
    ORDER BY e.embedding <=> :query_embedding      -- HNSW ANN scan
    LIMIT :seed_k
),

-- Undirected adjacency: every edge is walkable in both directions so that
-- undirected relations (e.g. SUBSTITUTE_FOR) and reverse questions ("what
-- depends on supplier X") both work. Materialized once, then joined in the
-- recursive term with a PLAIN JOIN (not LATERAL) — Postgres restricts the
-- recursive self-reference, and a plain join against this view is the
-- portable form (and the one the offline smoke test validates).
adj AS (
    SELECT src_id AS from_id, dst_id AS next_id, rel FROM graph.edges
    UNION ALL
    SELECT dst_id AS from_id, src_id AS next_id, rel FROM graph.edges
),

-- 2) GRAPH EXPANSION — walk edges out from each seed up to :max_hops.
--    Track depth to bound the walk and a per-path visited list to stop cycles.
walk AS (
    -- base: the seeds themselves at hop 0, carrying their semantic score
    SELECT
        s.node_id                              AS node_id,
        0                                      AS hop,
        s.similarity                           AS seed_similarity,
        s.node_id                              AS seed_id,
        ARRAY[s.node_id]                       AS visited,
        NULL::TEXT                             AS via_rel
    FROM seed s

    UNION ALL

    -- step: follow any edge touching the frontier node, either direction
    SELECT
        adj.next_id                            AS node_id,
        w.hop + 1                              AS hop,
        w.seed_similarity                      AS seed_similarity,
        w.seed_id                              AS seed_id,
        w.visited || adj.next_id               AS visited,
        adj.rel                                AS via_rel
    FROM walk w
    JOIN adj ON adj.from_id = w.node_id
    WHERE w.hop < :max_hops
      AND NOT (adj.next_id = ANY(w.visited))   -- per-path cycle guard (dedup happens in `scored`)
),

-- 3) SCORE — a node's relevance combines its best seed similarity with a
--    hop-distance decay (closer to a seed = more relevant). If a node is
--    reachable from multiple seeds, keep its strongest path.
scored AS (
    SELECT
        node_id,
        MAX(seed_similarity * POWER(0.5, hop)) AS graph_score,
        MIN(hop)                               AS nearest_hop,
        ARRAY_AGG(DISTINCT via_rel) FILTER (WHERE via_rel IS NOT NULL) AS rels
    FROM walk
    GROUP BY node_id
)

-- 4) ASSEMBLE CONTEXT — join back to node detail, rank by blended score, then let
--    source authority break the tie between an equally-relevant certified node and an
--    inferred one.
--
--    AUTHORITY, and why it is a multiplier rather than a sort key: ordering by
--    (authority, score) would let a barely-relevant certified node outrank a highly
--    relevant inferred one, which is worse than no authority at all. Multiplying keeps
--    relevance meaningful and makes the strength one tunable knob, like `seed_floor`.
--    `:authority_boost` defaults to 1.0, which is an exact no-op: every weight is then
--    1.0 and the ordering is arithmetically identical to the pre-authority behaviour,
--    certified rows present or not. Calibrate it up (1.2–2.0) once a certified core
--    exists; see "Configuration and tuning".
--
--    COALESCE IS LOAD-BEARING — do not remove it. A node whose props carry no
--    `source_method` yields NULL from the CASE, `graph_score * NULL` is NULL, and
--    Postgres sorts NULLs FIRST under ORDER BY ... DESC. That is the same rank-1
--    garbage failure the seed NaN guard above exists to prevent, arriving through a
--    different door. The smoketest pins it.
--    `:authority_boost` is bound ONCE, in the inner SELECT, and the outer query orders by
--    the computed column. Referencing the parameter three times (score, ORDER BY, output)
--    would bind it three times under drivers that do not de-duplicate named parameters,
--    turning one knob into three positional binds a caller has to supply in order.
SELECT
    node_id, node_type, name, props, nearest_hop, rels, score, source_class, authority_score
FROM (
    SELECT
        n.node_id,
        n.node_type,
        n.name,
        n.props,
        s.nearest_hop,
        s.rels,
        ROUND(s.graph_score::numeric, 4)                    AS score,
        -- Surfaced so a caller can cite which source class answered without parsing props.
        -- 'inferred' is the honest default: absence of a marker is not evidence of curation.
        COALESCE(n.props ->> 'source_method', 'inferred')   AS source_class,
        ROUND((s.graph_score * COALESCE(
            CASE WHEN n.props ->> 'source_method' = 'uc_certified'
                 THEN :authority_boost ELSE 1.0 END, 1.0))::numeric, 4) AS authority_score
    FROM scored s
    JOIN graph.nodes n ON n.node_id = s.node_id
) ranked
-- Tie broken on node_id so the order is deterministic across runs; an authority boost makes
-- exact ties materially more likely than the raw scores did. Mirrors the Python twin.
ORDER BY authority_score DESC, node_id
LIMIT 25;
