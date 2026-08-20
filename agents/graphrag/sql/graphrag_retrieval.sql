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
--                                      raises a "bind message supplies N parameters, but prepared
--                                      statement requires N+1" error. With :authority_boost added
--                                      there are now FIVE binds, so omitting one reports 4 vs 5.
--                                      Calibrate it to your embedding model's observed range,
--                                      not to a fixed constant (see README).
--   :authority_boost  NUMERIC       -- multiplier for a node whose props carry
--                                      source_method='uc_certified', so a curated definition wins
--                                      the tie against an equally-relevant inferred one. Bind 1.0
--                                      to rank exactly as before. REQUIRED BIND, same reason as
--                                      :seed_floor. Bind it as NUMERIC, not text: the CASE's
--                                      GREATEST resolves the type first, so a text bind fails at
--                                      plan time with "GREATEST types text and numeric cannot be
--                                      matched". Values below 1.0 are clamped up to 1.0 (see the
--                                      assemble stage) because a fractional or negative
--                                      multiplier would DEMOTE the certified node it is meant to
--                                      promote.
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

-- 4) ASSEMBLE CONTEXT — join back to node detail, rank by relevance, then let source
--    authority break the tie between an equally-relevant certified node and an inferred one.
--
--    AUTHORITY IS A MULTIPLIER, NOT A SORT KEY. Ordering by (authority, score) would let a
--    barely-relevant certified node outrank a highly relevant inferred one, which is worse
--    than no authority signal at all. Multiplying keeps relevance meaningful and makes the
--    strength one tunable knob, like `seed_floor`.
--
--    ONLY POSITIVE SCORES ARE BOOSTED, and that guard is load-bearing. `seed_similarity` is
--    `1 - cosine_distance`, so it spans [-1, 1], and the default `seed_floor = -1.0` admits
--    negative-similarity seeds by design. Multiplying a negative score makes it MORE
--    negative: a certified node at -0.10 with a boost of 2.0 becomes -0.20 and sorts BELOW
--    an inferred node at -0.15 — the boost would demote exactly what it exists to promote,
--    and worsen as the knob is raised. Boosting only above zero keeps the transform monotone.
--
--    `:authority_boost` defaults to 1.0, which leaves the weight at 1.0 for every row and the
--    ordering identical to the pre-authority behaviour, certified rows present or not. Note
--    the ORDER BY uses the UNROUNDED product on purpose: ordering by the rounded output
--    column would collapse scores that differ beyond 4 dp into ties (0.175024 and 0.175006
--    both round to 0.1750) and let the tiebreak reorder them, which is not a no-op.
--
--    A NULL bind needs no COALESCE: Postgres GREATEST ignores NULL inputs and returns NULL only
--    if EVERY argument is NULL, so GREATEST(NULL, 1.0) is 1.0. That means a NULL
--    :authority_boost already degrades to "no boost" rather than making the score NULL (which
--    would sort FIRST under ORDER BY ... DESC and hand rank 1 to a garbage row). An earlier
--    revision wrapped this in COALESCE for that reason; the clamp made it dead code.
SELECT
    node_id, node_type, name, props, nearest_hop, rels, score, source_class,
    -- Rounded for display only. The ORDER BY below deliberately uses the UNROUNDED key.
    ROUND(authority_rank_key::numeric, 4) AS authority_score
FROM (
    SELECT
        n.node_id,
        n.node_type,
        n.name,
        n.props,
        s.nearest_hop,
        s.rels,
        ROUND(s.graph_score::numeric, 4)                    AS score,
        -- Passed through verbatim, NOT normalised to a two-value domain, so that any producer's
        -- own label survives. Today nothing in this repo writes source_method into NODE props,
        -- so in practice you will see only 'uc_certified' or 'inferred' — the 'structured' /
        -- 'llm_enrichment' values in sql/gold_triplets_mapping.sql are EDGE provenance
        -- (e.props), a different table, and are not a live source of node values. The
        -- pass-through is future-proofing, not a current collision. Only 'uc_certified' is
        -- boosted; every other value, known or not, carries weight 1.0. Branch on
        -- == 'uc_certified', never on != 'inferred'.
        COALESCE(n.props ->> 'source_method', 'inferred')   AS source_class,
        -- GREATEST(..., 1.0) is the second half of the monotonicity guarantee. The
        -- graph_score > 0 test stops a NEGATIVE score from being made worse; this stops a
        -- boost BELOW 1.0 from doing the same thing on a positive score. Measured without it:
        -- certified 1.00 vs inferred 0.90 at boost 0.5 ranks the inferred node first, and at
        -- -2.0 the certified node lands at -2.00. Clamping means a boost can never rank a
        -- certified node below where it would have sat unboosted, whatever is bound.
        s.graph_score *
            CASE WHEN n.props ->> 'source_method' = 'uc_certified' AND s.graph_score > 0
                 THEN GREATEST(:authority_boost, 1.0) ELSE 1.0 END   AS authority_rank_key
    FROM scored s
    JOIN graph.nodes n ON n.node_id = s.node_id
) ranked
-- Tie broken on node_id for determinism across runs; a boost makes exact ties likelier than
-- the raw scores did. NOTE this orders TEXT by the database collation while the Python twin
-- compares Unicode codepoints, so the two agree only under C collation — it matters just for
-- ids differing in case or punctuation.
ORDER BY authority_rank_key DESC, node_id
LIMIT 25;
