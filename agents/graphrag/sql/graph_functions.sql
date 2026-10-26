-- Property graph queries over graph.nodes and graph.edges (schema.sql): an exact-match index on
-- name, a both-directions edge view for undirected patterns, and three reusable functions.
-- Plain SQL and PL/pgSQL only, no extension. Apply after schema.sql; safe to re-run.

-- Exact lookups by name (the usual first step of a multi-hop question). The trigram index in
-- schema.sql serves fuzzy matching; equality wants a btree.
CREATE INDEX IF NOT EXISTS nodes_name_idx ON graph.nodes (name);

-- Every edge seen from both ends, for undirected patterns. src_id/dst_id are the direction of
-- travel; key_src/key_dst/rel are the stored edge's key, so a walk can tell that crossing an edge
-- back the other way is the same edge. A self loop appears once, because an undirected Cypher
-- match binds it once; the adjacency CTE in graphrag_retrieval.sql lists it twice, which only
-- affects how retrieval scores a node's own loop.
CREATE OR REPLACE VIEW graph.edges_both AS
    SELECT src_id, dst_id, rel, props, src_id AS key_src, dst_id AS key_dst FROM graph.edges
    UNION ALL
    SELECT dst_id, src_id, rel, props, src_id, dst_id FROM graph.edges WHERE src_id <> dst_id;

-- In the functions below, direction is 'out' (follow src to dst), 'in' (dst to src) or 'both'.

-- Every node reachable from start_id within max_hops, at its shortest hop count. Breadth first,
-- so each node is expanded once however many paths reach it. start_id itself is not returned.
-- The visited set is a jsonb object, so each membership test is a keyed lookup, not a scan of an
-- array that grows to every node reached.
CREATE OR REPLACE FUNCTION graph.neighbors(
    start_id text, max_hops int, rel_types text[] DEFAULT NULL, direction text DEFAULT 'out')
RETURNS TABLE (node_id text, hops int)
LANGUAGE plpgsql STABLE AS $$
DECLARE
    frontier text[] := ARRAY[start_id];
    seen     jsonb;
    depth    int := 0;
BEGIN
    IF direction IS NULL OR direction NOT IN ('out', 'in', 'both') THEN
        RAISE EXCEPTION 'direction must be out, in or both, not %', direction;
    END IF;
    IF start_id IS NULL THEN
        RETURN;
    END IF;
    seen := jsonb_build_object(start_id, 0);
    WHILE depth < max_hops AND cardinality(frontier) > 0 LOOP
        depth := depth + 1;
        SELECT coalesce(array_agg(DISTINCT step.n), '{}') INTO frontier
        FROM (
            SELECT e.dst_id AS n FROM graph.edges e
            WHERE direction IN ('out', 'both') AND e.src_id = ANY (frontier)
              AND (rel_types IS NULL OR e.rel = ANY (rel_types))
            UNION ALL
            SELECT e.src_id FROM graph.edges e
            WHERE direction IN ('in', 'both') AND e.dst_id = ANY (frontier)
              AND (rel_types IS NULL OR e.rel = ANY (rel_types))
        ) AS step
        WHERE NOT seen ? step.n;
        seen := seen || coalesce((SELECT jsonb_object_agg(f, depth) FROM unnest(frontier) AS f), '{}');
        RETURN QUERY SELECT f, depth FROM unnest(frontier) AS f;
    END LOOP;
END $$;

-- One shortest path from src_id to dst_id as an array of node ids, or NULL if none within
-- max_hops. Breadth first; parent maps each node reached to the node it was first reached from,
-- and doubles as the visited set. Ties go to the smallest parent id, so the answer is
-- deterministic.
CREATE OR REPLACE FUNCTION graph.shortest_path(
    src_id text, dst_id text, max_hops int DEFAULT 6, rel_types text[] DEFAULT NULL,
    direction text DEFAULT 'out')
RETURNS text[]
LANGUAGE plpgsql STABLE AS $$
DECLARE
    frontier text[] := ARRAY[src_id];
    parent   jsonb;
    reached  jsonb;
    depth    int := 0;
    cur      text;
    path     text[];
BEGIN
    IF direction IS NULL OR direction NOT IN ('out', 'in', 'both') THEN
        RAISE EXCEPTION 'direction must be out, in or both, not %', direction;
    END IF;
    IF src_id IS NULL OR dst_id IS NULL THEN
        RETURN NULL;
    END IF;
    IF src_id = dst_id THEN  -- the empty path, if the node exists
        RETURN CASE WHEN EXISTS (SELECT 1 FROM graph.nodes n WHERE n.node_id = src_id)
                    THEN ARRAY[src_id] END;
    END IF;
    parent := jsonb_build_object(src_id, NULL);
    WHILE depth < max_hops AND cardinality(frontier) > 0 LOOP
        depth := depth + 1;
        WITH step AS (
            SELECT e.dst_id AS n, e.src_id AS p FROM graph.edges e
            WHERE direction IN ('out', 'both') AND e.src_id = ANY (frontier)
              AND (rel_types IS NULL OR e.rel = ANY (rel_types))
            UNION ALL
            SELECT e.src_id, e.dst_id FROM graph.edges e
            WHERE direction IN ('in', 'both') AND e.dst_id = ANY (frontier)
              AND (rel_types IS NULL OR e.rel = ANY (rel_types))
        ), fresh AS (
            SELECT DISTINCT ON (n) n, p FROM step WHERE NOT parent ? n ORDER BY n, p
        )
        SELECT coalesce(array_agg(n ORDER BY n), '{}'), coalesce(jsonb_object_agg(n, p), '{}')
        INTO frontier, reached FROM fresh;
        parent := parent || reached;
        IF reached ? dst_id THEN
            cur := dst_id;
            path := ARRAY[dst_id];
            WHILE cur <> src_id LOOP
                cur := parent ->> cur;
                path := cur || path;
            END LOOP;
            RETURN path;
        END IF;
    END LOOP;
    RETURN NULL;
END $$;

-- Follow a fixed sequence of relationship types out of start_id, one row per path. This is the
-- multi hop question shape: follow('film:x', ARRAY['DIRECTOR', 'MOTHER']) is the mother of the
-- director of film x.
CREATE OR REPLACE FUNCTION graph.follow(start_id text, rel_path text[])
RETURNS TABLE (node_id text, path text[])
LANGUAGE sql STABLE AS $$
    WITH RECURSIVE walk (vertex_id, step, path) AS (
        SELECT start_id, 0, ARRAY[start_id]
        WHERE EXISTS (SELECT 1 FROM graph.nodes n WHERE n.node_id = start_id)
        UNION ALL
        SELECT e.dst_id, walk.step + 1, walk.path || e.dst_id
        FROM walk JOIN graph.edges e ON e.src_id = walk.vertex_id AND e.rel = rel_path[walk.step + 1]
        WHERE walk.step < cardinality(rel_path)
    )
    SELECT vertex_id AS node_id, path FROM walk WHERE step = cardinality(rel_path);
$$;
