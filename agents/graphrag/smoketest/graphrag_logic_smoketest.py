"""Offline smoke test for the GraphRAG hybrid-retrieval LOGIC (no live infra).

WHAT THIS PROVES (offline, no Databricks / no Lakebase / no model endpoint):
  1. The vector-seed step ranks the semantically-closest node first.
  2. The recursive-CTE graph expansion pulls in graph-connected context that a
     pure vector search would MISS — the whole point of GraphRAG — bounded by
     :max_hops with a working cycle guard.
  3. The blended score (seed_similarity * 0.5^hop) ranks seeds above their
     neighbors and decays with hop distance.
  4. The BUILD-side graph assembly (notebooks/graph_build.assemble_graph) turns
     structured dims + LLM enrichment into the intended nodes/edges.
  5. The `seed_floor` distractor guard drops weak semantic seeds before the walk,
     and its -1.0 default leaves the pre-floor behavior unchanged.
  6. The UPSTREAM indexing path (graph_upstream): chunking keeps stable ids and
     overlap, trigram entity resolution collapses alias spellings into one node,
     and the emitted rows conform to the 8-column gold_triplets contract.

WHAT THIS DOES NOT COVER (needs live Lakebase, tracked in README):
  - pgvector's actual HNSW ANN operator (<=>) and index behavior — here we
    compute cosine in Python and seed deterministically, then run the SAME
    recursive-CTE traversal in DuckDB (SQL dialect ~equivalent for this logic).
  - Real model embeddings — we use a tiny deterministic bag-of-words vector.
  - Sync from UC / permissions / scale.

Run: uv run --python 3.11 --with duckdb --with numpy graphrag_logic_smoketest.py
"""
from __future__ import annotations

import sys

import duckdb
import numpy as np

FAILURES: list[str] = []


def check(name: str, got, want) -> None:
    ok = got == want
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got={got!r} want={want!r}")
    if not ok:
        FAILURES.append(name)


# --------------------------------------------------------------------------- graph
# A small retail supply-chain KG (matches the README anchor). node_id = 'type:key'.
NODES = [
    ("product:122", "Product",  "Skim Milk 2L"),
    ("product:130", "Product",  "Whole Milk 2L"),        # substitute for skim milk
    ("supplier:S2", "Supplier", "Supplier S2 (Boston)"),
    ("supplier:S3", "Supplier", "Supplier S3 (Boston)"),
    ("region:ne",   "Region",   "Northeast"),
    ("category:dairy", "Category", "Dairy"),
    ("product:400", "Product",  "Diesel Generator"),      # unrelated island (control)
    ("supplier:S9", "Supplier", "Supplier S9 (Denver)"),  # only supplies the generator
]
EDGES = [
    ("product:122", "supplier:S2", "SUPPLIED_BY"),
    ("product:122", "supplier:S3", "SUPPLIED_BY"),
    ("product:122", "category:dairy", "BELONGS_TO"),
    ("product:122", "product:130", "SUBSTITUTE_FOR"),
    ("product:130", "supplier:S2", "SUPPLIED_BY"),
    ("supplier:S2", "region:ne", "LOCATED_IN"),
    ("supplier:S3", "region:ne", "LOCATED_IN"),
    ("product:122", "region:ne", "SURGES_IN"),
    ("product:400", "supplier:S9", "SUPPLIED_BY"),         # unrelated island
]

# Deterministic tiny "embedding": bag-of-words over a fixed vocab, L2-normalized.
VOCAB = ["skim", "milk", "whole", "supplier", "boston", "denver", "northeast",
         "dairy", "diesel", "generator", "region", "category"]


def embed(text: str) -> np.ndarray:
    v = np.zeros(len(VOCAB), dtype=float)
    toks = text.lower().replace("(", " ").replace(")", " ").split()
    for i, w in enumerate(VOCAB):
        v[i] = sum(1.0 for t in toks if t == w)
    n = np.linalg.norm(v)
    return v / n if n else v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # both normalized


def main() -> int:
    con = duckdb.connect()
    con.execute("CREATE TABLE nodes(node_id TEXT, node_type TEXT, name TEXT)")
    con.executemany("INSERT INTO nodes VALUES (?,?,?)", NODES)
    con.execute("CREATE TABLE edges(src_id TEXT, dst_id TEXT, rel TEXT)")
    con.executemany("INSERT INTO edges VALUES (?,?,?)", EDGES)

    # ---- 1) SEMANTIC SEED (computed in Python, as a stand-in for pgvector <=>)
    q = embed("skim milk demand in the northeast")
    sims = [(nid, cosine(q, embed(name))) for nid, _t, name in NODES]
    sims.sort(key=lambda x: x[1], reverse=True)
    seed_k = 2
    seeds = sims[:seed_k]
    print("\n== TEST 1: semantic seed ==")
    for nid, s in seeds:
        print(f"    seed {nid}: sim={s:.3f}")
    check("top semantic seed is skim milk", seeds[0][0], "product:122")

    # feed seeds into DuckDB for the SAME recursive-CTE expansion the .sql uses
    con.execute("CREATE TABLE seed(node_id TEXT, similarity DOUBLE)")
    con.executemany("INSERT INTO seed VALUES (?,?)", [(nid, float(s)) for nid, s in seeds])

    max_hops = 2
    # Mirrors the `adj` / `walk` / `scored` stages of sql/graphrag_retrieval.sql: an `adj`
    # CTE (edges walkable both directions) joined with a PLAIN JOIN in the recursive term.
    # DuckDB dialect swaps: ARRAY[x]->[x], `||`->list_append, `= ANY`->list_contains.
    # It deliberately does NOT mirror the final assemble stage — authority ranking is
    # covered separately in TEST 8 against its own fixture, so this stays focused on traversal.
    walk_sql = f"""
    WITH RECURSIVE adj AS (
        SELECT src_id AS from_id, dst_id AS next_id, rel FROM edges
        UNION ALL
        SELECT dst_id AS from_id, src_id AS next_id, rel FROM edges
    ),
    walk AS (
        SELECT s.node_id AS node_id, 0 AS hop, s.similarity AS seed_similarity,
               s.node_id AS seed_id, [s.node_id] AS visited
        FROM seed s
        UNION ALL
        SELECT adj.next_id, w.hop + 1, w.seed_similarity, w.seed_id,
               list_append(w.visited, adj.next_id)
        FROM walk w
        JOIN adj ON adj.from_id = w.node_id
        WHERE w.hop < {max_hops}
          AND NOT list_contains(w.visited, adj.next_id)
    ),
    scored AS (
        SELECT node_id,
               MAX(seed_similarity * POWER(0.5, hop)) AS graph_score,
               MIN(hop) AS nearest_hop
        FROM walk GROUP BY node_id
    )
    SELECT s.node_id, n.node_type, n.name, s.nearest_hop,
           ROUND(s.graph_score, 4) AS score
    FROM scored s JOIN nodes n ON n.node_id = s.node_id
    ORDER BY s.graph_score DESC
    """
    rows = con.execute(walk_sql).fetchall()
    retrieved = {r[0]: r for r in rows}
    print("\n== TEST 2: graph expansion (recursive CTE, max_hops=2) ==")
    for r in rows:
        print(f"    {r[0]:16s} type={r[1]:9s} hop={r[3]} score={r[4]}")

    # 2a) GraphRAG pulls in connected context a flat vector search misses.
    # Suppliers S2/S3 and the substitute product 130 are NOT semantically similar
    # to "skim milk demand" but ARE graph-connected to the seed → must appear.
    for expected in ["supplier:S2", "supplier:S3", "product:130", "region:ne", "category:dairy"]:
        check(f"graph surfaced {expected}", expected in retrieved, True)

    # 2b) The unrelated island (diesel generator / S9) must NOT be pulled in.
    check("unrelated diesel generator excluded", "product:400" not in retrieved, True)
    check("unrelated supplier S9 excluded", "supplier:S9" not in retrieved, True)

    # 2c) Seed ranks above its neighbors (hop-decay works).
    milk = retrieved["product:122"][4]
    s2 = retrieved["supplier:S2"][4]
    check("seed scores above its 1-hop neighbor", milk > s2, True)

    # 2d) Depth bound respected: nothing beyond max_hops.
    check("no node deeper than max_hops", max(r[3] for r in rows) <= max_hops, True)

    con.close()

    # ---- 3) BUILD-side graph assembly (notebooks/graph_build.assemble_graph)
    print("\n== TEST 3: graph assembly from dims + enrichment ==")
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from graph_build import assemble_graph  # noqa: E402

    products = [
        {"product_id": 122, "name": "Skim Milk 2L", "category": "Dairy"},
        {"product_id": 130, "name": "Whole Milk 2L", "category": "Dairy"},
        {"product_id": 261, "name": "Diesel Generator", "category": "Equipment"},
    ]
    b_suppliers = [
        {"supplier_id": "S2", "name": "Supplier S2", "city": "Boston", "region": "Northeast"},
        {"supplier_id": "S3", "name": "Supplier S3", "city": "Boston", "region": "Northeast"},
        {"supplier_id": "S9", "name": "Supplier S9", "city": "Denver", "region": "West"},
    ]
    b_supplies = [
        {"product_id": 122, "supplier_id": "S2"}, {"product_id": 122, "supplier_id": "S3"},
        {"product_id": 130, "supplier_id": "S2"}, {"product_id": 261, "supplier_id": "S9"},
    ]
    b_surges = [{"product_id": 122, "region": "Northeast", "surge_ratio": 1.69}]
    b_enrich = {
        122: {"description": "Skim Milk 2L — a dairy product.", "substitute_ids": [130]},
        130: {"description": "Whole Milk 2L — a dairy product.", "substitute_ids": [122]},
        261: {"description": "Diesel Generator — an equipment product.", "substitute_ids": []},
    }
    bnodes, bedges = assemble_graph(products, b_suppliers, b_supplies, b_surges, b_enrich)

    # node_id convention + all expected node types present
    for nid in ["product:122", "supplier:S2", "region:northeast", "category:dairy"]:
        check(f"assembled node {nid}", nid in bnodes, True)
    # substitute stored once, sorted direction (product:122 -> product:130)
    check("substitute edge stored once (sorted)",
          ("product:122", "product:130", "SUBSTITUTE_FOR") in bedges
          and ("product:130", "product:122", "SUBSTITUTE_FOR") not in bedges, True)
    # the surge signal became an edge
    check("surge edge present", ("product:122", "region:northeast", "SURGES_IN") in bedges, True)
    # supplied_by wired from the availability feed
    check("supplied_by edge present", ("product:122", "supplier:S2", "SUPPLIED_BY") in bedges, True)
    # region carries an ltree-ready path
    check("region has ltree path", bnodes["region:northeast"]["path"], "us.northeast")

    # ---- 4) DANGLING-REFERENCE GUARD (Finding 1): hallucinated / out-of-dim ids
    #         must NOT produce edges to non-existent nodes (would break the FK on sync).
    print("\n== TEST 4: dangling-reference guard ==")
    bad_enrich = {
        122: {"description": "Skim Milk 2L.", "substitute_ids": [999]},  # 999 not in products
        130: {"description": "Whole Milk 2L.", "substitute_ids": [122]},
        261: {"description": "Diesel Generator.", "substitute_ids": []},
    }
    bad_supplies = b_supplies + [{"product_id": 888, "supplier_id": "S2"}]  # 888 unknown product
    bad_surges = b_surges + [{"product_id": 777, "region": "West"}]         # 777 unknown product
    gnodes, gedges = assemble_graph(products, b_suppliers, bad_supplies, bad_surges, bad_enrich)
    # every edge endpoint must exist as a node
    orphans = [(s, d, r) for (s, d, r) in gedges if s not in gnodes or d not in gnodes]
    check("no orphan edges despite bad input", orphans, [])
    # the good substitute (130<->122) still forms; the hallucinated 999 does not
    check("valid substitute still present",
          ("product:122", "product:130", "SUBSTITUTE_FOR") in gedges, True)
    check("hallucinated substitute 999 dropped",
          any("999" in s or "999" in d for (s, d, _r) in gedges), False)

    # ---- 5) query module (graph_retrieve.retrieve) matches the SQL logic
    print("\n== TEST 5: query module (pure retrieve) ==")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from graph_retrieve import retrieve as agent_retrieve  # noqa: E402

    # reuse the assembled graph from TEST 3; embed nodes with the same bag-of-words as agent
    agent_vocab = ["skim", "milk", "whole", "supplier", "boston", "denver", "northeast",
                   "dairy", "diesel", "generator", "region", "category", "product"]

    def bow(text):
        toks = text.lower().replace(":", " ").split()
        v = [float(sum(1 for t in toks if t == w)) for w in agent_vocab]
        n = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / n for x in v]

    a_emb = {nid: bow(f"{n['node_type']} {n['name']}") for nid, n in bnodes.items()}
    q = bow("skim milk northeast supplier")
    ranked = agent_retrieve(q, bnodes, bedges, a_emb, seed_k=2, max_hops=2)
    ranked_ids = [r["node_id"] for r in ranked]
    check("retrieve returns rows", len(ranked) > 0, True)
    check("top result is skim milk", ranked[0]["node_id"], "product:122")
    check("expands to a supplier", "supplier:S2" in ranked_ids, True)
    check("excludes unrelated generator", "product:261" not in ranked_ids, True)

    # ---- 6) seed_floor: the distractor guard on the semantic seed
    print("\n== TEST 6: seed_floor distractor guard ==")
    # Cosine similarities for this query against the bag-of-words embeddings:
    #   product:122 0.577 | supplier:S2/S3/S9 0.500 | region:northeast 0.354
    #   product:130 0.289 | category:dairy, product:261, region:west, ... 0.000
    # supplier:S9 is the distractor bridge — it clears a loose floor on the word
    # "supplier" alone, but the only thing it supplies is the unrelated diesel
    # generator, so an unfloored seed drags that whole subgraph into the ranking.
    wide = agent_retrieve(q, bnodes, bedges, a_emb, seed_k=6, max_hops=2)
    wide_ids = [r["node_id"] for r in wide]
    check("no floor: distractor generator pulled in", "product:261" in wide_ids, True)
    check("no floor: distractor region pulled in", "region:west" in wide_ids, True)

    # 0.55 admits only product:122 (0.577) — supplier:S9 (0.500) no longer seeds
    floored = agent_retrieve(q, bnodes, bedges, a_emb, seed_k=6, max_hops=2, seed_floor=0.55)
    floored_ids = [r["node_id"] for r in floored]
    check("floor 0.55: distractor generator excluded", "product:261" not in floored_ids, True)
    check("floor 0.55: distractor region excluded", "region:west" not in floored_ids, True)
    check("floor 0.55: on-topic seed still ranks first",
          floored[0]["node_id"] if floored else None, "product:122")
    check("floor 0.55: narrows the result set", len(floored) < len(wide), True)
    check("floor 0.55: keeps the on-topic supplier via graph expansion",
          "supplier:S2" in floored_ids, True)

    # Backward compatibility, pinned to the EXPECTED set rather than to another call
    # with the same default (which would be tautological). These 10 ids are the whole
    # graph — the pre-floor code returned exactly this for seed_k=6, max_hops=2.
    expected_unfloored = [
        "category:dairy", "category:equipment", "product:122", "product:130",
        "product:261", "region:northeast", "region:west",
        "supplier:S2", "supplier:S3", "supplier:S9",
    ]
    check("default floor reproduces the pre-floor result set",
          sorted(wide_ids), expected_unfloored)

    # a floor above every candidate's similarity admits no seeds at all -> zero rows.
    # NOTE this is the documented failure mode of an over-set floor: the caller must
    # detect the empty result and retry at -1.0 rather than answer from no context.
    check("floor above max similarity returns nothing",
          agent_retrieve(q, bnodes, bedges, a_emb, seed_k=6, seed_floor=0.99), [])

    # a degenerate all-zero embedding has undefined similarity and must never seed.
    # In SQL, pgvector returns NaN for it and NaN > every float, so a similarity-form
    # floor would admit it and NaN would sort FIRST; both sides exclude it instead.
    zero_emb = dict(a_emb)
    zero_emb["product:261"] = [0.0] * len(q)
    zeroed = agent_retrieve(q, bnodes, bedges, zero_emb, seed_k=6, max_hops=2, seed_floor=-1.0)
    check("all-zero embedding never becomes a seed",
          "product:261" not in [r["node_id"] for r in zeroed[:1]], True)

    # ---- 7) UPSTREAM indexing: documents -> gold_triplets
    print("\n== TEST 7: upstream indexing (chunk -> extract -> resolve -> triplets) ==")
    from graph_upstream import (  # noqa: E402
        DEFAULT_PREDICATES,
        SYMMETRIC_PREDICATES,
        Mention,
        build_gold_triplets,
        canonical_id,
        chunk,
        index_document,
        normalize_predicate,
        resolve_entities,
        trigram_containment,
        trigram_sim,
    )

    # -- chunking: stable ids, sentence-aware packing, bounded size, word-safe overlap
    doc = ("Acme Foods supplies skim milk. ACME Foods Inc. is based in Boston. "
           "Northeast demand for skim milk is surging. Acme ships from Boston daily. "
           "Whole milk substitutes for skim milk in most recipes.")
    passages = chunk("doc:1", doc, size=80, overlap=20)
    check("chunking produced multiple passages", len(passages) > 1, True)
    check("chunk ids are stable and ordered",
          [p.chunk_id for p in passages][:2], ["doc:1:c0", "doc:1:c1"])
    check("every passage keeps its doc pointer",
          all(p.doc_id == "doc:1" for p in passages), True)
    check("no passage is empty", all(p.text.strip() for p in passages), True)
    # a run with NO sentence terminator must still be bounded: parsed PDFs (tables, slides, OCR)
    # routinely contain these, and extraction sends one model call per passage.
    unpunctuated = chunk("doc:2", "word " * 200, size=60)
    check("text without sentence terminators is still split",
          len(unpunctuated) > 1, True)
    # The guarantee that matters is that a long unpunctuated run does NOT become one
    # whole-document passage sent to the model as a single prompt. The exact ceiling is
    # size plus carried overlap plus the last packed segment, so bound it at 2x size
    # rather than asserting an off-by-a-word figure.
    check("no passage approaches the whole document",
          max(len(p.text) for p in unpunctuated) <= 2 * 60, True)
    check("a 1000-char unpunctuated run yields many passages, not one",
          len(unpunctuated) >= 10, True)
    check("overlap does not start a passage mid-word",
          all(not p.text.startswith(" ") for p in passages), True)

    # -- trigram measures
    check("alias pair scores above the merge threshold",
          trigram_sim("Acme Foods", "ACME Foods Inc.") >= 0.55, True)
    check("unrelated pair scores below the merge threshold",
          trigram_sim("Acme Foods", "Diesel Generator") < 0.55, True)
    check("trigram similarity is symmetric",
          round(trigram_sim("Acme", "ACME Foods"), 6)
          == round(trigram_sim("ACME Foods", "Acme"), 6), True)
    check("Jaccard MISSES the legal-suffix alias (why containment exists)",
          trigram_sim("Acme Foods", "Acme Foods Incorporated") < 0.55, True)
    check("containment catches the legal-suffix alias",
          trigram_containment("Acme Foods", "Acme Foods Incorporated") >= 0.8, True)
    check("containment still rejects an unrelated pair",
          trigram_containment("Acme Foods", "Diesel Generator") < 0.8, True)

    # -- entity resolution: three spellings of one company collapse to a single canonical node
    mentions = [
        Mention("Acme Foods", "Company", "SUPPLIED_BY", "Skim Milk 2L", "Product", 0.9, "doc:1:c0"),
        Mention("ACME Foods Inc.", "Company", "LOCATED_IN", "Boston", "Location", 0.8, "doc:1:c1"),
        Mention("Acme", "Company", "LOCATED_IN", "Boston", "Location", 0.7, "doc:1:c3"),
    ]
    res = resolve_entities(mentions)
    company_ids = {res.canonical_id[(n, "Company")]
                   for n in ("Acme Foods", "ACME Foods Inc.", "Acme")}
    check("three alias spellings collapse to ONE canonical id", len(company_ids), 1)
    check("canonical display name is the fullest spelling",
          res.canonical_name[company_ids.pop()], "ACME Foods Inc.")
    check("each merged alias is recorded for audit", len(res.same_as), 2)
    check("canonical ids carry a type prefix (graph_build convention)",
          res.canonical_id[("Acme Foods", "Company")].startswith("company:"), True)

    # -- same NAME, different TYPE must stay distinct. Resolution is keyed by (surface, type),
    #    so the two get separate ids; keying on the surface alone silently merged them into one
    #    graph.nodes row with two conflicting node_types.
    dual = [
        Mention("Acme Foods", "Company", "LOCATED_IN", "Boston", "Location", 0.9, "c0"),
        Mention("Acme Foods", "Product", "BELONGS_TO", "Dairy", "Category", 0.9, "c1"),
    ]
    dres = resolve_entities(dual)
    check("same name + different type get DIFFERENT canonical ids",
          dres.canonical_id[("Acme Foods", "Company")]
          != dres.canonical_id[("Acme Foods", "Product")], True)
    drows = build_gold_triplets(dual, dres)
    by_id = {}
    for r in drows:
        by_id.setdefault(r["subject_id"], set()).add(r["subject_type"])
        by_id.setdefault(r["object_id"], set()).add(r["object_type"])
    check("no node id carries two conflicting types",
          all(len(v) == 1 for v in by_id.values()), True)

    # -- contract conformance
    CONTRACT = {"subject_id", "subject_type", "predicate", "object_id", "object_type",
                "confidence", "source_method", "source_agent"}
    rows = build_gold_triplets(mentions, res)
    check("every row carries exactly the 8 contract columns",
          all(set(r) == CONTRACT for r in rows), True)
    check("relations are attributed to llm_extract",
          {r["source_method"] for r in rows if r["predicate"] != "SAME_AS"}, {"llm_extract"})
    check("alias edges are attributed to entity_resolution",
          {r["source_method"] for r in rows if r["predicate"] == "SAME_AS"}, {"entity_resolution"})
    check("subjects/objects were rewritten to canonical ids",
          all(":" in r["subject_id"] and ":" in r["object_id"] for r in rows), True)
    # SAME_AS rows must carry the node's REAL type, not a literal "Entity"
    same_as_types = {r["subject_type"] for r in rows if r["predicate"] == "SAME_AS"}
    check("SAME_AS rows carry the resolved type, not 'Entity'",
          "Entity" not in same_as_types, True)
    # no self-loops: case/punctuation variants slug identically and would violate the
    # (src_id, dst_id, rel) primary key on graph.edges
    variants = [
        Mention("ACME Foods Inc.", "Company", "LOCATED_IN", "Boston", "Location", 0.9, "c0"),
        Mention("ACME Foods Inc", "Company", "LOCATED_IN", "Boston", "Location", 0.9, "c1"),
        Mention("acme foods inc", "Company", "LOCATED_IN", "Boston", "Location", 0.9, "c2"),
    ]
    vrows = build_gold_triplets(variants, resolve_entities(variants))
    check("no self-referential rows (would break the edges primary key)",
          [r for r in vrows if r["subject_id"] == r["object_id"]], [])
    check("no duplicate (subject, predicate, object) rows",
          len({(r["subject_id"], r["predicate"], r["object_id"]) for r in vrows}), len(vrows))

    # -- confidence: absent / malformed / out-of-range become None, matching the SQL view
    conf_cases = [
        Mention("A Co", "Company", "LOCATED_IN", "Boston", "Location", None, "c0"),
        Mention("B Co", "Company", "LOCATED_IN", "Boston", "Location", 95, "c1"),
        Mention("C Co", "Company", "LOCATED_IN", "Boston", "Location", "high", "c2"),
        Mention("D Co", "Company", "LOCATED_IN", "Boston", "Location", 0.85, "c3"),
    ]
    crows = {r["subject_id"]: r["confidence"]
             for r in build_gold_triplets(conf_cases, resolve_entities(conf_cases))}
    check("missing confidence -> None", crows["company:a_co"], None)
    check("out-of-range confidence -> None", crows["company:b_co"], None)
    check("non-numeric confidence -> None", crows["company:c_co"], None)
    check("valid confidence preserved", crows["company:d_co"], 0.85)

    # -- a mention missing a subject/object/predicate is skipped, not crashed on
    junk = [
        Mention(None, "Company", "LOCATED_IN", "Boston", "Location", 0.9, "c0"),
        Mention("A Co", "Company", None, "Boston", "Location", 0.9, "c1"),
        Mention("A Co", "Company", "LOCATED_IN", "", "Location", 0.9, "c2"),
        Mention("A Co", "Company", "LOCATED_IN", "Boston", "Location", 0.9, "c3"),
    ]
    jrows = build_gold_triplets(junk, resolve_entities(junk))
    check("malformed mentions are skipped, valid one survives",
          [r["predicate"] for r in jrows], ["LOCATED_IN"])

    # -- symmetric predicates stored ONCE with sorted endpoints, matching graph_build.py
    sym = [
        Mention("Skim Milk", "Product", "SUBSTITUTE_FOR", "Whole Milk", "Product", 0.9, "c0"),
        Mention("Whole Milk", "Product", "SUBSTITUTE_FOR", "Skim Milk", "Product", 0.9, "c1"),
    ]
    sym_rows = build_gold_triplets(sym, resolve_entities(sym))
    sub = [r for r in sym_rows if r["predicate"] == "SUBSTITUTE_FOR"]
    check("SUBSTITUTE_FOR is emitted once, not both directions", len(sub), 1)
    check("symmetric endpoints are sorted", sub[0]["subject_id"] <= sub[0]["object_id"], True)
    check("SUBSTITUTE_FOR is declared symmetric",
          "SUBSTITUTE_FOR" in SYMMETRIC_PREDICATES, True)
    # SAME_AS is directional by design (alias -> canonical) and must NOT be symmetric
    check("SAME_AS is NOT symmetric (stays directional)",
          "SAME_AS" not in SYMMETRIC_PREDICATES, True)

    # -- relation vocabulary, normalized before the membership test
    check("normalize_predicate uppercase-snakes a raw phrase",
          normalize_predicate("is based in"), "IS_BASED_IN")
    formatted = [
        Mention("Acme", "Company", "located_in", "Boston", "Location", 0.9, "c0"),
        Mention("Milk", "Product", "SUPPLIED BY", "Acme", "Company", 0.9, "c1"),
        Mention("Acme", "Company", "IS_PRIMARY_DAIRY_SUPPLIER_IN", "Boston", "Location", 0.9, "c2"),
    ]
    fres = resolve_entities(formatted)
    dropped = build_gold_triplets(formatted, fres,
                                  allowed_predicates=DEFAULT_PREDICATES, on_unknown="drop")
    preds_kept = sorted(r["predicate"] for r in dropped if r["predicate"] != "SAME_AS")
    # lowercase and space-separated in-vocabulary predicates must SURVIVE normalization
    check("lowercase/spaced in-vocabulary predicates survive the allowlist",
          preds_kept, ["LOCATED_IN", "SUPPLIED_BY"])
    kept_all = build_gold_triplets(formatted, fres)
    check("default keeps out-of-vocabulary predicates",
          "IS_PRIMARY_DAIRY_SUPPLIER_IN" in {r["predicate"] for r in kept_all}, True)
    check("the example's relation vocabulary matches sql/schema.sql",
          DEFAULT_PREDICATES,
          frozenset({"SUPPLIED_BY", "LOCATED_IN", "BELONGS_TO", "SUBSTITUTE_FOR", "SURGES_IN"}))
    # an invalid on_unknown must fail loudly rather than silently disable the backstop
    try:
        build_gold_triplets(formatted, fres, allowed_predicates=DEFAULT_PREDICATES,
                            on_unknown="DROP")
        check("invalid on_unknown raises", False, True)
    except ValueError:
        check("invalid on_unknown raises ValueError", True, True)

    # -- end to end through index_document with a deterministic stub extractor
    def stub_extractor(ps):
        out = []
        for p in ps:
            if "supplies" in p.text.lower():
                out.append(Mention("Acme Foods", "Company", "SUPPLIED_BY", "Skim Milk 2L",
                                   "Product", 0.9, p.chunk_id))
            if "boston" in p.text.lower():
                out.append(Mention("ACME Foods Inc.", "Company", "LOCATED_IN", "Boston",
                                   "Location", 0.85, p.chunk_id))
        return out

    e2e = index_document("doc:1", doc, stub_extractor)
    check("index_document emits rows", len(e2e) > 0, True)
    check("end-to-end rows conform to the contract",
          all(set(r) == CONTRACT for r in e2e), True)
    preds = {r["predicate"] for r in e2e}
    check("end-to-end captured the SUPPLIED_BY relation", "SUPPLIED_BY" in preds, True)
    check("end-to-end merged the company aliases (SAME_AS emitted)", "SAME_AS" in preds, True)
    check("duplicate relations across chunks are de-duplicated",
          len([r for r in e2e if r["predicate"] == "LOCATED_IN"]), 1)
    check("canonical_id is a stable, type-prefixed slug",
          canonical_id("ACME Foods Inc.", "Company"), "company:acme_foods_inc")

    # ----------------------------------------------------------------- authority
    # A certified node should win the tie against an equally-relevant inferred one, WITHOUT
    # letting a barely-relevant certified node outrank a much more relevant inferred one.
    # That is why authority is a multiplier and not a sort key, and these checks pin the
    # difference — a hard (authority, score) sort would pass the first check and fail the last.
    print("\nTEST 8 — authority ranking (certified core over inferred graph)")

    baseline = agent_retrieve(q, bnodes, bedges, a_emb, seed_k=6, max_hops=2)
    baseline_ids = [r["node_id"] for r in baseline]

    # every node here is unmarked, so every row must read as inferred and its authority
    # score must equal its raw score — the no-source_method path that would go NULL in SQL.
    check("unmarked nodes report source_class=inferred",
          {r["source_class"] for r in baseline}, {"inferred"})
    check("unmarked nodes: authority_score == score (no NULL/None leak)",
          all(r["authority_score"] == r["score"] for r in baseline), True)

    # certify the runner-up, then prove the default is an exact no-op even with a certified
    # row present. This is the backward-compatibility guarantee, not an assumption.
    runner_up = baseline[1]["node_id"]
    certified = {nid: (dict(n, props={"source_method": "uc_certified"}) if nid == runner_up
                       else n)
                 for nid, n in bnodes.items()}
    default_boost = agent_retrieve(q, certified, bedges, a_emb, seed_k=6, max_hops=2)
    check("authority_boost=1.0 is an exact no-op with certified rows present",
          [r["node_id"] for r in default_boost], baseline_ids)
    check("certified row is labelled",
          next(r["source_class"] for r in default_boost if r["node_id"] == runner_up),
          "uc_certified")

    # a boost large enough to close the observed gap must actually promote it, or the
    # feature does nothing.
    # Guard the arithmetic: `score` is already rounded, so a fixture whose runner-up rounds to
    # 0.0 would raise ZeroDivisionError, and a negative runner-up would invert the ratio and make
    # `strong` below silently assert the opposite of what it claims.
    check("authority gap fixture is positive (guards the ratio below)",
          baseline[0]["score"] > 0 and baseline[1]["score"] > 0, True)
    gap = baseline[0]["score"] / baseline[1]["score"]
    strong = agent_retrieve(q, certified, bedges, a_emb, seed_k=6, max_hops=2,
                            authority_boost=gap * 1.10)
    check("a sufficient boost promotes the certified node to rank 1",
          strong[0]["node_id"], runner_up)

    # ...but relevance must still dominate: certifying the WEAKEST node with a modest boost
    # must not hand it rank 1. This is the assertion a sort-key implementation fails.
    weakest = baseline[-1]["node_id"]
    check("weakest-node fixture is positive (guards the ratio below)",
          baseline[-1]["score"] > 0, True)
    weak_gap = baseline[0]["score"] / baseline[-1]["score"]
    certified_weak = {nid: (dict(n, props={"source_method": "uc_certified"})
                            if nid == weakest else n)
                      for nid, n in bnodes.items()}
    modest = agent_retrieve(q, certified_weak, bedges, a_emb, seed_k=6, max_hops=2,
                            authority_boost=1.2)
    check("a modest boost does NOT let the least relevant certified node reach rank 1",
          modest[0]["node_id"] == weakest, False)
    check("the relevance gap it would have to close is real (>1.2x)", weak_gap > 1.2, True)

    # ---- the same rules, executed as SQL. TEST 8 above only exercises graph_retrieve.py;
    # without this the SQL half of the ranking has no coverage at all and can drift silently
    # (it already did once: the SQL gained the positive-score guard a commit before the
    # Python twin did, and every test stayed green).
    acon = duckdb.connect()
    acon.execute("CREATE TABLE anodes(node_id TEXT, props JSON)")
    acon.executemany("INSERT INTO anodes VALUES (?,?)", [
        ("cert:hi",   '{"source_method":"uc_certified"}'),   # certified, strong
        ("inf:hi",    '{"source_method":"inferred"}'),        # inferred, strongest
        ("other:mid", '{"source_method":"structured"}'),      # a real gold_triplets value
        ("bare:mid",  '{}'),                                  # no source_method at all
        ("cert:neg",  '{"source_method":"uc_certified"}'),    # certified, NEGATIVE score
        ("inf:neg",   '{"source_method":"inferred"}'),        # inferred, less negative
    ])
    acon.execute("CREATE TABLE ascored(node_id TEXT, graph_score DOUBLE)")
    acon.executemany("INSERT INTO ascored VALUES (?,?)", [
        ("cert:hi", 0.50), ("inf:hi", 0.60), ("other:mid", 0.30),
        ("bare:mid", 0.30), ("cert:neg", -0.10), ("inf:neg", -0.15),
    ])

    def authority_sql(boost):
        """A TRANSCRIPTION of the assemble stage of sql/graphrag_retrieval.sql, DuckDB dialect.

        NOT the file itself — nothing ties the two together, so any change to the real query's
        weight, clamp, ordering or source_class handling must be mirrored here by hand. This has
        already drifted twice: the positive-score guard and then the GREATEST clamp each landed
        in the .sql file one step before this copy, and the suite stayed green in between.
        """
        return acon.execute(f"""
            SELECT node_id, source_class, ROUND(rank_key, 4) AS authority_score
            FROM (
                SELECT n.node_id,
                       COALESCE(json_extract_string(n.props, '$.source_method'), 'inferred')
                           AS source_class,
                       s.graph_score * COALESCE(
                           CASE WHEN json_extract_string(n.props, '$.source_method')
                                     = 'uc_certified' AND s.graph_score > 0
                                THEN GREATEST({boost}, 1.0) ELSE 1.0 END, 1.0) AS rank_key
                FROM ascored s JOIN anodes n ON n.node_id = s.node_id
            ) r
            ORDER BY rank_key DESC, node_id
        """).fetchall()

    base_sql = authority_sql(1.0)
    check("SQL: boost=1.0 orders by raw score (no-op)",
          [r[0] for r in base_sql],
          ["inf:hi", "cert:hi", "bare:mid", "other:mid", "cert:neg", "inf:neg"])
    check("SQL: a node with no source_method reads as inferred, weight 1.0",
          next((r[1], r[2]) for r in base_sql if r[0] == "bare:mid"), ("inferred", 0.3))
    check("SQL: a foreign source_method passes through verbatim, not normalised",
          next(r[1] for r in base_sql if r[0] == "other:mid"), "structured")

    boosted = authority_sql(2.0)
    check("SQL: boost promotes the certified node past the stronger inferred one",
          [r[0] for r in boosted][:2], ["cert:hi", "inf:hi"])
    # the guard that matters: a boost must never push a certified node DOWN
    neg_before = [r[0] for r in base_sql].index("cert:neg")
    neg_after = [r[0] for r in boosted].index("cert:neg")
    check("SQL: boosting never demotes a negative-score certified node",
          neg_after <= neg_before, True)
    check("SQL: the negative certified node keeps its unboosted score",
          next(r[2] for r in boosted if r[0] == "cert:neg"), -0.1)
    # ---- edge cases the remediation above claimed but no test pinned. Each of these was
    # verified to FAIL when its guard is removed; without them the guards are decoration.

    # (i) NEGATIVE score, Python half. TEST 8's fixture is all-positive by construction (the
    # "fixture is positive" checks guarantee it), so deleting `and score > 0` from
    # graph_retrieve.py left the whole suite green. Explicit vectors are used because
    # bag-of-words embeddings are non-negative and can never produce a negative cosine.
    neg_nodes = {
        "cert:neg": {"node_type": "N", "name": "certified, anti-correlated",
                     "props": {"source_method": "uc_certified"}},
        "inf:neg":  {"node_type": "N", "name": "inferred, less anti-correlated", "props": {}},
    }
    neg_emb = {"cert:neg": [-1.0, 0.0], "inf:neg": [-0.9, -0.436]}   # cosines ~ -1.0 and ~ -0.9
    nq = [1.0, 0.0]
    neg_base = agent_retrieve(nq, neg_nodes, [], neg_emb, seed_k=2, max_hops=0,
                              seed_floor=-1.0)
    neg_boost = agent_retrieve(nq, neg_nodes, [], neg_emb, seed_k=2, max_hops=0,
                               seed_floor=-1.0, authority_boost=2.0)
    check("py: negative-score fixture really is negative",
          all(r["score"] < 0 for r in neg_base), True)
    check("py: boosting never demotes a negative-score certified node",
          [r["node_id"] for r in neg_boost], [r["node_id"] for r in neg_base])
    check("py: the negative certified node keeps its unboosted score",
          next(r["score"] for r in neg_boost if r["node_id"] == "cert:neg"),
          next(r["score"] for r in neg_base if r["node_id"] == "cert:neg"))

    # (ii) ORDERING must use the UNROUNDED key. Every other fixture score is exact at <= 4 dp,
    # so ordering on the rounded column is indistinguishable from ordering on the raw product
    # and a regression to the rounded column passes silently. These two differ at the 5th dp
    # and round to the SAME 4 dp value, so only unrounded ordering keeps them apart.
    tie_nodes = {"a:hi": {"node_type": "N", "name": "a", "props": {}},
                 "b:lo": {"node_type": "N", "name": "b", "props": {}}}
    tie_emb = {"a:hi": [1.0, 0.0], "b:lo": [0.99999, 0.00447]}
    tied = agent_retrieve([1.0, 0.0], tie_nodes, [], tie_emb, seed_k=2, max_hops=0,
                          seed_floor=-1.0)
    check("py: the tie fixture rounds to one value at 4 dp",
          len({r["score"] for r in tied}), 1)
    check("py: but the raw scores differ, so ordering is by the unrounded key",
          [r["node_id"] for r in tied], ["a:hi", "b:lo"])

    # (iii) a None boost must coalesce to 1.0 like the SQL, not raise.
    none_boost = agent_retrieve(q, certified, bedges, a_emb, seed_k=6, max_hops=2,
                                authority_boost=None)
    check("py: authority_boost=None behaves as 1.0 (matches SQL COALESCE)",
          [r["node_id"] for r in none_boost], baseline_ids)

    # (iv) a boost BELOW 1.0 must not demote either — the clamp, not just the sign guard.
    for bad in (0.5, 0.0, -2.0):
        clamped = agent_retrieve(q, certified, bedges, a_emb, seed_k=6, max_hops=2,
                                 authority_boost=bad)
        check(f"py: boost={bad} is clamped, certified node not demoted",
              [r["node_id"] for r in clamped], baseline_ids)
    lo = authority_sql(0.5)
    check("SQL: boost=0.5 is clamped, ordering unchanged",
          [r[0] for r in lo], [r[0] for r in base_sql])
    neg_sql = authority_sql(-2.0)
    check("SQL: boost=-2.0 is clamped, ordering unchanged",
          [r[0] for r in neg_sql], [r[0] for r in base_sql])

    acon.close()

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} FAILED -> {FAILURES}")
        return 1
    print("RESULT: ALL PASSED — GraphRAG retrieval logic verified offline.")
    print("(pgvector HNSW operator + real embeddings validated only on live Lakebase.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
