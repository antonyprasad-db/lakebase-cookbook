# GraphRAG on Lakebase

Knowledge-graph-augmented RAG served entirely from **Lakebase** (managed Postgres). Classic RAG
retrieves text chunks by vector similarity and misses *relationships* ("which suppliers cover the
products whose demand is surging, and what substitutes share those suppliers?"). GraphRAG adds a
knowledge graph so retrieval can **traverse relationships**, not just match text.

## How it works

- **Graph store** — `nodes` + `edges` relational tables (typed, JSONB props).
- **Traversal** — a **recursive CTE** walks edges out from a seed, bounded by `max_hops`.
- **Semantic seed** — **pgvector** HNSW (cosine) finds the entry nodes for a question.
- **Hybrid retrieval** — seed with vectors, expand along the graph, rank by a blended
  `seed_similarity × 0.5^hop` score. See [`sql/graphrag_retrieval.sql`](sql/graphrag_retrieval.sql).

A generic Databricks **model-serving** endpoint supplies embeddings and answer synthesis — swap in
whichever provider your workspace has registered. Nothing here is cloud-specific.

## Honest capability note

Lakebase is managed Postgres — there is **no native property-graph engine** (no Apache AGE, no
Cypher). This example builds GraphRAG on relational + vector primitives, all verified as supported
Lakebase extensions:

| Need | How | Extension |
| --- | --- | --- |
| Graph store | `nodes` / `edges` tables | core |
| Multi-hop traversal | recursive CTE (`WITH RECURSIVE`) | core |
| Hierarchies | `ltree` paths | `ltree` |
| Semantic seed | pgvector HNSW, cosine (`<=>`) | `vector` |
| Fuzzy entity link | trigram similarity | `pg_trgm` |

Position it as *"graph traversal via recursive CTEs + semantic retrieval via pgvector"* — great for
bounded k-hop operational retrieval; **not** a billion-edge graph-analytics engine.

## Building the graph from documents (upstream indexing)

`graph_build.py` starts from *structured* dims. If your source is documents, `graph_upstream.py`
covers the phase before that:

```
parse (ai_parse_document) -> chunk -> extract (ai_query, constrained schema)
  -> entity resolution (trigram + union-find) -> gold_triplets
```

Chunking, entity resolution and triplet assembly are pure Python and covered by the offline smoke
test. The two steps that need a workspace — parse and extract — are in
[`sql/upstream_ai_functions.sql`](sql/upstream_ai_functions.sql).

Two things matter more than the rest, both learned from running this live:

- **Entity resolution is the load-bearing step.** The same entity appears written many ways
  ("Acme", "Acme Foods", "ACME Foods Inc."), and each spelling would otherwise become its own
  node — fragmenting the graph exactly where multi-hop retrieval needs it joined. Note that
  Jaccard trigram similarity alone is *not* enough: "Acme Foods" vs "Acme Foods Incorporated"
  scores only `0.42`, so legal suffixes and abbreviations slip past it. `resolve_entities()` also
  applies a containment measure (the shape of `pg_trgm`'s `word_similarity()`), which scores that
  pair `0.92`. Merges are written back as `SAME_AS` edges so the decision is auditable in the
  graph rather than lost in the pipeline.
- **Pin the relation vocabulary.** Constraining entity types is not enough. Left free, the model
  invents a predicate per sentence — a live run over two short documents produced
  `SUPPLIES_TO_RETAILERS_ACROSS`, `OPERATES_DISTRIBUTION_CENTER_IN`, `IS_LOCATED_IN` and
  `BELONGS_TO_CATEGORY`, the last two near-misses for this schema's `LOCATED_IN` and
  `BELONGS_TO`. Every spelling becomes its own edge label and the typed-path logic in the
  retrieval SQL degrades toward an untyped walk. Constrain the predicate list in the prompt, and
  pass `allowed_predicates=DEFAULT_PREDICATES, on_unknown="drop"` as a backstop.

Two AI-Functions details the SQL had wrong until it was run against a live workspace, both now
fixed and commented in place:

- **`ai_parse_document` has no `document.text` field.** It returns schema v2.0, whose text lives
  in `document.elements[].content` with a `type` per element. Reading `parsed:document.text`
  yields NULL and the pipeline silently produces nothing — concatenate the text elements instead.
- **`ai_query`'s DDL-string `responseFormat` rejects a top-level array.** Use the JSON-schema form
  to constrain the model and `from_json` to type the result. The SQL carries the exact error you
  get otherwise.

## Operational guardrails

Defaults that keep retrieval fast, precise, and cost-predictable on real Lakebase:

- **Bound `max_hops` to 2 on dense graphs.** Recursive-CTE cost is *density-driven*: at 4 hops on a
  dense graph the walk approaches a full-graph scan, while 2 hops stays bounded. Raise hops only on
  sparse graphs. (Indicative scale figures — tens of ms at 2 hops on ~1M nodes / 10M edges, seconds
  at 4 hops — come from a separate benchmark of this pattern, not from this repo's smoke test.)
- **Set a `seed_floor` under distractor-heavy corpora.** Weak semantic seeds drag unrelated subgraphs
  into the ranking and dilute precision; a cosine floor drops them. **Calibrate the floor to
  your embedding model's actual similarity distribution** — it is not a portable constant.
  Measured on live Lakebase with `databricks-gte-large-en`, this example graph's node
  similarities span only `0.38`-`0.71`, so a `0.3` floor changes nothing while `0.555` drops
  every distractor and keeps every on-topic node (the on-topic supplier below the floor still
  arrives via graph expansion). Cosine ranges
  `[-1, 1]`, so `-1.0` keeps all seeds (identical to no floor). **`:seed_floor` is a required
  bind** — Postgres has no server-side default for a named parameter, so existing callers of the
  SQL must pass it even to keep the old behavior (bind `-1.0`); omitting a required bind raises a
  `bind message supplies N parameters, but prepared statement requires N+1` error. The exact
  numbers depend on the driver, since `:query_embedding` appears three times and drivers differ
  on whether repeated names collapse to one bind. The Python twin
  defaults it to `-1.0`, so Python callers need no change. **Watch the empty-seed case:** a floor
  above every node's similarity empties the seed CTE and the query returns *zero* rows — detect
  that in your caller and either retry at `-1.0` or decline to answer, rather than synthesizing
  from no context. See `:seed_floor` in
  [`sql/graphrag_retrieval.sql`](sql/graphrag_retrieval.sql).
- **`:authority_boost` is a required bind, exactly like `:seed_floor`.** Adding it changes the SQL's
  parameter arity, so an existing caller that does not pass it raises
  `bind message supplies 4 parameters, but prepared statement requires 5` — the ranking is
  backward-compatible, the *bind signature* is not. Bind `1.0` to keep the previous behaviour
  exactly. The Python twin defaults it to `1.0`, so Python callers need no change. Values below
  `1.0` are clamped up to `1.0` in both halves, because a fractional or negative multiplier would
  demote the certified node the boost exists to promote. **A non-finite or absent boost means "no
  boost":** NaN, +/-Infinity and NULL all resolve to `1.0` on both halves rather than erroring,
  since a boost derived from a score ratio can reach any of them; a Decimal or numeric string is
  honoured normally. Calibrate it once a certified core actually
  exists — start around `1.2`-`2.0` and check against your own graph, since the useful value depends
  on the score gaps your embedding model produces, not on a portable constant.
- **Keep a minimum Lakebase compute above zero for latency-sensitive serving.** After scale-to-zero,
  the first query pays a cold-start penalty of several× the warm latency while index pages load in
  (measured on a live instance at ~6× on a small graph; scale-dependent).
- **Seed with plain pgvector, not a BM25+vector hybrid.** Lexical fusion adds latency and can *hurt*
  ranking on this workload — the relationship recall comes from graph expansion, not lexical overlap.
- **Treat the HNSW build as a one-time cost that scales with corpus size** (indicatively seconds at
  10⁴ nodes, minutes at 10⁵), independent of query latency. Storing embeddings as `halfvec` (fp16)
  cuts the embedding-plus-HNSW-index storage ~2.5× vs fp32 (measured) at negligible recall cost — a
  cheap lever for large graphs. It is **not** a drop-in edit of `schema.sql`: `vector_cosine_ops`
  rejects `halfvec`, so switching the column type also requires `halfvec_cosine_ops` on the index
  (or an expression index `((embedding::halfvec(1024)) halfvec_cosine_ops)`) *and* a matching
  `::halfvec` cast on the bound query vector, or the `<=>` lookup fails with a missing-operator
  error.
- **Route simple, single-entity questions to the managed path first.** If a question needs no
  multi-hop traversal, ontology-enriched Genie or a plain vector lookup is cheaper; reserve GraphRAG
  for genuinely relational / multi-hop queries.

## Cross-stack interop (`gold_triplets`)

The `nodes` / `edges` tables map to a portable **8-column triplet shape** — `subject_id, subject_type,
predicate, object_id, object_type, confidence, source_method, source_agent` — so the same graph can be
emitted or consumed by other graph stacks (Spark/Python builders, other engines) **without migration**.
Undirected relations (`SUBSTITUTE_FOR`) are stored once with sorted endpoints and made bidirectional
at query time by the retrieval SQL, so the view emits **both directions** for them — a directed
consumer would otherwise miss the reverse. De-duplicate symmetric predicates on ingest.
It's a column contract, not a dependency. See [`sql/gold_triplets_mapping.sql`](sql/gold_triplets_mapping.sql).

## Authority: a certified core over the inferred graph

Retrieval so far ranks by **relevance** only — semantic seed similarity decayed by hop distance. That
cannot tell a triplet inferred from a document apart from a definition somebody curated on purpose,
which is the question that actually gets asked of a governed context layer: *is this the authoritative
definition, and where did it come from?*

A node marked as certified carries it in `props`, the same place edges already carry provenance:

```json
{"source_method": "uc_certified"}
```

Retrieval then multiplies that node's score by `:authority_boost` and returns two extra columns:
`source_class` (`uc_certified` / `inferred`) so an answer can cite what resolved it, and
`authority_score`, the rounded weighted value. Note the ordering uses the UNROUNDED
weighted score, not this column — see the caveats below before re-sorting client-side on it.

**A multiplier, not a sort key.** Ordering by `(authority, score)` would let a barely-relevant
certified node outrank a highly relevant inferred one — worse than having no authority signal at all.
Multiplying keeps relevance meaningful and makes the strength one tunable knob. The smoke test pins
both directions: a sufficient boost *does* promote a certified node to rank 1, and a modest boost
*does not* let the least relevant certified node reach rank 1. A sort-key implementation passes the
first and fails the second. Both halves are covered, including the negative-score and
below-1.0-boost guards. Note the SQL side is exercised as a **hand transcription** of the assemble
stage against its own fixture, not by running `sql/graphrag_retrieval.sql` itself — nothing ties the
two together, so a change to the query must be mirrored in the test by hand. Only a live Lakebase run
exercises the real file.

**`:authority_boost` defaults to 1.0, which leaves every weight at 1.0.** Scores are then unchanged
and distinctly-scored rows keep their relative order. It is **not** a byte-identical no-op, though,
and the difference is not hypothetical — measured live against a 1219-node graph, two pairs in the
top ten were exact score ties (`0.2945` and `0.1473`), and they now order deterministically by
`node_id` where the previous query left tied rows in whatever order Postgres returned. That is a
deliberate improvement over an unstable sort, but it is a change. Three further caveats:

- The ranking orders by the **unrounded** product. Ordering by the rounded `authority_score` column
  would collapse scores that differ beyond 4 dp into ties — `0.175024` and `0.175006` both round to
  `0.1750` — and let the tiebreak reorder them, which is *not* a no-op. The rounded value is output
  only.
- The `node_id` tiebreak orders `TEXT` by **database collation** in Postgres and by Unicode codepoints
  in the Python twin, so the two agree only under `C` collation. It matters only for ids differing in
  case or punctuation.
- The query reads `n.props`, so `graph.nodes` must have that column. It is in
  [`sql/schema.sql`](sql/schema.sql), but a graph built by an older or bespoke script may lack it —
  one such graph turned up during live testing. The query then fails with
  `column n.props does not exist` on *every* retrieval, not just certified rows. This is pre-existing
  (the previous query also selected `n.props`), and the fix is one statement:
  `ALTER TABLE graph.nodes ADD COLUMN IF NOT EXISTS props JSONB NOT NULL DEFAULT '{}'::jsonb`.

**Only positive scores are boosted, and that guard is load-bearing.** `seed_similarity` is
`1 - cosine_distance`, so it spans `[-1, 1]`, and the default `seed_floor = -1.0` admits
negative-similarity seeds by design. Multiplying a negative score makes it *more* negative: a
certified node at `-0.10` with a boost of `2.0` becomes `-0.20` and sorts **below** an inferred node
at `-0.15`, so the boost would demote exactly what it exists to promote, and get worse as you raise
the knob. Boosting only above zero keeps the transform monotone. The smoke test pins this in both
halves.

**`source_class` is passed through verbatim, not normalised to two values.** It is whatever
`n.props ->> 'source_method'` holds, defaulting to `inferred` only when the key is absent, so any
producer's own label survives. In practice today you will see only `uc_certified` or `inferred`,
because nothing in this repo writes `source_method` into **node** props yet — the `structured` and
`llm_enrichment` values in [`sql/gold_triplets_mapping.sql`](sql/gold_triplets_mapping.sql) are
**edge** provenance (`e.props`), a different table. The pass-through is future-proofing rather than a
current collision. Only `uc_certified` is boosted; every other value carries weight `1.0`.
**Branch on `== 'uc_certified'`, never on `!= 'inferred'`.**

**Where certified nodes come from is deliberately not this example's business.** Anything that can
write the `props` key participates — a Unity Catalog semantics exporter, a curation UI, a hand-written
insert. That keeps the ranking seam independent of any one producer's availability, and it is why this
half carries no new prerequisite beyond what the example already needs.

## Certified core: Unity Catalog metric views as authoritative nodes

The authority stage above boosts a node whose props carry `source_method: "uc_certified"`.
[`graph_certified.py`](graph_certified.py) is what produces those nodes, from Unity Catalog
**metric views** — the objects the docs call the core implementation of Unity Catalog semantics.

Each metric view becomes a node, each field becomes a node, and a directional edge joins them:

```
metricview:cat.sch.order_details  --HAS_MEASURE-->    measure:cat.sch.order_details.total_revenue
                                  --HAS_DIMENSION-->  dimension:cat.sch.order_details.ship_date
```

Fields are nodes rather than props on the view for a concrete reason: retrieval seeds on node
embeddings and expands along edges, so a question about "revenue" must be able to match the
*measure* and then reach its view. A measure buried in a JSON blob is unreachable, and authority
ranking only ever re-orders nodes retrieval already found. That is also why `embedding_text()`
embeds the display name, the description and the curator's synonyms rather than the column name —
nobody asks a question using `total_revenue`.

### The read path takes two queries

| Source | Gives | Missing |
| --- | --- | --- |
| `information_schema.tables` (`table_type = 'METRIC_VIEW'`) | discovery | — |
| `information_schema.columns` | names, types, comments | **cannot tell a measure from a dimension** |
| `DESCRIBE EXTENDED <view>` | the measure flag, `display_name`, `synonyms`, `Owner`, `Created Time` | — |

Both `data_type` and `full_data_type` report a measure as plain `bigint`; only DESCRIBE marks it,
by suffixing the type with ` measure`. Since measure-versus-dimension *is* the semantic layer,
DESCRIBE is not optional. It is this module's one brittle coupling, so it is isolated in
`parse_describe()` and pinned against verbatim live output.

`Owner` and `Created Time` are carried onto the view node. Retrieval does not consult them today,
but they are the raw material for a richer authority function than "certified or not".

### `information_schema.semantic_*` is Snowflake's, not Unity Catalog's

`semantic_views`, `semantic_dimensions`, `semantic_metrics` and `semantic_relationships` look like
exactly the structured source this should read. **They are not.** In one workspace survey, all 12
catalogs exposing them were Snowflake connections surfaced through Lakehouse Federation, and the
native catalog holding an actual Databricks metric view exposed none of them. Reading them yields a
builder that works only against federated Snowflake and silently finds nothing in Unity Catalog.

### Scope, and why only metric views are required

Unity Catalog business semantics has four parts: metric views, **domains**, governed **Pages**, and
**certification/deprecation** signals. Only metric views are required here, so the prerequisite
stays "you have a metric view" and the example runs for a reader without an allow-list. The other
three are deliberate future enrichment.

Genie and Lakeview materialise their own metric views under
`__databricks_internal_catalog_lakeview_*`; discovery filters them, because in one sample they
outnumbered user objects on the first page of results.

### Running it

The builder is pure Python behind a `SqlReader` protocol, so it does not care how you reach Unity
Catalog — `spark.sql`, the SQL connector, or the Statement Execution API all satisfy it. Notebook
step 5 reads this workspace's metric views and reports what it would emit.

**It writes nothing.** `certified_rows()` returns nodes and edges and stops there, which is what
keeps the module testable with no database. Loading them is yours to do: note `graph_build.py`
contains no loader either, and the notebook's load step is commented-out scaffolding, so there is
no existing writer to reuse.

**Loading them is necessary but not sufficient for the boost to fire.** Retrieval seeds only from
`graph.node_embeddings`, so a certified node with no embedding can never be a seed; and this module
emits only view→field edges, so nothing joins the certified subgraph to the business graph and
expansion cannot reach it from a business seed. Until you embed these nodes — with
`embedding_text()`, not the bare column name — and optionally add your own edges linking a measure
to what it describes, authority ranking fires only when the question itself matches a metric-view
field. That is a real constraint of this design, worth knowing before you calibrate the boost.

## Layout

```
graphrag/
├── databricks.yml       # Asset Bundle — deploys the build/query notebook as a job
├── pyproject.toml       # uv deps + ruff config
├── sql/
│   ├── schema.sql              # nodes / edges / node_embeddings + HNSW / ltree / pg_trgm
│   ├── graphrag_retrieval.sql  # hybrid: pgvector seed (+ seed_floor) → recursive-CTE expansion → ranked context
│   ├── gold_triplets_mapping.sql  # portable 8-column triplet view over nodes/edges (cross-stack interop)
│   └── upstream_ai_functions.sql  # parse + typed extract (AI Functions) + pg_trgm blocking
├── graph_build.py       # pure assemble_graph(dims, enrichment) → (nodes, edges); guards the FK
├── graph_retrieve.py    # pure in-memory twin of the retrieval SQL (for offline dev/test)
├── graph_upstream.py    # documents → chunk → extract → entity resolution → gold_triplets
├── graph_certified.py     # Unity Catalog metric views -> certified nodes/edges
├── notebooks/
│   └── graphrag_build_and_query.py   # build the graph, embed, write to Lakebase, query
└── smoketest/
    ├── graphrag_logic_smoketest.py   # offline validation (no Lakebase needed)
    └── certified_core_smoketest.py   # DESCRIBE parsing + projection (no warehouse needed)
```

## Run the offline smoke test

Validates the retrieval + build logic with no Lakebase, no model endpoint:

```bash
cd agents/graphrag
uv run --python 3.11 --with duckdb --with numpy smoketest/graphrag_logic_smoketest.py
```

134 assertions: semantic seed, graph expansion surfacing connected context flat RAG misses,
authority ranking (certified-over-inferred, the positive-score and below-1.0 clamps, NaN /
+Infinity / None boosts asserted on BOTH halves, odd bind types — Decimal, an int wider than a
float, a string — reaching neither an exception nor a non-finite score, unrounded ordering,
verbatim `source_class`),
dangling-edge guard, blended-score ranking, depth bound, and the `seed_floor` distractor guard.

The certified-core builder has its own suite, which needs no warehouse — the `SqlReader` protocol
lets a fake replay recorded output:

```bash
uv run --python 3.11 smoketest/certified_core_smoketest.py
```

Its `DESCRIBE` fixture is verbatim live output, because parsing `DESCRIBE` is the module's one
brittle coupling and inventing its shape would prove nothing. It covers measure-vs-dimension
detection, a field named `# Orders` (which must not be mistaken for the metadata boundary),
3-column and 2-column `DESCRIBE` rows, malformed metadata JSON, and rejection of catalog names
that would otherwise be interpolated into SQL.

## Deploy (Asset Bundle)

```bash
cd agents/graphrag
# set your Lakebase database path + endpoints, then:
databricks bundle deploy -t dev \
  --var lakebase_database="projects/<id>/branches/production/databases/<db>"
databricks bundle run graphrag_build -t dev
```

The bundle deploys `notebooks/graphrag_build_and_query.py` as a job. The notebook assembles a
small example supply-chain graph and embeds its nodes via your embedding endpoint out of the box.

> **The two Lakebase I/O cells are scaffolding you complete** (marked ✍️ COMPLETE THIS): the
> Postgres connection is workspace-specific, so the notebook documents the exact
> credential/connection pattern (Databricks-issued OAuth token → apply `sql/schema.sql` → INSERT
> the graph → run `sql/graphrag_retrieval.sql`) rather than hardcoding an instance. The
> **retrieval logic itself is fully validated offline** by the smoke test below, so you can trust
> the query before you wire the connection.

## Prerequisites

- Databricks workspace with **Lakebase (Autoscaling)** enabled
- A model-serving **embeddings** endpoint and a **chat** endpoint registered in the workspace
- `databricks` CLI + `uv`
- `graph.nodes` must have a `props` column — it is in [`sql/schema.sql`](sql/schema.sql), but a
  graph built by an older or bespoke script may lack it; see the note under Operational guardrails
- **For the certified core only:** at least one Unity Catalog **metric view** you can read. Nothing
  else in the example needs it, and the ranking behaves identically without one

---

*Adapted from a Databricks Field Engineering AWS-integration asset (Unity AI Gateway + Amazon
Bedrock variant); this cookbook version is cloud-agnostic.*
