# Databricks notebook source
# MAGIC %md
# MAGIC # GraphRAG on Lakebase — build the graph, then query it
# MAGIC
# MAGIC Knowledge-graph-augmented RAG served entirely from **Lakebase** (managed Postgres):
# MAGIC the graph is `nodes`/`edges` tables, traversal is a **recursive CTE**, and the semantic
# MAGIC seed is **pgvector** HNSW. A generic Databricks **model-serving** endpoint supplies
# MAGIC embeddings and answer synthesis (swap in whichever provider your workspace has
# MAGIC registered — this example is cloud-agnostic).
# MAGIC
# MAGIC **Honest capability note:** Lakebase is managed Postgres — there is no native
# MAGIC property-graph engine (no Apache AGE / Cypher). GraphRAG here is built on relational +
# MAGIC vector primitives: recursive-CTE traversal + `ltree` + pgvector. Great for bounded
# MAGIC k-hop operational retrieval; not a billion-edge graph-analytics engine.
# MAGIC
# MAGIC This notebook is deployed via the bundle (`databricks.yml`). Parameters come from the
# MAGIC bundle variables (`catalog`, `schema`, `chat_endpoint`, `embedding_endpoint`).

# COMMAND ----------

dbutils.widgets.text("catalog", "main")          # noqa: F821
dbutils.widgets.text("schema", "graphrag")        # noqa: F821
dbutils.widgets.text("embedding_endpoint", "databricks-gte-large-en")  # noqa: F821
dbutils.widgets.text("chat_endpoint", "databricks-claude-sonnet-4-5")   # noqa: F821
dbutils.widgets.text("lakebase_database", "")     # noqa: F821  # projects/<id>/branches/.../databases/<db>

CATALOG = dbutils.widgets.get("catalog")          # noqa: F821
SCHEMA = dbutils.widgets.get("schema")            # noqa: F821
EMBED_ENDPOINT = dbutils.widgets.get("embedding_endpoint")  # noqa: F821
CHAT_ENDPOINT = dbutils.widgets.get("chat_endpoint")        # noqa: F821
LAKEBASE_DATABASE = dbutils.widgets.get("lakebase_database")  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. A tiny example knowledge graph
# MAGIC A retail supply-chain slice: products, suppliers, regions, categories, and a substitute
# MAGIC relationship. `assemble_graph` (see `graph_build.py`) turns simple dims into nodes/edges;
# MAGIC it guards the edges->nodes foreign key so a bad reference never creates an orphan edge.

# COMMAND ----------

from graph_build import assemble_graph  # noqa: E402

products = [
    {"product_id": 122, "name": "Skim Milk 2L", "category": "Dairy"},
    {"product_id": 130, "name": "Whole Milk 2L", "category": "Dairy"},
]
suppliers = [
    {"supplier_id": "S2", "name": "Supplier S2", "city": "Boston", "region": "Northeast"},
    {"supplier_id": "S3", "name": "Supplier S3", "city": "Boston", "region": "Northeast"},
]
supplies = [
    {"product_id": 122, "supplier_id": "S2"},
    {"product_id": 122, "supplier_id": "S3"},
    {"product_id": 130, "supplier_id": "S2"},
]
surges = [{"product_id": 122, "region": "Northeast"}]
enrichment = {
    122: {"description": "Skim Milk 2L — a dairy product.", "substitute_ids": [130]},
    130: {"description": "Whole Milk 2L — a dairy product.", "substitute_ids": [122]},
}
nodes, edges = assemble_graph(products, suppliers, supplies, surges, enrichment)
print(f"nodes={len(nodes)} edges={len(edges)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Embeddings via a Databricks model-serving endpoint
# MAGIC Generic — any embeddings endpoint registered in your workspace.

# COMMAND ----------

from databricks.sdk import WorkspaceClient  # noqa: E402

w = WorkspaceClient()


def embed(texts):
    # resp.data items are typed objects (EmbeddingsV1ResponseEmbeddingElement) — use
    # attribute access, not dict subscript. Verified against databricks-sdk on a live run.
    resp = w.serving_endpoints.query(name=EMBED_ENDPOINT, input=texts)
    return [d.embedding for d in resp.data]


node_ids = list(nodes.keys())
texts = [f"{nodes[n]['node_type']}: {nodes[n]['name']}" for n in node_ids]
embeddings = dict(zip(node_ids, embed(texts), strict=True))
print(f"embedded {len(embeddings)} nodes")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Write to Lakebase + create indexes — ✍️ COMPLETE THIS
# MAGIC The connection is workspace-specific, so this cell is **scaffolding you complete** with
# MAGIC your Lakebase project's details (`LAKEBASE_DATABASE` is passed in from the bundle var).
# MAGIC Use a Databricks-issued Postgres credential — a short-lived OAuth token, no long-lived
# MAGIC passwords. The pattern (see the databricks-lakebase docs) is:

# COMMAND ----------

# ✍️ COMPLETE: connect with a Databricks-issued credential and load the graph.
#
#   from databricks.sdk import WorkspaceClient
#   import psycopg, uuid
#   inst_name = LAKEBASE_DATABASE.split("/")[0:2]  # derive your instance/project as needed
#   inst = w.database.get_database_instance(name="<your-instance>")
#   cred = w.database.generate_database_credential(
#       request_id=str(uuid.uuid4()), instance_names=["<your-instance>"])
#   with psycopg.connect(host=inst.read_write_dns, dbname="databricks_postgres",
#                        user=w.current_user.me().user_name, password=cred.token,
#                        sslmode="require") as conn:
#       conn.execute(open("../sql/schema.sql").read())          # extensions + tables + indexes
#       # INSERT nodes / edges / node_embeddings (embeddings as pgvector literals)
#
# Until completed, the steps below show what the query WOULD run.
_target = LAKEBASE_DATABASE or "(set --var lakebase_database)"
print(f"[scaffold] target Lakebase database = {_target}")
print("Complete the cell above to apply sql/schema.sql and INSERT the graph.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Query — hybrid GraphRAG retrieval — ✍️ COMPLETE THIS
# MAGIC Once the graph is loaded (step 3), run `sql/graphrag_retrieval.sql` against the same
# MAGIC Lakebase connection: pgvector cosine seed -> recursive-CTE k-hop expansion. Bind
# MAGIC `:query_embedding` (embed the user question), `:seed_k`, `:max_hops`, `:seed_floor`,
# MAGIC (`:seed_floor` is the min cosine a seed must clear; `-1.0` = keep all, same as before —
# MAGIC raise toward
# MAGIC a floor calibrated to your model's similarity range — ~`0.55` for
# MAGIC `databricks-gte-large-en` on this graph). `:authority_boost` of `1.0` keeps the
# MAGIC pre-authority ranking; all five binds are required. To validate the logic
# MAGIC *without* Lakebase, run `smoketest/graphrag_logic_smoketest.py`.

# COMMAND ----------

question = "Which suppliers can cover skim milk demand surging in the Northeast?"
q_emb = embed([question])[0]
print(f"embedded question (dim={len(q_emb)}). Bind it as :query_embedding in "
      "sql/graphrag_retrieval.sql with :seed_k, :max_hops, :seed_floor (-1.0 = keep all) and "
      ":authority_boost (1.0 = rank as before), "
      f"execute against Lakebase, then pass the ranked context to: {CHAT_ENDPOINT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4b. See authority ranking work — ✍️ OPTIONAL
# MAGIC `:authority_boost` does nothing until some node is marked as curated. Step 5 builds those
# MAGIC from Unity Catalog metric views; this step marks one by hand instead, so you can see the
# MAGIC ranking move without any metric view at all. Anything that can write the `props` key
# MAGIC participates, which keeps the ranking independent of where curation comes from.
# MAGIC
# MAGIC To watch it work end to end without wiring a producer, mark one node by hand and re-run
# MAGIC the query twice. Pick a node the query already returns but *not* the top hit, so the
# MAGIC promotion is visible:
# MAGIC
# MAGIC ```sql
# MAGIC -- mark one node as curated (any producer would write this same key)
# MAGIC UPDATE graph.nodes
# MAGIC    SET props = props || '{"source_method": "uc_certified"}'::jsonb
# MAGIC  WHERE node_id = 'supplier:S3';
# MAGIC ```
# MAGIC
# MAGIC Now run the retrieval twice, changing only the one bind:
# MAGIC
# MAGIC - `:authority_boost = 1.0` — ranking is unchanged from before, and `supplier:S3` now
# MAGIC   reports `source_class = 'uc_certified'` so an answer can cite it.
# MAGIC - `:authority_boost = 2.0` — `supplier:S3` moves up, because its relevance score is
# MAGIC   multiplied while everything else keeps weight `1.0`.
# MAGIC
# MAGIC What to look for, because it is the point of the design: a **modest** boost does not hand
# MAGIC rank 1 to a weakly-related certified node. Mark `category:dairy` instead — it is in this
# MAGIC graph but further from the question — and at `1.2` the on-topic nodes stay above it, because
# MAGIC relevance still dominates and authority only breaks near-ties. Be precise about the limit
# MAGIC though: the boost is an unbounded multiplier with only a lower clamp, so a *large* enough
# MAGIC value will promote any positively-scored certified node to rank 1. That is a calibration
# MAGIC choice, not a guarantee the query makes. Undo with:
# MAGIC
# MAGIC ```sql
# MAGIC UPDATE graph.nodes SET props = props - 'source_method' WHERE node_id = 'supplier:S3';
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Certified core from Unity Catalog metric views — ✍️ OPTIONAL
# MAGIC Step 4b marked a node certified by hand. This does it from real curated semantics:
# MAGIC `graph_certified.py` reads this workspace's Unity Catalog metric views and emits nodes
# MAGIC and edges already tagged `source_method='uc_certified'`.
# MAGIC
# MAGIC **Two things this does NOT do, and both matter.** It writes nothing — you load the rows
# MAGIC yourself. And loading them is not sufficient for the boost to fire: retrieval seeds
# MAGIC exclusively from `graph.node_embeddings`, so a certified node with no embedding can never
# MAGIC be a seed; and `certified_rows()` emits only view→field edges, so nothing joins the
# MAGIC certified subgraph to the business graph and expansion cannot reach it from a business
# MAGIC seed either. Until you embed these nodes (and, if you want cross-over, add your own edges
# MAGIC linking a measure to the tables or entities it describes), the boost fires only when the
# MAGIC question itself semantically matches a metric-view field. That is a real design
# MAGIC constraint, not a wiring-free win.
# MAGIC
# MAGIC Requires at least one metric view you can read. Scope it to one catalog — unscoped
# MAGIC discovery reads every catalog you can see, which on a large workspace is thousands of rows.

# COMMAND ----------

# `SqlReader` is a Protocol, so anything that returns rows satisfies it. spark.sql is the
# obvious choice inside a notebook; the module itself has no Spark dependency, which is what
# keeps it unit-testable with no warehouse.
class SparkReader:
    def rows(self, sql):
        return [list(r) for r in spark.sql(sql).collect()]  # noqa: F821


# ✍️ set this to a catalog that actually contains a metric view
CERTIFIED_CATALOG = None  # e.g. "my_catalog"

if CERTIFIED_CATALOG:
    from graph_certified import certified_rows, discover_metric_views, read_metric_view

    reader = SparkReader()
    views = discover_metric_views(reader, catalog=CERTIFIED_CATALOG)
    print(f"found {len(views)} user metric view(s) in {CERTIFIED_CATALOG}")

    certified_nodes, certified_edges = [], []
    skipped = []
    for cat, sch, name in views:
        # Discovery returns whatever information_schema holds, but read_metric_view validates
        # identifiers and parse_describe refuses shapes it cannot trust (no detail block, repeated
        # field names). A non-ASCII view name like `ventas_año` is legitimate in Unity Catalog and
        # is rejected here, so one awkward view must not discard every other view's work.
        try:
            view = read_metric_view(reader, cat, sch, name)
            rows = certified_rows(view)
        except Exception as e:
            # Exception, not ValueError: ValueError covers only this module's own refusals, while
            # the likeliest real failure comes from spark.sql inside read_metric_view — an
            # AnalysisException for PERMISSION_DENIED (BROWSE but not SELECT on one view, which
            # information_schema still lists) or TABLE_OR_VIEW_NOT_FOUND (dropped between
            # discovery and DESCRIBE). Neither is a ValueError, so the cell used to abort mid-loop
            # and discard every row already built. The type is recorded so a permission problem
            # reads differently from a shape this module rejected on purpose.
            skipped.append((f"{cat}.{sch}.{name}", f"{type(e).__name__}: {e}"))
            continue
        certified_nodes.extend(rows.nodes)
        certified_edges.extend(rows.edges)
        measures = sum(1 for f in view.fields if f.is_measure)
        enrich = ("" if view.enrichment_available
                  else "  [!] no display_name/synonyms: reachable only by raw column name")
        print(f"  {view.fqn}: {measures} measure(s), "
              f"{len(view.fields) - measures} dimension(s), "
              f"owner={view.owner or 'unknown'}{enrich}")

    for fqn, why in skipped:
        print(f"  SKIPPED {fqn}: {why}")

    if views and not certified_nodes:
        # Every view skipped is not "all your views are awkward" — it is far likelier that nothing
        # ran: no Spark session, a workspace-wide DESCRIBE format change, or a permission problem
        # on the whole catalog. Without this the cell prints "0 certified node(s) ready to load"
        # and the load-them-yourself paragraph, which reads as success.
        raise RuntimeError(
            f"all {len(views)} discovered view(s) were skipped — see the SKIPPED lines above; "
            "this is a systemic failure, not a per-view one"
        )

    print(f"\n{len(certified_nodes)} certified node(s) and "
          f"{len(certified_edges)} edge(s) ready to load.")
    print("There is no loader in this example to reuse — step 3 is commented-out scaffolding — "
          "so INSERT these into graph.nodes / graph.edges yourself (props as JSONB), then embed "
          "each node with "
          "graph_certified.embedding_text(node) — NOT its bare name. A question says "
          "'how much did we make', not 'total_revenue', so the display name, description and "
          "the curator's synonyms are what make the certified core reachable at all. Authority "
          "ranking only re-orders nodes retrieval already found.")
else:
    print("CERTIFIED_CATALOG is unset — skipping. Set it to a catalog containing a metric view.")

# COMMAND ----------

# MAGIC %md
# MAGIC The retrieved context is the graph-connected neighborhood of the semantic seed —
# MAGIC suppliers, substitutes, and region that a flat vector search would miss. Hand it to the
# MAGIC chat endpoint to synthesize the final answer.
