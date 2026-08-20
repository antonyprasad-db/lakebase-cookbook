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
# MAGIC `:authority_boost` (1.0 keeps the pre-authority ranking; all five binds are required)
# MAGIC (min cosine a seed must clear; `-1.0` = keep all, same as before — raise toward
# MAGIC a floor calibrated to your model's similarity range — ~`0.55` for
# MAGIC `databricks-gte-large-en` on this graph). To validate the logic *without* Lakebase,
# MAGIC run `smoketest/graphrag_logic_smoketest.py`.

# COMMAND ----------

question = "Which suppliers can cover skim milk demand surging in the Northeast?"
q_emb = embed([question])[0]
print(f"embedded question (dim={len(q_emb)}). Bind it as :query_embedding in "
      "sql/graphrag_retrieval.sql with :seed_k, :max_hops, :seed_floor (-1.0 = keep all) and "
      ":authority_boost (1.0 = rank as before), "
      f"execute against Lakebase, then pass the ranked context to: {CHAT_ENDPOINT}")

# COMMAND ----------

# MAGIC %md
# MAGIC The retrieved context is the graph-connected neighborhood of the semantic seed —
# MAGIC suppliers, substitutes, and region that a flat vector search would miss. Hand it to the
# MAGIC chat endpoint to synthesize the final answer.
