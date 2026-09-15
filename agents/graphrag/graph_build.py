"""Pure graph-assembly logic shared by notebook 02 and the offline smoke test.

Kept dependency-free (no spark, no databricks.sdk) so the build logic can be unit-tested
offline. Notebook 02 imports `assemble_graph`; the smoke test imports it too and asserts the
resulting nodes/edges match the intended supply-chain knowledge graph.
"""

from __future__ import annotations


def region_key(r: str) -> str:
    return "region:" + r.strip().lower().replace(" ", "_")


def category_key(c: str) -> str:
    return "category:" + c.strip().lower().replace(" ", "_")


def assemble_graph(products, suppliers, supplies, surges, enrichment):
    """Turn structured supply-chain dims + LLM enrichment into (nodes, edges).

    Returns:
      nodes: dict node_id -> {"node_type","name","props","path"}
      edges: set of (src_id, dst_id, rel)

    node_id convention: 'type:key' (product:122, supplier:S2, region:northeast,
    category:dairy) — matches sql/graphrag_retrieval.sql and the retrieval smoke test.
    """
    nodes: dict = {}
    edges: set = set()

    def add_node(node_id, node_type, name, props=None, path=None):
        nodes[node_id] = {"node_type": node_type, "name": name, "props": props or {}, "path": path}

    def add_edge(src, dst, rel):
        # Guard the edges->nodes foreign key: never create an edge to a node we
        # did not materialize. Upstream data can dangle — a hallucinated/filtered
        # substitute id, a supplies/surge row referencing an unknown product —
        # and an orphan edge would fail the insert/sync on live Lakebase.
        if src in nodes and dst in nodes:
            edges.add((src, dst, rel))
            return True
        dropped.append((src, dst, rel))
        return False

    dropped: list = []

    # products (+ enrichment description) and category membership
    for p in products:
        pid = p["product_id"]
        desc = enrichment.get(pid, {}).get("description", p["name"])
        add_node(
            f"product:{pid}", "Product", p["name"], {"category": p["category"], "description": desc}
        )
        add_node(category_key(p["category"]), "Category", p["category"])
        add_edge(f"product:{pid}", category_key(p["category"]), "BELONGS_TO")

    # suppliers + region (with ltree-ready path)
    for s in suppliers:
        add_node(f"supplier:{s['supplier_id']}", "Supplier", s["name"], {"city": s.get("city")})
        reg = s["region"]
        add_node(region_key(reg), "Region", reg, path=f"us.{reg.strip().lower().replace(' ', '_')}")
        add_edge(f"supplier:{s['supplier_id']}", region_key(reg), "LOCATED_IN")

    # product SUPPLIED_BY supplier (product may be absent from the products dim)
    for sp in supplies:
        add_edge(f"product:{sp['product_id']}", f"supplier:{sp['supplier_id']}", "SUPPLIED_BY")

    # substitutes from enrichment — stored once (sorted) to avoid direction dupes.
    # A substitute id may point to a product not in the dim (e.g. LLM hallucination)
    # — add_edge drops it rather than creating an orphan node reference.
    for pid, e in enrichment.items():
        for sub in e.get("substitute_ids", []):
            a, b = sorted([pid, sub])
            add_edge(f"product:{a}", f"product:{b}", "SUBSTITUTE_FOR")

    # forecast surge signals: product SURGES_IN region
    for sg in surges:
        add_edge(f"product:{sg['product_id']}", region_key(sg["region"]), "SURGES_IN")

    if dropped:
        tail = " ..." if len(dropped) > 5 else ""
        print(
            f"[graph_build] dropped {len(dropped)} edge(s) with missing endpoint(s): "
            f"{dropped[:5]}{tail}"
        )

    return nodes, edges
