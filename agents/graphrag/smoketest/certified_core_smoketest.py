"""Offline checks for graph_certified.py — no warehouse, no Lakebase, no model endpoint.

The DESCRIBE fixture below is VERBATIM output from a live Unity Catalog metric view
(cchan_catalog.metric_view_lab.order_details_metric_view), trimmed in field count but not in
shape: same four columns, same ' measure' type suffix, same spacer row, same
'# Detailed Table Information' boundary, same metadata JSON. Parsing DESCRIBE is the one
brittle coupling in the module, so it is pinned against real output rather than an invention.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from graph_certified import (  # noqa: E402
    CERTIFIED_SOURCE_METHOD,
    DIMENSION_NODE_TYPE,
    HAS_DIMENSION,
    HAS_MEASURE,
    MEASURE_NODE_TYPE,
    VIEW_NODE_TYPE,
    certified_rows,
    discover_metric_views,
    embedding_text,
    is_internal_catalog,
    parse_describe,
    read_metric_view,
)

FAILURES: list[str] = []


def check(name: str, got, want) -> None:
    ok = got == want
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got={got!r} want={want!r}")
    if not ok:
        FAILURES.append(name)


# --- verbatim DESCRIBE EXTENDED shape (4 columns) --------------------------------------------
DESCRIBE_ROWS = [
    ["order_ID", "bigint", "Customer's order ID",
     '{"display_name":"Order ID","synonyms":["Order ID","ID of Order"]}'],
    ["ship_date", "date", "Ship date of order",
     '{"display_name":"Ship Date","synonyms":["Shipping date","ship date"]}'],
    ["total_revenue", "decimal(37,2) measure", "Total revenue after discounts",
     '{"display_name":"Total Revenue ($)","synonyms":["Revenue","Total Revenue"]}'],
    ["total_item_count", "bigint measure", "All total ordered item count", '{}'],
    ["broken_meta", "string", "metadata is not JSON", "{not json"],
    ["", "", "", ""],
    ["# Detailed Table Information", "", "", ""],
    ["Catalog", "cchan_catalog", "", ""],
    ["Database", "metric_view_lab", "", ""],
    ["Table", "order_details_metric_view", "", ""],
    ["Owner", "chitra.chandrakumar@databricks.com", "", ""],
    ["Created Time", "Wed Dec 17 17:27:39 UTC 2025", "", ""],
    ["Type", "METRIC_VIEW", "", ""],
]


class FakeReader:
    """Injected through the SqlReader Protocol — the reason no warehouse is needed."""

    def __init__(self, tables_rows, describe_rows):
        self._tables, self._describe, self.seen = tables_rows, describe_rows, []

    def rows(self, sql):
        self.seen.append(sql)
        return self._describe if sql.startswith("DESCRIBE") else self._tables


def main() -> int:
    print("== TEST 1: DESCRIBE parsing (the module's one brittle coupling) ==")
    view = parse_describe(DESCRIBE_ROWS, "cchan_catalog", "metric_view_lab",
                          "order_details_metric_view")
    check("fqn is assembled", view.fqn, "cchan_catalog.metric_view_lab.order_details_metric_view")
    check("field count excludes the spacer and the metadata block", len(view.fields), 5)
    check("owner is lifted from the metadata block", view.owner,
          "chitra.chandrakumar@databricks.com")
    check("created time is lifted", view.created_at, "Wed Dec 17 17:27:39 UTC 2025")
    check("metadata keys did NOT leak in as fields",
          [f.name for f in view.fields if f.name in ("Owner", "Catalog", "Type")], [])

    by_name = {f.name: f for f in view.fields}
    # the ' measure' suffix is the ONLY signal distinguishing a measure; information_schema
    # reports both as plain types, so this is the assertion that matters most in the file
    check("a measure is detected from the type suffix", by_name["total_revenue"].is_measure, True)
    check("the suffix is stripped from the type", by_name["total_revenue"].data_type,
          "decimal(37,2)")
    check("a dimension is not a measure", by_name["ship_date"].is_measure, False)
    check("dimension type is untouched", by_name["ship_date"].data_type, "date")
    check("display_name is read from the metadata JSON", by_name["total_revenue"].display_name,
          "Total Revenue ($)")
    check("synonyms are read from the metadata JSON", list(by_name["ship_date"].synonyms),
          ["Shipping date", "ship date"])
    check("empty metadata JSON is harmless", by_name["total_item_count"].display_name, "")
    # a curator's typo in the metadata must not cost us the field entirely
    check("MALFORMED metadata still yields the field", by_name["broken_meta"].name, "broken_meta")
    check("malformed metadata degrades to no display_name",
          by_name["broken_meta"].display_name, "")
    check("malformed metadata keeps the measure flag correct",
          by_name["broken_meta"].is_measure, False)

    print("\n== TEST 2: discovery filters Genie/Lakeview internals ==")
    check("internal prefix is recognised",
          is_internal_catalog("__databricks_internal_catalog_lakeview_1444828305810485"), True)
    check("a user catalog is not internal", is_internal_catalog("cchan_catalog"), False)
    reader = FakeReader(
        [["cchan_catalog", "metric_view_lab", "order_details_metric_view"],
         ["__databricks_internal_catalog_lakeview_144", "abc", "lakegenie_01f158"],
         ["another_catalog", "sales", "revenue_mv"]],
        DESCRIBE_ROWS,
    )
    found = discover_metric_views(reader)
    check("internal metric views are dropped", [c for c, _, _ in found],
          ["cchan_catalog", "another_catalog"])
    check("discovery filters on table_type", "table_type = 'METRIC_VIEW'" in reader.seen[0], True)
    scoped = FakeReader([["c", "s", "n"]], DESCRIBE_ROWS)
    discover_metric_views(scoped, catalog="my_cat")
    check("a catalog scope reaches the query", "table_catalog = 'my_cat'" in scoped.seen[0], True)
    quoted = FakeReader([], DESCRIBE_ROWS)
    discover_metric_views(quoted, catalog="o'brien")
    check("a quote in the catalog name is escaped, not injected",
          "table_catalog = 'o''brien'" in quoted.seen[0], True)

    print("\n== TEST 3: projection onto graph rows ==")
    rows = certified_rows(view)
    check("one node for the view plus one per field", len(rows.nodes), 6)
    check("one edge per field", len(rows.edges), 5)
    # the contract with sql/graphrag_retrieval.sql: without this the nodes rank as inferred
    check("EVERY node carries the certified marker",
          {n["props"]["source_method"] for n in rows.nodes}, {CERTIFIED_SOURCE_METHOD})
    check("every edge carries it too",
          {e["props"]["source_method"] for e in rows.edges}, {CERTIFIED_SOURCE_METHOD})
    types = {n["node_type"] for n in rows.nodes}
    check("node types are view/measure/dimension", types,
          {VIEW_NODE_TYPE, MEASURE_NODE_TYPE, DIMENSION_NODE_TYPE})
    check("measures get the measure relation",
          sorted({e["rel"] for e in rows.edges}), sorted([HAS_MEASURE, HAS_DIMENSION]))
    view_node = next(n for n in rows.nodes if n["node_type"] == VIEW_NODE_TYPE)
    check("authority metadata rides on the view node", view_node["props"]["owner"],
          "chitra.chandrakumar@databricks.com")
    check("node ids are namespaced and stable",
          sorted(n["node_id"] for n in rows.nodes)[0],
          "dimension:cchan_catalog.metric_view_lab.order_details_metric_view.broken_meta")
    rev = next(n for n in rows.nodes if n["node_id"].endswith(".total_revenue"))
    check("a field node prefers the curator's display name", rev["name"], "Total Revenue ($)")
    check("a field with no display_name falls back to its column name",
          next(n for n in rows.nodes if n["node_id"].endswith(".total_item_count"))["name"],
          "total_item_count")
    # absent metadata must be absent, not an empty string that reads as "known to be blank"
    bare = parse_describe([["x", "int", "", ""], ["# Detailed Table Information", "", "", ""]],
                          "c", "s", "n")
    check("no owner means no owner key at all",
          "owner" in certified_rows(bare).nodes[0]["props"], False)

    print("\n== TEST 4: embedding text (reachability of the certified core) ==")
    # authority ranking only ever RE-ORDERS nodes retrieval already reached, so a certified node
    # that no question can match is invisible no matter how high the boost
    check("embedding text includes name, comment and synonyms",
          embedding_text(rev),
          "Total Revenue ($) Total revenue after discounts Revenue Total Revenue")
    check("embedding text of a bare node is just its name",
          embedding_text({"name": "x", "props": {}}), "x")

    print("\n== TEST 5: read_metric_view wires discovery to parsing ==")
    r2 = FakeReader([], DESCRIBE_ROWS)
    v2 = read_metric_view(r2, "cchan_catalog", "metric_view_lab", "order_details_metric_view")
    check("identifiers are backtick-quoted in DESCRIBE",
          "`cchan_catalog`.`metric_view_lab`.`order_details_metric_view`" in r2.seen[0], True)
    check("parsed view matches the direct parse", v2, view)

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} FAILED -> {FAILURES}")
        return 1
    print("RESULT: ALL PASSED — certified-core builder verified offline.")
    print("(Live Unity Catalog read path exercised separately; see the README.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
