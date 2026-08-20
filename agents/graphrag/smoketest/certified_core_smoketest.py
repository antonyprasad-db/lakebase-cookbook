"""Offline checks for graph_certified.py — no warehouse, no Lakebase, no model endpoint.

The DESCRIBE fixture below is VERBATIM output from a live Unity Catalog metric view
(cchan_catalog.metric_view_lab.order_details_metric_view), trimmed in field count but not in
shape: same four columns, same ' measure' type suffix, same spacer row, same
'# Detailed Table Information' boundary, same metadata JSON. Parsing DESCRIBE is the one brittle
coupling in the module, so its SHAPE is pinned against real output rather than invented.

Two rows ARE synthetic and marked as such: `broken_meta` (metadata that is not JSON) and the
hostile inputs below. No live DESCRIBE emits '{not json'; that row exists to pin the degradation
path, and calling the whole fixture verbatim would overstate it.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from graph_certified import (  # noqa: E402
    CERTIFIED_SOURCE_METHOD,
    HAS_DIMENSION,
    HAS_MEASURE,
    VIEW_NODE_TYPE,
    certified_rows,
    discover_metric_views,
    embedding_text,
    is_internal_catalog,
    node_id_for_field,
    node_id_for_view,
    parse_describe,
    read_metric_view,
)

FAILURES: list[str] = []

# Every realistic DESCRIBE EXTENDED carries this row, and parse_describe now REQUIRES it: without
# it the metadata block would be emitted as certified fields (an owner's email becoming a node's
# data_type, tagged authoritative). Fixtures include it for that reason.
BOUNDARY = ["# Detailed Table Information", "", "", ""]


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

    print("\n== TEST 1b: hostile and legacy DESCRIBE shapes ==")
    # a measure named "# Orders" is routine, and treating '#' alone as the metadata boundary
    # truncated the field list AND folded later fields into the metadata dict
    hashed = parse_describe(
        [["ship_date", "date", "d", "{}"],
         ["# Orders", "bigint measure", "count", "{}"],
         ["total_revenue", "decimal(10,2) measure", "rev", "{}"],
         ["", "", "", ""],
         ["# Detailed Table Information", "", "", ""],
         ["Owner", "me@example.com", "", ""]],
        "c", "s", "n")
    check("a field named '# Orders' is not mistaken for the metadata boundary",
          [f.name for f in hashed.fields], ["ship_date", "# Orders", "total_revenue"])
    check("and it keeps its measure flag",
          next(f.is_measure for f in hashed.fields if f.name == "# Orders"), True)
    check("the real boundary still ends the field block", hashed.owner, "me@example.com")

    # A 3-column DESCRIBE EXTENDED (older DBR): enrichment degrades, it does not raise. NOT the
    # plain-DESCRIBE shape — that emits no detail block at all and now raises, by design.
    three = parse_describe([["total_revenue", "decimal(10,2) measure", "rev"],
                            ["# Detailed Table Information", "", ""]], "c", "s", "n")
    check("a 3-column DESCRIBE still yields the field", len(three.fields), 1)
    check("...and keeps the measure flag", three.fields[0].is_measure, True)
    check("...with enrichment simply absent", three.fields[0].display_name, "")
    # 2-column rows: the boundary row is still required, and a 2-column boundary must work
    check("a 2-column row does not raise",
          len(parse_describe([["f", "int"], ["# Detailed Table Information", ""]],
                             "c", "s", "n").fields), 1)

    # metadata that is valid JSON but the wrong type
    for bad_meta in ("[1,2]", '"a string"', "123", "null"):
        got = parse_describe([["f", "int", "c", bad_meta], BOUNDARY], "c", "s", "n").fields[0]
        check(f"metadata {bad_meta!r} degrades safely", (got.display_name, got.synonyms), ("", ()))
    nonlist = parse_describe([["f", "int", "c", '{"synonyms":"notalist"}'], BOUNDARY],
                             "c", "s", "n")
    check("a non-list synonyms value is ignored", nonlist.fields[0].synonyms, ())
    dirty = parse_describe([["f", "int", "c", '{"synonyms":[null,5,{"a":1},"real",""]}'],
                            BOUNDARY], "c", "s", "n")
    check("non-string synonym elements are dropped", dirty.fields[0].synonyms, ("real",))

    # a DIFFERENT '# ...' section must be skipped, not treated as the boundary: latching there
    # drops every field after it. This pins the empty-data_type discriminator from the second
    # side — without it, reverting the fix to `col == _DETAIL_BOUNDARY` alone stays green.
    sectioned = parse_describe(
        [["a", "int", "", ""],
         ["# Partition Information", "", "", ""],
         ["p", "string", "", ""],
         ["# Metadata Columns", "", "", ""],
         ["q", "int", "", ""],
         ["# Detailed Table Information", "", "", ""],
         ["Owner", "me@example.com", "", ""]],
        "c", "s", "n")
    # absurd but it pins the clause: even a FIELD named exactly like the boundary is a field,
    # because it carries a data_type. Without `and not data_type` this row ends the field list.
    impostor = parse_describe(
        [["# Detailed Table Information", "bigint measure", "a field, not a header", "{}"],
         ["real_field", "int", "", ""],
         ["# Detailed Table Information", "", "", ""],
         ["Owner", "me@example.com", "", ""]],
        "c", "s", "n")
    check("a FIELD named like the boundary is still a field (it has a data_type)",
          [f.name for f in impostor.fields],
          ["# Detailed Table Information", "real_field"])
    check("...and the real (typeless) boundary still ends the block", impostor.owner,
          "me@example.com")

    # the REAL Spark shape: a section header is followed by a bare '# col_name | data_type |
    # comment' SUB-HEADER whose data_type is NOT empty, then body rows. Skipping only the header
    # left both being read as fields — measured, that emitted `dimension:<fqn>.# col_name` with
    # data_type "data_type", plus every partition column, all tagged certified and boosted.
    spark_shape = parse_describe(
        [["a", "int", "", ""],
         ["# Partition Information", "", "", ""],
         ["# col_name", "data_type", "comment", ""],
         ["region", "string", "", ""],
         ["# Metadata Columns", "", "", ""],
         ["_metadata", "struct", "", ""],
         ["# Detailed Table Information", "", "", ""],
         ["Owner", "me@example.com", "", ""]],
        "c", "s", "n")
    check("a section's SUB-HEADER and BODY are not read as fields",
          [f.name for f in spark_shape.fields], ["a"])
    check("...and the detail block is still reached", spark_shape.owner, "me@example.com")

    # uppercase MEASURE: the one signal that matters must not degrade silently
    upper = parse_describe([["m", "BIGINT MEASURE", "", ""], BOUNDARY, ["Owner", "x", "", ""]],
                           "c", "s", "n")
    check("the measure suffix match is case-insensitive", upper.fields[0].is_measure, True)

    # non-string display_name stringified a Python repr into the node name
    nonstr = parse_describe([["r", "int", "c", '{"display_name":{"a":1}}'], BOUNDARY,
                             ["Owner", "x", "", ""]], "c", "s", "n")
    check("a non-string display_name is ignored, not repr'd", nonstr.fields[0].display_name, "")

    # a typeless row that is not a '#' header is not a field by this module's own discriminator
    ghost = parse_describe([["good", "int", "", ""], ["ghost", "", "", ""], BOUNDARY,
                            ["Owner", "x", "", ""]], "c", "s", "n")
    check("a typeless non-header row is not emitted as a field",
          [f.name for f in ghost.fields], ["good"])

    # the view node had only its name to embed, because Comment was never lifted
    with_comment = parse_describe(
        [["r", "int", "c", "{}"], BOUNDARY,
         ["Comment", "Order-level revenue metrics", "", ""], ["Owner", "x", "", ""]],
        "c", "s", "order_details")
    check("the metric view's own Comment is lifted", with_comment.comment,
          "Order-level revenue metrics")
    check("so a MetricView node embeds more than its bare name",
          embedding_text(certified_rows(with_comment).nodes[0]),
          "order_details Order-level revenue metrics")

    # This previously asserted ["a","p","q"] — i.e. that fields AFTER a section header survive.
    # That was the wrong contract: rows after a section header belong to that section, not to the
    # field list, which is why the sub-header and partition columns were leaking in as certified
    # fields. The field block ends at the FIRST section header.
    check("the field block ends at the first section header",
          [f.name for f in sectioned.fields], ["a"])
    check("...and the detail block is still reached", sectioned.owner, "me@example.com")

    # a missing boundary must RAISE: otherwise 'Owner' becomes a dimension node whose data_type
    # is an email address, tagged uc_certified and therefore boosted as authoritative
    raised = False
    try:
        parse_describe([["Owner", "real@example.com", "", ""], ["Type", "METRIC_VIEW", "", ""]],
                       "c", "s", "n")
    except ValueError:
        raised = True
    check("a DESCRIBE with no boundary raises rather than emitting metadata as fields",
          raised, True)

    # ...and a SECTION is not the boundary. If _is_detail_boundary matched any '# ...' header,
    # '# Partition Information' would satisfy the guard above, so output with sections but no
    # detail block would parse — losing Owner/Created Time silently and defeating the check.
    section_only = False
    try:
        parse_describe([["a", "int", "", ""],
                        ["# Partition Information", "", "", ""],
                        ["region", "string", "", ""]], "c", "s", "n")
    except ValueError:
        section_only = True
    check("a section header does not satisfy the boundary requirement", section_only, True)

    # duplicate names would collide on graph.nodes' PK and on edges' (src,dst,rel) PK
    dup_raised = False
    try:
        parse_describe([["amount", "int", "", ""], ["amount", "bigint", "", ""],
                        ["# Detailed Table Information", "", "", ""]], "c", "s", "n")
    except ValueError:
        dup_raised = True
    check("duplicate field names raise rather than producing duplicate primary keys",
          dup_raised, True)

    # non-string cells: SqlReader advertises the SQL connector and Statement Execution API,
    # which return typed values, so .strip() on an int must not blow up
    typed = parse_describe([[123, 456, None, None],
                            ["# Detailed Table Information", "", "", ""],
                            ["Owner", "x", "", ""]], "c", "s", "n")
    check("non-string cells coerce instead of raising", typed.fields[0].data_type, "456")

    # a whitespace-only display_name must not replace the real column name
    ws = parse_describe([["rev", "int", "c", '{"display_name":"   "}'],
                         ["# Detailed Table Information", "", "", ""],
                         ["Owner", "x", "", ""]], "c", "s", "n")
    ws_node = certified_rows(ws).nodes[1]
    check("a whitespace-only display_name falls back to the column name", ws_node["name"], "rev")
    check("the raw column name is recoverable from props, not just node_id",
          ws_node["props"].get("field_name"), "rev")

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
    check("discovery orders its results", "ORDER BY" in reader.seen[0], True)
    # startswith, not substring: a user catalog named x__databricks_internal_y is legitimate
    check("the internal filter matches a PREFIX, not a substring",
          is_internal_catalog("x__databricks_internal_y"), False)
    scoped = FakeReader([["c", "s", "n"]], DESCRIBE_ROWS)
    discover_metric_views(scoped, catalog="my_cat")
    check("a catalog scope reaches the query", "table_catalog = 'my_cat'" in scoped.seen[0], True)
    # Quote-doubling was the original approach and it is NOT safe here: Spark honours backslash
    # escapes, so "x\\' OR true --" survives doubling and injects, turning the predicate into
    # (... AND table_catalog='x') OR true — every row, ORDER BY dropped. Validation, not escaping.
    for hostile in ("x\\' OR true --", "x'; DROP TABLE y--", "x\\", "a b", "'", "o'brien"):
        rejected = False
        try:
            discover_metric_views(FakeReader([], DESCRIBE_ROWS), catalog=hostile)
        except ValueError:
            rejected = True
        check(f"catalog {hostile!r} is rejected, not interpolated", rejected, True)
    # catalog="" previously skipped BOTH validation and the predicate, silently becoming the
    # unscoped whole-workspace scan the docstring warns against — failing open on a blank config
    # value, which is exactly what dbutils.widgets.get() returns when unset.
    empty_rejected = False
    try:
        discover_metric_views(FakeReader([], DESCRIBE_ROWS), catalog="")
    except ValueError:
        empty_rejected = True
    check("an empty catalog is rejected, not silently unscoped", empty_rejected, True)
    unscoped = FakeReader([], DESCRIBE_ROWS)
    discover_metric_views(unscoped, catalog=None)
    check("catalog=None is unscoped by design",
          "table_catalog = " in unscoped.seen[0], False)

    for ok_name in ("my_catalog", "cchan_catalog", "a-b_1"):
        accepted = True
        try:
            discover_metric_views(FakeReader([], DESCRIBE_ROWS), catalog=ok_name)
        except ValueError:
            accepted = False
        check(f"a legitimate catalog {ok_name!r} is accepted", accepted, True)

    print("\n== TEST 3: projection onto graph rows ==")
    rows = certified_rows(view)
    check("one node for the view plus one per field", len(rows.nodes), 6)
    check("one edge per field", len(rows.edges), 5)
    # the contract with sql/graphrag_retrieval.sql: without this the nodes rank as inferred
    check("EVERY node carries the certified marker",
          {n["props"]["source_method"] for n in rows.nodes}, {CERTIFIED_SOURCE_METHOD})
    check("every edge carries it too",
          {e["props"]["source_method"] for e in rows.edges}, {CERTIFIED_SOURCE_METHOD})
    # Per field, NOT set-membership over {Measure, Dimension}: a wholesale swap of the two is
    # invisible to a set assertion, and this mapping is the module's entire reason for existing.
    by_id = {n["node_id"]: n for n in rows.nodes}
    rev_id = node_id_for_field(view, by_name["total_revenue"])
    ship_id = node_id_for_field(view, by_name["ship_date"])
    # LITERALS, not the imported constants: comparing a value against the constant that produced
    # it is a tautology, so swapping MEASURE_NODE_TYPE and DIMENSION_NODE_TYPE's values (or
    # renaming VIEW_NODE_TYPE) stayed green while the docs kept documenting the old names.
    check("a measure becomes a Measure node", (by_id.get(rev_id) or {}).get("node_type"),
          "Measure")
    check("a dimension becomes a Dimension node", (by_id.get(ship_id) or {}).get("node_type"),
          "Dimension")
    check("the view node type is literally MetricView",
          (by_id.get(node_id_for_view(view)) or {}).get("node_type"), "MetricView")
    # keyed by dst_id, which only works while edges point view -> field. Use .get so a
    # direction regression reports as a failed check rather than a KeyError traceback.
    edge_for = {e["dst_id"]: e for e in rows.edges}
    check("a measure edge is HAS_MEASURE", (edge_for.get(rev_id) or {}).get("rel"), HAS_MEASURE)
    check("a dimension edge is HAS_DIMENSION", (edge_for.get(ship_id) or {}).get("rel"),
          HAS_DIMENSION)
    # direction is asserted because README/mdx and the ASCII diagram all promise view -> field
    check("edges point view -> field, not the reverse",
          (edge_for.get(rev_id) or {}).get("src_id"), node_id_for_view(view))

    # The LITERAL value, once. Comparing against the imported constant is a tautology: rename the
    # constant to a typo and the suite stays green while sql/graphrag_retrieval.sql and
    # graph_retrieve.py keep hardcoding 'uc_certified', so the nodes land and rank as inferred —
    # the exact silent no-op the module docstring warns about.
    check("the certified marker is literally 'uc_certified'",
          ((by_id.get(rev_id) or {}).get("props") or {}).get("source_method"), "uc_certified")
    check("the relation literals are HAS_MEASURE / HAS_DIMENSION",
          ((edge_for.get(rev_id) or {}).get("rel"), (edge_for.get(ship_id) or {}).get("rel")),
          ("HAS_MEASURE", "HAS_DIMENSION"))
    # both ride into gold_triplets' 8-column contract, so a wrong value ships downstream
    check("source_agent is set on nodes and edges",
          (((by_id.get(rev_id) or {}).get("props") or {}).get("source_agent"),
           ((edge_for.get(rev_id) or {}).get("props") or {}).get("source_agent")),
          ("uc-certified-core", "uc-certified-core"))
    check("curated edges carry confidence 1.0",
          ((edge_for.get(rev_id) or {}).get("props") or {}).get("confidence"), 1.0)
    view_node = next(n for n in rows.nodes if n["node_type"] == VIEW_NODE_TYPE)
    check("authority metadata rides on the view node",
          (view_node.get("props") or {}).get("owner"), "chitra.chandrakumar@databricks.com")
    # these three ride into gold_triplets / are the reason DESCRIBE is mandatory, and all three
    # previously survived removal
    check("a field node carries its metric_view back-reference",
          ((by_id.get(rev_id) or {}).get("props") or {}).get("metric_view"),
          "cchan_catalog.metric_view_lab.order_details_metric_view")
    check("a field node carries its data_type",
          ((by_id.get(rev_id) or {}).get("props") or {}).get("data_type"), "decimal(37,2)")
    check("a field node carries its comment when present",
          ((by_id.get(rev_id) or {}).get("props") or {}).get("comment"),
          "Total revenue after discounts")
    check("the view node id carries the metricview: prefix", node_id_for_view(view),
          "metricview:cchan_catalog.metric_view_lab.order_details_metric_view")
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
    check("no created time means no created_at key either",
          "created_at" in certified_rows(bare).nodes[0]["props"], False)
    check("...but a present created_at IS carried",
          certified_rows(view).nodes[0]["props"].get("created_at"),
          "Wed Dec 17 17:27:39 UTC 2025")

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
    # plain DESCRIBE returns no 4th metadata column and no metadata block, so display_name,
    # synonyms, Owner and Created Time would all silently vanish
    check("it is DESCRIBE EXTENDED, not plain DESCRIBE",
          r2.seen[0].startswith("DESCRIBE EXTENDED "), True)
    # backticks are NOT a safety boundary: Spark escapes one by doubling it, so a name holding a
    # backtick breaks out. All three identifiers are validated, not just the catalog.
    for part in ("n` ; DROP TABLE graph.nodes -- ", "x' OR true --", "a b"):
        rej = False
        try:
            read_metric_view(FakeReader([], DESCRIBE_ROWS), "c", "s", part)
        except ValueError:
            rej = True
        check(f"read_metric_view rejects a hostile name {part[:22]!r}", rej, True)
    rej_cat = False
    try:
        read_metric_view(FakeReader([], DESCRIBE_ROWS), "x' OR true --", "s", "n")
    except ValueError:
        rej_cat = True
    check("read_metric_view validates the catalog too", rej_cat, True)
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
