"""Certified core — Unity Catalog metric views as authoritative nodes in the graph.

`sql/graphrag_retrieval.sql` boosts a node whose props carry
``{"source_method": "uc_certified"}``. Nothing produced those nodes; this module does, from
Unity Catalog **metric views** — the objects the docs call the core implementation of Unity
Catalog semantics ("reusable SQL objects that define and govern business KPIs").

WHY METRIC VIEWS AND NOTHING ELSE
Unity Catalog business semantics has four parts: metric views, domains, governed Pages, and
certification/deprecation signals. Only metric views are used here, deliberately: the required
path of a cookbook example must run for a reader without an allow-list, and metric views are
demonstrably broadly available (a survey of one workspace found 4,210 user metric views across
603 catalogs). The other three are left as future enrichment precisely so this module's
prerequisite stays "you have a metric view".

WHY THIS DOES NOT USE THE `semantic_*` SYSTEM TABLES
`information_schema.semantic_views` / `semantic_dimensions` / `semantic_metrics` /
`semantic_relationships` look like exactly the structured source this needs. They are not:
surveying one workspace, all 12 catalogs exposing them were Snowflake connections surfaced
through Lakehouse Federation, and the native catalog holding an actual Databricks metric view
exposed none of them. They describe **Snowflake** semantic views. Reading them would silently
produce a builder that works only against federated Snowflake and finds nothing in Unity Catalog.

THE READ PATH, AND WHY IT TAKES TWO QUERIES
  1. `system.information_schema.tables WHERE table_type = 'METRIC_VIEW'` — discovery. Structured
     and cheap. Internal catalogs must be filtered: Genie/Lakeview keep their own metric views
     under `__databricks_internal_catalog_lakeview_*`, and in one sample two of the first three
     hits were those rather than user objects.
  2. `DESCRIBE EXTENDED <fqn>` — the field detail. `information_schema.columns` gives names,
     types and comments for a metric view, but **cannot distinguish a measure from a dimension**:
     both `data_type` and `full_data_type` report a measure as plain `bigint`. Only DESCRIBE
     marks it, by suffixing the type with ` measure`. Since measure-vs-dimension is the whole
     point of a semantic layer, DESCRIBE is not optional.

DESCRIBE's fourth column is named `metadata` and carries the semantic payload as JSON
(`display_name`, `synonyms`), so that part is parsed as JSON rather than scraped from prose.
The one genuinely brittle coupling is the ` measure` suffix and the `# Detailed Table
Information` boundary row; both are isolated in `parse_describe` so a format change breaks one
pure function with direct test coverage, rather than the writer.

AUTHORITY METADATA
DESCRIBE also yields `Owner` and `Created Time`. Those are carried onto the emitted nodes
because they are the raw material for "who wrote it, how fresh" — the ranking in
`sql/graphrag_retrieval.sql` does not consult them today, but a richer authority function would.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

# Emitted on every node this module produces. `sql/graphrag_retrieval.sql` boosts exactly this
# value, and `graph_retrieve.py` mirrors it. Changing it here without changing both is a silent
# no-op: the nodes land, carry an unrecognised label, and are ranked as `inferred`.
CERTIFIED_SOURCE_METHOD = "uc_certified"

# Written into `source_agent` on the triplet projection so a row's producer is identifiable
# alongside `gold_triplets`' existing 'lakebase-graphrag' rows.
SOURCE_AGENT = "uc-certified-core"

# Genie and Lakeview materialise their own metric views under these catalogs. They are internal
# implementation detail, not curated business semantics, and in one sample they outnumbered user
# objects in the first page of results.
INTERNAL_CATALOG_PREFIXES = ("__databricks_internal",)

# Node types this module introduces. Kept distinct from the example's business types (Product,
# Supplier, ...) so a reader can tell curated semantic objects from graph content at a glance.
VIEW_NODE_TYPE = "MetricView"
MEASURE_NODE_TYPE = "Measure"
DIMENSION_NODE_TYPE = "Dimension"

# Relations from a metric view to its fields. Both are directional (view -> field) and therefore
# must NOT be added to `symmetric_rels` in sql/gold_triplets_mapping.sql, or they start doubling.
HAS_MEASURE = "HAS_MEASURE"
HAS_DIMENSION = "HAS_DIMENSION"

_MEASURE_SUFFIX = " measure"
_DETAIL_BOUNDARY_PREFIX = "#"


class SqlReader(Protocol):
    """Seam for running SQL against Unity Catalog.

    A Protocol rather than a concrete client so the module is testable with no warehouse: the
    smoke test injects a fake that replays recorded DESCRIBE output. Implementations may wrap
    `spark.sql`, the SQL connector, or the Statement Execution API — this module only needs
    "give me rows as lists of strings".
    """

    def rows(self, sql: str) -> list[list[Any]]: ...


@dataclass(frozen=True)
class SemanticField:
    """One dimension or measure of a metric view."""

    name: str
    data_type: str
    comment: str = ""
    is_measure: bool = False
    display_name: str = ""
    synonyms: tuple[str, ...] = ()


@dataclass(frozen=True)
class CertifiedView:
    """A metric view, resolved to the parts that carry business meaning."""

    catalog: str
    schema: str
    name: str
    fields: tuple[SemanticField, ...] = ()
    owner: str = ""
    created_at: str = ""

    @property
    def fqn(self) -> str:
        return f"{self.catalog}.{self.schema}.{self.name}"


def is_internal_catalog(catalog: str) -> bool:
    """Genie/Lakeview internals masquerade as metric views; exclude them."""
    return any(catalog.startswith(p) for p in INTERNAL_CATALOG_PREFIXES)


def discover_metric_views(
    reader: SqlReader, catalog: str | None = None,
) -> list[tuple[str, str, str]]:
    """Return (catalog, schema, name) for each user metric view.

    Scoped to one catalog when given. Unscoped, this reads every catalog the caller can see,
    which on a large workspace is thousands of rows — pass a catalog in anything but a demo.
    """
    where = ["table_type = 'METRIC_VIEW'"]
    if catalog:
        # Quote-escaped rather than interpolated raw: a catalog name is caller-supplied.
        where.append(f"table_catalog = '{catalog.replace(chr(39), chr(39) * 2)}'")
    sql = (
        "SELECT table_catalog, table_schema, table_name "
        "FROM system.information_schema.tables "
        f"WHERE {' AND '.join(where)} "
        "ORDER BY table_catalog, table_schema, table_name"
    )
    return [
        (c, s, n) for c, s, n in ((r[0], r[1], r[2]) for r in reader.rows(sql))
        if not is_internal_catalog(c)
    ]


def parse_describe(rows: list[list[Any]], catalog: str, schema: str, name: str) -> CertifiedView:
    """Turn `DESCRIBE EXTENDED <metric view>` output into a CertifiedView.

    Isolates every brittle assumption about DESCRIBE's shape, so a format change breaks this
    one pure function rather than the writer. The shape, verified against a live metric view:

      col_name         | data_type      | comment       | metadata
      ship_date        | date           | Ship date ... | {"display_name":.., "synonyms":[..]}
      total_item_count | bigint measure | All total ... | {...}
                            |                  |                |
      # Detailed Table Info |                  |                |
      Owner                 | someone@ex.com   |                |
      Created Time          | Wed Dec 17 ...   |                |

    Field rows come first; a row whose col_name starts with '#' begins the metadata block, and
    everything after it is key/value. A measure is marked ONLY by the ' measure' suffix on
    data_type — `information_schema` does not expose it at all.
    """
    fields: list[SemanticField] = []
    meta: dict[str, str] = {}
    in_detail = False

    for row in rows:
        col = (row[0] or "").strip() if len(row) > 0 else ""
        val = (row[1] or "").strip() if len(row) > 1 else ""
        if not col:
            continue                                  # spacer row between the two blocks
        if col.startswith(_DETAIL_BOUNDARY_PREFIX):
            in_detail = True
            continue
        if in_detail:
            meta[col] = val
            continue

        comment = (row[2] or "").strip() if len(row) > 2 else ""
        raw_meta = (row[3] or "").strip() if len(row) > 3 else ""
        is_measure = val.endswith(_MEASURE_SUFFIX)
        data_type = val[: -len(_MEASURE_SUFFIX)].strip() if is_measure else val

        display_name, synonyms = "", ()
        if raw_meta:
            # Malformed metadata must not lose the field: the name, type and measure flag are
            # the load-bearing parts, and display_name/synonyms are enrichment.
            try:
                parsed = json.loads(raw_meta)
            except ValueError:
                parsed = {}
            if isinstance(parsed, dict):
                display_name = str(parsed.get("display_name") or "")
                syn = parsed.get("synonyms")
                synonyms = tuple(str(s) for s in syn) if isinstance(syn, list) else ()

        fields.append(SemanticField(
            name=col, data_type=data_type, comment=comment,
            is_measure=is_measure, display_name=display_name, synonyms=synonyms,
        ))

    return CertifiedView(
        catalog=catalog, schema=schema, name=name, fields=tuple(fields),
        owner=meta.get("Owner", ""), created_at=meta.get("Created Time", ""),
    )


def read_metric_view(reader: SqlReader, catalog: str, schema: str, name: str) -> CertifiedView:
    """Discovery already gave us the identity; this fills in the semantics."""
    return parse_describe(
        reader.rows(f"DESCRIBE EXTENDED `{catalog}`.`{schema}`.`{name}`"),
        catalog, schema, name,
    )


def node_id_for_view(view: CertifiedView) -> str:
    return f"metricview:{view.fqn}"


def node_id_for_field(view: CertifiedView, f: SemanticField) -> str:
    kind = "measure" if f.is_measure else "dimension"
    return f"{kind}:{view.fqn}.{f.name}"


@dataclass
class GraphRows:
    """Nodes and edges ready for graph_build's writer, in the example's own shapes."""

    nodes: list[dict] = field(default_factory=list)
    edges: list[dict] = field(default_factory=list)


def certified_rows(view: CertifiedView) -> GraphRows:
    """Project a metric view onto nodes and edges, every node marked certified.

    One node for the view, one per field, and a directional edge from view to field. Fields
    become nodes rather than props on the view because retrieval seeds on node embeddings and
    expands along edges: a question about "revenue" must be able to match the *measure* and
    then reach its view, which is impossible if measures are buried in a JSON blob.

    `props.source_method` is set on every row — that is the whole contract with the ranking.
    """
    out = GraphRows()
    view_props = {
        "source_method": CERTIFIED_SOURCE_METHOD,
        "source_agent": SOURCE_AGENT,
        "fqn": view.fqn,
    }
    # Only carry authority metadata that is actually present; an empty string in props reads as
    # "known to be blank" rather than "not collected".
    if view.owner:
        view_props["owner"] = view.owner
    if view.created_at:
        view_props["created_at"] = view.created_at

    out.nodes.append({
        "node_id": node_id_for_view(view),
        "node_type": VIEW_NODE_TYPE,
        "name": view.name,
        "props": view_props,
    })

    for f in view.fields:
        props = {
            "source_method": CERTIFIED_SOURCE_METHOD,
            "source_agent": SOURCE_AGENT,
            "data_type": f.data_type,
            "metric_view": view.fqn,
        }
        if f.comment:
            props["comment"] = f.comment
        if f.display_name:
            props["display_name"] = f.display_name
        if f.synonyms:
            props["synonyms"] = list(f.synonyms)
        out.nodes.append({
            "node_id": node_id_for_field(view, f),
            # The human label prefers display_name: it is what a curator chose to call the
            # field, and it is what an embedding of this node should match a question against.
            "name": f.display_name or f.name,
            "node_type": MEASURE_NODE_TYPE if f.is_measure else DIMENSION_NODE_TYPE,
            "props": props,
        })
        out.edges.append({
            "src_id": node_id_for_view(view),
            "dst_id": node_id_for_field(view, f),
            "rel": HAS_MEASURE if f.is_measure else HAS_DIMENSION,
            "props": {
                "source_method": CERTIFIED_SOURCE_METHOD,
                "source_agent": SOURCE_AGENT,
                "confidence": 1.0,   # curated by a human, not inferred
            },
        })
    return out


def embedding_text(node: dict) -> str:
    """The string to embed for a certified node.

    Name plus comment plus synonyms, because a question rarely uses the column name: "how much
    did we make" should reach `total_revenue` through its display name, its description, or one
    of the synonyms a curator wrote down. Returning only the name would make the certified core
    nearly unreachable by semantic seed, and authority ranking only ever re-orders nodes that
    retrieval already reached.
    """
    props = node.get("props") or {}
    parts = [str(node.get("name") or "")]
    if props.get("comment"):
        parts.append(str(props["comment"]))
    syn = props.get("synonyms")
    if isinstance(syn, list) and syn:
        parts.extend(str(s) for s in syn)
    return " ".join(p for p in parts if p)
