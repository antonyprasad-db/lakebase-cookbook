"""Certified core — Unity Catalog metric views as authoritative nodes in the graph.

`sql/graphrag_retrieval.sql` boosts a node whose props carry
``{"source_method": "uc_certified"}``. Nothing produced those nodes; this module does, from
Unity Catalog **metric views** — the objects the docs call the core implementation of Unity
Catalog semantics ("reusable SQL objects that define and govern business KPIs").

WHY METRIC VIEWS AND NOTHING ELSE
Unity Catalog business semantics has four parts: metric views, domains, governed Pages, and
certification/deprecation signals. Only metric views are used here, deliberately: the required
path of a cookbook example should run for a reader without an allow-list, and metric views are the
part of that surface with the widest reach. For calibration rather than proof: on one large
Databricks-internal workspace,
``SELECT count(*), count(DISTINCT table_catalog) FROM system.information_schema.tables
WHERE table_type = 'METRIC_VIEW' AND table_catalog NOT LIKE '__databricks_internal%'``
returned 4,210 across 603 catalogs. That is one workspace and an internal one, so treat it as
"widely used where it is used", not as a guarantee about any reader's workspace. (It does not
contradict the sampling note below: internals dominated the first page of an *unfiltered*,
unordered scan; the count above excludes them.) The other three parts are future enrichment so
this module's prerequisite stays "you have a metric view".

WHY THIS DOES NOT USE THE `semantic_*` SYSTEM TABLES
`information_schema.semantic_views` / `semantic_dimensions` / `semantic_metrics` /
`semantic_relationships` look like exactly the structured source this needs. They are not:
surveying one workspace, all 12 catalogs exposing them were Snowflake connections surfaced through
Lakehouse Federation, and the native catalog holding an actual Databricks metric view exposed none
of them. The defensible reading of that evidence: they are populated for **federated Snowflake**
semantic views and not for native Databricks metric views today. Whether that is permanent or
simply not implemented yet is unknown — either way, reading them finds nothing for a Databricks
metric view now, which is why this module does not.

THE READ PATH, AND WHY IT TAKES TWO QUERIES
  1. `system.information_schema.tables WHERE table_type = 'METRIC_VIEW'` — discovery. Structured
     and cheap. Internal catalogs must be filtered: Genie/Lakeview keep their own metric views
     under `__databricks_internal_catalog_lakeview_*`, and in one sample two of the first three
     hits were those rather than user objects.
  2. `DESCRIBE EXTENDED <fqn>` — the field detail, including which fields are measures.

     Why not `information_schema.columns`, which would be structured and need no parsing?
     Because measured against a live metric view it reports a measure as plain `bigint` in BOTH
     `data_type` and `full_data_type` — it carries no measure marker at all. This module does not
     query it, so that observation is not exercised by any test here; it is why the design went
     the way it did, not something the code proves.

     Two structured alternatives were NOT ruled out and are worth revisiting if the parsing here
     ever bites: a metric view's own YAML definition has explicit `measures:` and `dimensions:`
     blocks and is reachable via `SHOW CREATE TABLE`, and recent runtimes support
     `DESCRIBE ... AS JSON`. Either would remove the ` measure`-suffix scraping this module calls
     its one brittle coupling. They were skipped for reach (older runtimes) and because the YAML
     needs a second parser, not because they are worse.

DESCRIBE's fourth column is named `metadata` and carries the semantic payload as JSON
(`display_name`, `synonyms`), so that part is parsed as JSON rather than scraped from prose.
The one genuinely brittle coupling is the ` measure` suffix and the `# Detailed Table
Information` boundary row; both are isolated in `parse_describe` so a format change breaks one
pure function with direct test coverage, rather than the writer.

AUTHORITY METADATA
DESCRIBE also yields `Owner` and `Created Time`. Those are carried onto the emitted **view**
node (not the field nodes) because they are the raw material for "who wrote it, how fresh" — the
ranking in
`sql/graphrag_retrieval.sql` does not consult them today, but a richer authority function would.
"""

from __future__ import annotations

import json
import re
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

# Unity Catalog identifiers permitted without quoting. Deliberately narrow: this is an
# allow-list guarding string interpolation, so anything unusual should be rejected loudly
# rather than accommodated.
_CATALOG_RE = re.compile(r"[A-Za-z0-9_-]+")
_SUBHEADER_TYPE_CELL = "data_type"   # literal 2nd cell of the "# col_name" sub-header row
_SUBHEADER_NAME_CELL = "col_name"    # literal 1st cell, minus its "#"
_DETAIL_BOUNDARY = "# Detailed Table Information"

# A '#' prefix ALONE cannot mark the boundary: measures are routinely named "# Orders" or
# "# of Customers", and treating those as the boundary silently truncates the field list and
# folds every later field into the metadata dict. Section headers always have an empty
# data_type, whereas a field always has one, so that is the real discriminator.
def _is_section_header(col: str, data_type: str) -> bool:
    """Any '# ...' section header. Headers carry an empty data_type; a field always has one.

    That discriminator matters because measures are routinely named "# Orders" or
    "# of Customers", and treating every '#' as structural truncates the field list.
    """
    return col.startswith("#") and not data_type


def _is_column_subheader(col: str, data_type: str) -> bool:
    """The '# col_name | data_type | comment' SUB-HEADER row: structural, never a field.

    Third of the three DESCRIBE shape rules, kept with the other two so a format change breaks
    one predicate rather than the loop.

    Checked BEFORE _is_section_header, which matters: when this row renders with an EMPTY type
    cell it satisfies the section-header test too, and letting that win latches `in_sections` and
    silently swallows the entire field block. Measured on that rendering, ordering this first
    parses the view correctly (fields intact, owner intact) where the other order produced no
    fields at all.

    TWO signals, either sufficient, because each single signal fails in a different direction:
    keying on the '#' prefix alone truncates the field list (a measure named "# Orders" or
    "# of Customers" is routine and the suite pins it), and keying on the type cell alone leaks
    every variant rendering — measured, backquoted, double-quoted, "data type" with a space, and
    a localized cell all slid through and re-emitted the node. Neither literal can collide with
    real data: no Spark type is spelled "data_type", and no real measure is named "# col_name".

    Known limit: a fully localized sub-header, where BOTH cells are translated, is not detected.
    """
    if not col.startswith("#"):
        return False
    return (data_type.strip().lower() == _SUBHEADER_TYPE_CELL
            or col.lstrip("#").strip().lower() == _SUBHEADER_NAME_CELL)


def _is_detail_boundary(col: str) -> bool:
    """Which section header starts the key/value metadata block.

    No data_type check here: this is only ever reached once `_is_section_header` has returned
    True, which already requires an empty data_type. An earlier revision repeated the check,
    which made it dead code — a mutation removing it changed nothing, which is how it was found.
    """
    return col == _DETAIL_BOUNDARY


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
    # The metric view's own description. Lifted because without it a MetricView node has
    # nothing but its name to embed, and both the README and the notebook tell the reader to
    # embed with embedding_text() rather than the bare name.
    comment: str = ""
    # Whether any field actually carried enrichment (a display_name or synonyms). Defaults to
    # FALSE on purpose: this is an observation, so only the code path that looked may claim it was
    # there. A hand-built CertifiedView, or one of the alternative producers the module docstring
    # mentions, would otherwise silently report enrichment it never read.
    #
    # It tests CONTENT, not the row WIDTH. An uncurated view returns four columns with an empty or
    # '{}' metadata cell on every row, which is the common real case — checking `len(row) > 3`
    # would call that enriched. Tolerating the absence is right (name, type and the measure flag
    # still load); being silent about it is not, because display_name and synonyms are what make
    # the certified core reachable by semantic seed at all — nobody asks a question using
    # `total_revenue` — and authority ranking only re-orders what retrieval already found.
    enrichment_available: bool = False

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
    # `is not None`, not truthiness: catalog="" previously skipped validation AND the predicate,
    # so a caller wiring this to dbutils.widgets.get() or os.environ.get("CATALOG", "") silently
    # got the unscoped whole-workspace scan this docstring warns against. Failing open on a
    # blank config value is the wrong direction.
    if catalog is not None:
        # VALIDATED, not escaped. Doubling single quotes is the Postgres habit and it does not
        # hold here: Spark SQL honours backslash escapes, so a catalog of "x\\' OR true --"
        # survives doubling as 'x\\'' OR true --' and injects — the predicate becomes
        # (... AND table_catalog='x') OR true, returning every row and dropping the ORDER BY.
        # An allow-list is the only version of this that is actually safe.
        if not _CATALOG_RE.fullmatch(catalog):
            raise ValueError(
                f"catalog must match {_CATALOG_RE.pattern} (got {catalog!r}); "
                "identifiers are not escaped into this query"
            )
        where.append(f"table_catalog = '{catalog}'")
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
      # Detailed Table Information         |               |
      Owner                 | someone@ex.com   |                |
      Created Time          | Wed Dec 17 ...   |                |

    Field rows come first. Section headers are the rows whose col_name starts with '#' AND whose
    data_type is empty — a field always has a type, which is what keeps a measure named
    "# Orders" from being read as structure. Only `# Detailed Table Information` ends the field
    block; other sections (`# Partition Information`, `# Metadata Columns`) are skipped, because
    treating them as the boundary drops every field after them. A measure is marked ONLY by the
    ' measure' suffix on data_type — `information_schema` does not expose it at all.

    Raises ValueError if the boundary is absent (the metadata block would otherwise be emitted
    as certified fields) or if a field name repeats (duplicate primary keys downstream).

    Three DESCRIBE shape assumptions live in the predicates above, not here: the ` measure`
    type suffix, the `# Detailed Table Information` boundary row, and the
    `# col_name`/`data_type` sub-header literals.
    """
    fields: list[SemanticField] = []
    meta: dict[str, str] = {}
    in_detail = False
    saw_enrichment = False
    in_sections = False
    saw_boundary = False

    for row in rows:
        # str() coerces, rather than assuming: the SqlReader docstring advertises the SQL
        # connector and the Statement Execution API, which return typed values, so an int or a
        # datetime here would otherwise raise AttributeError on .strip().
        col = str(row[0] or "").strip() if len(row) > 0 else ""
        val = str(row[1] or "").strip() if len(row) > 1 else ""
        if not col:
            continue                                  # spacer row between the two blocks
        # THE FIELD BLOCK IS ONLY WHAT PRECEDES THE FIRST SECTION HEADER, and this is stricter
        # than it looks on purpose. DESCRIBE EXTENDED emits '# Partition Information' and
        # '# Metadata Columns' on various runtimes, each followed by a bare
        # '# col_name | data_type | comment' SUB-HEADER whose data_type cell is NOT empty, and
        # then the section's body rows. Skipping only the header row leaves both the sub-header
        # and the body being read as fields: measured, that produced
        # `dimension:<fqn>.# col_name` with props.data_type = "data_type", plus every partition
        # column, all tagged uc_certified and therefore boosted as authoritative. Anchoring on
        # "before the first header" makes the section's contents unreachable regardless of what
        # shape they take.
        # Before the section-header test on purpose — see _is_column_subheader's docstring: with
        # an empty type cell this row looks like a section header, and letting that win swallows
        # the whole field block.
        if _is_column_subheader(col, val):
            continue
        if _is_section_header(col, val):
            in_sections = True
            if _is_detail_boundary(col):
                in_detail = True
                saw_boundary = True
            continue
        if in_sections and not in_detail:
            continue    # inside a non-detail section: sub-header or body, never a field
        if in_detail:
            # First NON-EMPTY write wins. The real detail block is read before anything following
            # it, so a later section cannot beat it — that closes the overwrite where a following
            # block's own Owner or Comment row replaced the authority metadata (and `comment`
            # feeds embedding_text(), so the node ended up embedded from a foreign section while
            # still tagged uc_certified).
            #
            # The `val and` half matters independently: a plain setdefault makes an EMPTY first
            # value permanent, so a blank Owner or Comment row followed by a populated one
            # discarded the real value — which certified_rows() then omits entirely, leaving the
            # MetricView node with only its name to embed. Last-write-wins used to recover that.
            if val and col not in meta:
                meta[col] = val
            continue

        if not val:
            # By this module's own discriminator a field always has a data_type, so a typeless
            # row that is not a '#' header is not a field either. Emitting it would create a
            # certified Dimension whose data_type is "".
            continue
        comment = str(row[2] or "").strip() if len(row) > 2 else ""
        raw_meta = str(row[3] or "").strip() if len(row) > 3 else ""
        # Case-insensitive: this is the module's one brittle coupling, and a runtime emitting
        # "BIGINT MEASURE" would otherwise classify every measure as a Dimension, silently,
        # while still tagging it certified. Cheap insurance on the one signal that matters.
        is_measure = val.lower().endswith(_MEASURE_SUFFIX)
        data_type = val[: -len(_MEASURE_SUFFIX)].strip() if is_measure else val

        display_name, synonyms = "", ()
        # set below, once we know the parse actually produced something
        if raw_meta:
            # Malformed metadata must not lose the field: the name, type and measure flag are
            # the load-bearing parts, and display_name/synonyms are enrichment.
            try:
                parsed = json.loads(raw_meta)
            except ValueError:
                parsed = {}
            if isinstance(parsed, dict):
                # isinstance, matching the synonyms guard below: a non-string value
                # stringifies a Python repr into the node name and the embedded text
                # ({"a": 1} became "{'a': 1}").
                dn = parsed.get("display_name")
                display_name = dn.strip() if isinstance(dn, str) else ""
                syn = parsed.get("synonyms")
                # Only string elements: a malformed list like [null, 5, {"a": 1}] would
                # otherwise reach props.synonyms and the embedded text as "None", "5" and
                # "{'a': 1}".
                synonyms = (tuple(x for x in syn if isinstance(x, str) and x)
                            if isinstance(syn, list) else ())

        if display_name or synonyms:
            saw_enrichment = True
        fields.append(SemanticField(
            name=col, data_type=data_type, comment=comment,
            is_measure=is_measure, display_name=display_name, synonyms=synonyms,
        ))

    # Refuse to guess when the shape is not what we expect. Without this, a renamed or absent
    # boundary turns the metadata block itself into fields: 'Owner' becomes a dimension node
    # whose data_type is somebody's email address, tagged uc_certified and therefore BOOSTED by
    # the ranking. Marking garbage authoritative is the worst available outcome, so this raises
    # rather than degrading.
    if not saw_boundary:
        raise ValueError(
            f"DESCRIBE output for {catalog}.{schema}.{name} has no {_DETAIL_BOUNDARY!r} row; "
            "refusing to parse, because without it the metadata block would be emitted as "
            "certified fields"
        )

    # F4 — duplicate names would collide on graph.nodes' PK and on edges' (src,dst,rel) PK,
    # so the caller's load either aborts or silently drops rows. Surface it here instead.
    names = [f.name for f in fields]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(
            f"DESCRIBE output for {catalog}.{schema}.{name} repeats field name(s) {dupes}; "
            "these would produce duplicate node_id and duplicate (src_id, dst_id, rel), both "
            "primary keys in sql/schema.sql"
        )

    return CertifiedView(
        catalog=catalog, schema=schema, name=name, fields=tuple(fields),
        owner=meta.get("Owner", ""), created_at=meta.get("Created Time", ""),
        comment=meta.get("Comment", ""), enrichment_available=saw_enrichment,
    )


def read_metric_view(reader: SqlReader, catalog: str, schema: str, name: str) -> CertifiedView:
    """Discovery already gave us the identity; this fills in the semantics.

    All three identifiers are validated, not quoted-and-hoped. Backtick quoting is not a
    safety boundary: Spark escapes a backtick by doubling it, so a name containing one breaks
    out of the quotes — `DESCRIBE EXTENDED \`c\`.\`s\`.\`n\` ; DROP TABLE ... --\`` is a
    valid statement pair. Discovery normally supplies these from
    information_schema, but the function is public and must not assume that.
    """
    for part, label in ((catalog, "catalog"), (schema, "schema"), (name, "name")):
        if not _CATALOG_RE.fullmatch(part):
            raise ValueError(
                f"{label} must match {_CATALOG_RE.pattern} (got {part!r}); identifiers are "
                "interpolated into this query and backticks do not make that safe"
            )
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
    """Nodes and edges for a caller to load.

    Note the shapes differ from `graph_build.assemble_graph`, which returns nodes as a dict keyed
    by node_id (values carrying a `path`) and edges as a set of bare `(src, dst, rel)` tuples with
    no props. These are lists of dicts with an inline `node_id`, no `path`, and edge `props`. Both
    satisfy `sql/schema.sql` — `path` is nullable and edge `props` defaults to `'{}'` — but a
    loader written against `assemble_graph`'s output needs adapting, and the naive adaptation
    drops the edge props that carry the whole provenance contract.
    """

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
    if view.comment:
        view_props["comment"] = view.comment

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
            # The raw column name, because `name` may hold a curator's display_name and the
            # column is otherwise recoverable only by string-parsing node_id — which blocks a
            # consumer from generating SQL against the metric view.
            "field_name": f.name,
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
