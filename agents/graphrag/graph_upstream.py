"""Upstream indexing: raw document text -> gold_triplets rows.

`graph_build.py` assembles a graph from *structured* dims plus LLM enrichment. This module
covers the phase before that — turning unstructured documents into the same graph shape:

    parse -> chunk -> extract (typed triples) -> entity resolution -> gold_triplets

Parse and extract are the two steps that need a workspace: `ai_parse_document` for files in a
Volume and `ai_query` with a response schema for typed extraction. Both live in
`sql/upstream_ai_functions.sql`. Everything here is pure Python — chunking, trigram entity
resolution and triplet assembly — with extraction behind an `Extractor` protocol so the offline
smoke test drives it deterministically with no infra.

Entity resolution is the part that matters most in practice: the same entity is written many
ways across documents ("Acme", "Acme Foods", "ACME Foods Inc.") and each spelling would
otherwise become its own node, fragmenting the graph exactly where multi-hop retrieval needs it
joined. We block candidates by character-trigram similarity and merge with union-find, mirroring
`pg_trgm` from the serve layer up into the build.

Output lands in the 8-column `gold_triplets` contract (see `sql/gold_triplets_mapping.sql`), so
embeddings and Lakebase sync downstream are unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

# Predicates that are symmetric, i.e. A-rel-B means B-rel-A. graph_build.py stores these ONCE
# with sorted endpoints, and sql/graphrag_retrieval.sql walks them in both directions via its
# `adj` CTE. Upstream must follow the same convention: emitting both directions here would be
# doubled again when gold_triplets projects its symmetric reverse, so we normalize instead.
#
# SAME_AS is deliberately NOT in this set. It is directional by design — alias -> canonical —
# so that the canonical node is identifiable from the edge. Do not add it to `symmetric_rels`
# in sql/gold_triplets_mapping.sql either, or alias edges start doubling.
SYMMETRIC_PREDICATES = frozenset({"SUBSTITUTE_FOR"})

# The relation vocabulary this example's graph uses (see sql/schema.sql). Constrain extraction to
# it: left free, an LLM invents a new label per sentence — a live run over two short documents
# produced SUPPLIES_TO_RETAILERS_ACROSS, OPERATES_DISTRIBUTION_CENTER_IN, IS_LOCATED_IN and
# BELONGS_TO_CATEGORY, where the last two are near-misses for LOCATED_IN and BELONGS_TO. Every
# spelling becomes its own edge label, so the relation namespace fragments and the typed-path
# logic in sql/graphrag_retrieval.sql degrades toward an untyped walk. Entity types are already
# constrained by the response schema; predicates need the same treatment.
DEFAULT_PREDICATES = frozenset(
    {
        "SUPPLIED_BY",
        "LOCATED_IN",
        "BELONGS_TO",
        "SUBSTITUTE_FOR",
        "SURGES_IN",
    }
)


# --------------------------------------------------------------------------- chunk
@dataclass
class Passage:
    chunk_id: str
    doc_id: str
    text: str


def _split_oversized(segment: str, size: int) -> list[str]:
    """Hard-split a segment with no sentence terminator into <= `size` pieces on word boundaries.

    Parsed documents routinely contain long runs without `.!?` — tables, bulleted slides,
    headings, OCR output. Without this bound the whole run becomes one passage, and since
    extraction calls the model once per passage that means one enormous prompt.
    """
    if len(segment) <= size:
        return [segment]
    pieces, buf = [], ""
    for word in segment.split():
        if buf and len(buf) + len(word) + 1 > size:
            pieces.append(buf)
            buf = word
        else:
            buf = f"{buf} {word}".strip()
        while len(buf) > size:  # a single word longer than `size`
            pieces.append(buf[:size])
            buf = buf[size:]
    if buf:
        pieces.append(buf)
    return pieces


def _tail(text: str, overlap: int) -> str:
    """Last <= `overlap` characters, trimmed forward to a word boundary so a passage never
    begins with a word fragment ("ution center in Boston")."""
    if overlap <= 0 or not text:
        return ""
    tail = text[-overlap:]
    if len(text) > overlap and not text[-overlap - 1].isspace():
        cut = tail.find(" ")
        tail = tail[cut + 1 :] if cut != -1 else ""
    return tail.strip()


def chunk(doc_id: str, text: str, size: int = 240, overlap: int = 40) -> list[Passage]:
    """Split parsed text into overlapping passages with stable ids and a source pointer.

    Sentence-aware: packs whole sentences up to `size` characters, carrying `overlap` characters
    of word-boundary-aligned context into the next passage so a fact spanning a boundary is not
    lost. Segments with no sentence terminator are hard-split so no passage exceeds `size` by
    more than the carried overlap.
    """
    segments: list[str] = []
    for sent in re.split(r"(?<=[.!?])\s+", text.strip()):
        segments.extend(_split_oversized(sent, size))

    passages: list[Passage] = []
    buf = ""
    for s in segments:
        if not s:
            continue
        if buf and len(buf) + len(s) + 1 > size:
            passages.append(Passage(f"{doc_id}:c{len(passages)}", doc_id, buf.strip()))
            carried = _tail(buf, overlap)
            buf = f"{carried} {s}".strip() if carried else s
        else:
            buf = f"{buf} {s}".strip()
    if buf.strip():
        passages.append(Passage(f"{doc_id}:c{len(passages)}", doc_id, buf.strip()))
    return passages


# --------------------------------------------------------------------------- extract
@dataclass
class Mention:
    """One extracted triple, still carrying raw (unresolved) entity surface forms."""

    subject: str
    subject_type: str
    predicate: str
    object: str
    object_type: str
    confidence: float
    chunk_id: str


class Extractor(Protocol):
    """Turn passages into typed mentions. In production this is `ai_query` with a response
    schema (see sql/upstream_ai_functions.sql); offline it is a deterministic stub."""

    def __call__(self, passages: list[Passage]) -> list[Mention]: ...


def normalize_predicate(raw: str) -> str:
    """Uppercase-snake a raw predicate: "is based in" / "located_in" -> IS_BASED_IN / LOCATED_IN.

    Applied before the `allowed_predicates` membership test — an LLM returns "located_in" or
    "SUPPLIED BY" often enough that comparing raw output against the vocabulary would discard
    legitimately-stated, in-vocabulary relations on formatting alone.
    """
    return re.sub(r"[^A-Z0-9]+", "_", (raw or "").strip().upper()).strip("_")


# --------------------------------------------------------------------------- entity resolution
def _trigrams(s: str) -> set[str]:
    s = "  " + re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip() + "  "
    return {s[i : i + 3] for i in range(len(s) - 2)}


def trigram_sim(a: str, b: str) -> float:
    """Jaccard similarity over character trigrams — the same shape as pg_trgm's `similarity()`."""
    ta, tb = _trigrams(a), _trigrams(b)
    return len(ta & tb) / len(ta | tb) if (ta | tb) else 0.0


def trigram_containment(a: str, b: str) -> float:
    """How completely the SHORTER form is contained in the longer: |A n B| / min(|A|, |B|).

    Mirrors pg_trgm's `word_similarity()`, which scores a string against the most similar
    *portion* of another. Jaccard alone is the wrong tool for entity aliases because it
    penalizes length difference: "Acme Foods" vs "Acme Foods Incorporated" scores 0.42 by
    Jaccard (below any sane threshold) but 0.92 by containment. Legal suffixes and
    abbreviations — Inc., Ltd., Corp., initialisms — are the single most common way the same
    organization appears written differently, so a resolver that misses them fragments the
    graph precisely where multi-hop retrieval needs it joined.
    """
    ta, tb = _trigrams(a), _trigrams(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def canonical_id(name: str, node_type: str) -> str:
    """Node id for a resolved entity, as `<type>:<slug>` — the convention graph_build.py uses
    (`product:122`, `supplier:S2`).

    The type prefix is load-bearing: without it a Company "Acme Foods" and a Product "Acme
    Foods" slug to the same id, giving one graph.nodes row two conflicting node_types.
    """
    slug = re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_")
    kind = re.sub(r"[^a-z0-9]+", "_", (node_type or "entity").lower()).strip("_") or "entity"
    return f"{kind}:{slug}"


@dataclass
class Resolution:
    """Keyed by (surface_form, entity_type), not surface alone — see canonical_id()."""

    canonical_id: dict[tuple[str, str], str] = field(default_factory=dict)
    canonical_name: dict[str, str] = field(default_factory=dict)  # canonical id -> display name
    same_as: list[tuple[str, str, str]] = field(default_factory=list)  # (surface, type, canon id)


def resolve_entities(
    mentions: list[Mention],
    threshold: float = 0.55,
    containment_threshold: float = 0.8,
    min_trigrams: int = 4,
) -> Resolution:
    """Trigram blocking plus union-find: merge same-type surface forms that look like aliases.

    Two forms merge when EITHER measure clears its threshold:
      * `trigram_sim` (Jaccard) >= `threshold` — forms of comparable length;
      * `trigram_containment` >= `containment_threshold` — a short form inside a longer one
        (abbreviations, legal suffixes), which Jaccard systematically misses.

    `min_trigrams` guards the containment path: a very short surface form is trivially contained
    in many longer ones, so forms below this many trigrams only merge on the Jaccard path.

    Canonical display name is the longest surface form in a cluster, on the assumption that the
    fullest spelling ("ACME Foods Inc.") is the most identifying. Candidates are keyed by
    (surface, type) and only same-type forms merge, so a Company and a Product sharing a name
    stay distinct.

    LIMITATION worth knowing before you point this at a real corpus: containment cannot tell a
    legitimate abbreviation from a genuine short-name collision. "Acme" merging into "Acme
    Foods" is right; "Acme" merging into "Acme Widgets" when they are different companies is
    wrong, and no character-level measure can separate those two cases. On a graph where a bad
    merge is expensive, review the emitted SAME_AS edges (that is why they are written to the
    graph) or add a disambiguating signal such as co-occurring locations or identifiers.
    """
    keys: list[tuple[str, str]] = []
    seen_keys: set[tuple[str, str]] = set()
    for m in mentions:
        for surface, kind in ((m.subject, m.subject_type), (m.object, m.object_type)):
            if not (surface or "").strip():
                continue
            k = (surface, kind)
            if k not in seen_keys:
                seen_keys.add(k)
                keys.append(k)

    parent: dict[tuple[str, str], tuple[str, str]] = {k: k for k in keys}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            if a[1] != b[1]:  # only same-type forms are candidates
                continue
            if trigram_sim(a[0], b[0]) >= threshold:
                parent[find(a)] = find(b)
            elif (
                min(len(_trigrams(a[0])), len(_trigrams(b[0]))) >= min_trigrams
                and trigram_containment(a[0], b[0]) >= containment_threshold
            ):
                parent[find(a)] = find(b)

    clusters: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for k in keys:
        clusters.setdefault(find(k), []).append(k)

    res = Resolution()
    for members in clusters.values():
        rep_surface, rep_type = max(members, key=lambda k: len(k[0]))
        cid = canonical_id(rep_surface, rep_type)
        res.canonical_name[cid] = rep_surface
        for surface, kind in members:
            res.canonical_id[(surface, kind)] = cid
            # Skip an alias whose own id already equals the canonical id: case and punctuation
            # variants ("ACME Foods Inc." / "acme foods inc") slug identically, and emitting it
            # would be a self-loop that also violates graph.edges' (src, dst, rel) primary key.
            if canonical_id(surface, kind) != cid:
                res.same_as.append((surface, kind, cid))
    return res


# --------------------------------------------------------------------------- assemble
def build_gold_triplets(
    mentions: list[Mention],
    res: Resolution,
    *,
    source_agent: str = "graphrag-upstream",
    allowed_predicates: frozenset[str] | None = None,
    on_unknown: str = "keep",
) -> list[dict]:
    """Rewrite mentions onto canonical ids and emit gold_triplets rows.

    Also emits one SAME_AS row per merged alias, so the resolution decision is auditable in the
    graph itself rather than lost in the pipeline — you can always ask why two spellings became
    one node. Symmetric predicates get sorted endpoints (see SYMMETRIC_PREDICATES).

    `source_method` vocabulary: `llm_extract` for an extracted relation, `entity_resolution` for
    a SAME_AS alias edge. The view in sql/gold_triplets_mapping.sql uses `structured` and
    `llm_enrichment` for the graph_build path; all four are valid values of the same column.

    `allowed_predicates` pins the relation vocabulary (see DEFAULT_PREDICATES for why).
    Predicates are normalized before the membership test. With `on_unknown="drop"` a relation
    outside the vocabulary is discarded; with "keep" it is emitted as-is, which is fine for
    exploration but fragments the namespace at scale. Constrain the extraction prompt to the
    same vocabulary as well — filtering here only cleans up what the model already paid for.

    Extractor output is not trusted: a mention missing a subject, object or predicate is skipped,
    and a confidence that is absent or outside [0, 1] becomes None so the gap stays visible
    downstream (gold_triplets_mapping.sql applies the same rule). A surface form absent from
    `res` is resolved to its own canonical id rather than raising, so passing a Resolution
    computed over a different batch degrades instead of crashing.
    """
    if on_unknown not in ("keep", "drop"):
        raise ValueError(f"on_unknown must be 'keep' or 'drop', got {on_unknown!r}")

    def _conf(v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return round(f, 4) if 0.0 <= f <= 1.0 else None

    rows: list[dict] = []
    seen: set = set()
    for m in mentions:
        pred = normalize_predicate(m.predicate)
        if not (m.subject or "").strip() or not (m.object or "").strip() or not pred:
            continue
        if (
            allowed_predicates is not None
            and on_unknown == "drop"
            and pred not in allowed_predicates
        ):
            continue
        s_id = res.canonical_id.get(
            (m.subject, m.subject_type), canonical_id(m.subject, m.subject_type)
        )
        o_id = res.canonical_id.get(
            (m.object, m.object_type), canonical_id(m.object, m.object_type)
        )
        s_type, o_type = m.subject_type, m.object_type
        if pred in SYMMETRIC_PREDICATES and s_id > o_id:
            s_id, o_id = o_id, s_id
            s_type, o_type = o_type, s_type
        key = (s_id, pred, o_id)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "subject_id": s_id,
                "subject_type": s_type,
                "predicate": pred,
                "object_id": o_id,
                "object_type": o_type,
                "confidence": _conf(m.confidence),
                "source_method": "llm_extract",
                "source_agent": source_agent,
            }
        )
    for surface, kind, cid in res.same_as:
        alias_id = canonical_id(surface, kind)
        key = (alias_id, "SAME_AS", cid)
        if alias_id == cid or key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                # the resolved type, NOT a literal "Entity": graph.nodes.node_type is single-valued,
                # so a SAME_AS row typed differently from the same node's relation rows would make
                # the ingested node type depend on row order.
                "subject_id": alias_id,
                "subject_type": kind,
                "predicate": "SAME_AS",
                "object_id": cid,
                "object_type": kind,
                "confidence": 1.0,
                "source_method": "entity_resolution",
                "source_agent": source_agent,
            }
        )
    return rows


def index_document(
    doc_id: str,
    text: str,
    extractor: Extractor,
    *,
    source_agent: str = "graphrag-upstream",
    allowed_predicates: frozenset[str] | None = None,
    on_unknown: str = "keep",
) -> list[dict]:
    """End to end: chunk -> extract -> resolve -> gold_triplets rows.

    Parse happens upstream of this — pass already-parsed text, e.g. the output of
    `ai_parse_document` (see sql/upstream_ai_functions.sql). Pass
    `allowed_predicates=DEFAULT_PREDICATES, on_unknown="drop"` to pin the relation vocabulary.
    """
    passages = chunk(doc_id, text)
    mentions = extractor(passages)
    res = resolve_entities(mentions)
    return build_gold_triplets(
        mentions,
        res,
        source_agent=source_agent,
        allowed_predicates=allowed_predicates,
        on_unknown=on_unknown,
    )
