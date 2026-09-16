"""Embed the narrative half of the fund fact sheets into a `fund_factsheets`
Chroma collection (PLAN.md §5 step 3 / §4.1's "both stores, split by field
type").

Reads data/processed/fund_factsheets_structured.json (built by
ingest_factsheet.py) rather than re-parsing the PDFs - that extraction is
already grounded, so this step only has to decide chunk boundaries and
embed, not extract. Numeric/lookup fields (SRI, minimum investment,
key_facts rows) are deliberately NOT duplicated here - they live in the
`funds` / `fund_key_facts` SQL tables (ingest_clients.py) where a suitability
check can join against them. This collection carries only what a semantic
search over "which funds are risky for X reason" or a citation needs
verbatim: the objective, who it's for, the risks, and any extra_information
that didn't fit the structured schema.

One chunk per section per fund (objective / who-for / key-risks / each
extra_information item), tagged with fund_name + section so retrieval can
be scoped to one fund or searched across all of them.

Usage:
    python ingest_fund_vectors.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from pypdf import PdfReader

# rerank.py lives at the repository root (sibling of data_processing/); make
# sure it's importable regardless of how this script is invoked.
sys.path.insert(0, str(Path(__file__).parent.parent))

from vector_store import get_chroma_collection, keyword_boosted_query, replace_chunks_for
from rerank import rerank

DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FUNDS_JSON = PROCESSED_DIR / "fund_factsheets_structured.json"
COLLECTION_NAME = "fund_factsheets"


# --------------------------------------------------------------------------
# Page tracking: the narrative fields here come from ingest_factsheet.py's
# VLM extraction (reading page *images*, not the PDF text layer), so no page
# number survives into fund_factsheets_structured.json. Recovering one here
# is pure Python, no LLM call: match each field's (near-verbatim) text
# against each page's own extracted text, the same near-verbatim assumption
# ingest_factsheet.py's own grounding_check already relies on.
# --------------------------------------------------------------------------

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def page_texts_for(source_file: str) -> list[str]:
    reader = PdfReader(str(DATA_DIR / source_file))
    return [_normalize(p.extract_text() or "") for p in reader.pages]


def page_for_snippets(page_texts: list[str], snippets: list[str]) -> str | None:
    """Best-effort page locator for a field's text against a fact sheet's
    per-page text. Checks each snippet (a bullet, or the field's whole text
    as a single-item list) independently and votes by page, since a
    multi-bullet field's bullets are reliably near-verbatim individually even
    when the field as a whole doesn't appear as one contiguous substring.
    Returns "p.N" if every matched snippet lands on the same page, "pp.N-M"
    if they split across pages, or None if nothing matched (can't tell -
    the caller falls back to citing the whole document)."""
    votes = []
    for snippet in snippets:
        norm = _normalize(snippet)
        if not norm:
            continue
        for i, pt in enumerate(page_texts):
            if norm in pt:
                votes.append(i + 1)
                break
    if not votes:
        return None
    if min(votes) == max(votes):
        return f"p.{votes[0]}"
    return f"pp.{min(votes)}-{max(votes)}"


def sections_for_fund(fund: dict) -> list[dict]:
    """Every chunk's embedded text is prefixed with the fund name. Found by
    testing: a query that explicitly named a fund still only won its own
    who_is_this_for chunk by a 0.02 margin over 3 unrelated funds' chunks
    (0.66 vs 0.62-0.64), because the embedded text itself never mentions
    which fund it's about - only the metadata does, and Chroma's embedding
    function only sees `documents`, not `metadatas`. Prepending the fund
    name (a standard "contextual retrieval" technique) puts that identity
    signal into the vector itself.

    Each section also carries `snippets`: the field's original (un-prefixed)
    text, matched against the PDF's per-page text by ingest_fund() to recover
    a page number - see page_for_snippets."""
    fund_name = fund["fund_name"]

    sections = [{
        "section": "objective",
        "text": f"{fund_name} - objective:\n{fund['fund_objective_or_product_description']}",
        "snippets": [fund["fund_objective_or_product_description"]],
    }]

    if fund["who_is_this_for"]:
        sections.append({
            "section": "who_is_this_for",
            "text": f"{fund_name} - who is this designed for?\n" + "\n".join(f"- {b}" for b in fund["who_is_this_for"]),
            "snippets": fund["who_is_this_for"],
        })

    if fund["key_risks"]:
        sections.append({
            "section": "key_risks",
            "text": f"{fund_name} - key risks:\n" + "\n".join(f"- {b}" for b in fund["key_risks"]),
            "snippets": fund["key_risks"],
        })

    for i, item in enumerate(fund.get("extra_information") or []):
        sections.append({"section": f"extra_information_{i}", "text": f"{fund_name}: {item}", "snippets": [item]})

    # key_facts rows are also in the `fund_key_facts` SQL table (exact lookup
    # once you know the fund), but weren't searchable at all here - a cross-fund
    # query like "which fund has a management fee under 1%" had nothing to
    # match against without already knowing which fund to ask about. One chunk
    # per row makes each fact independently retrievable by semantic search.
    for i, kf in enumerate(fund.get("key_facts") or []):
        sections.append({
            "section": f"key_fact_{i}",
            "text": f"{fund_name} - {kf['label']}: {kf['value']}",
            "snippets": [f"{kf['label']}", kf["value"]],
        })

    return sections


def ingest_fund(fund: dict, collection) -> dict:
    fund_name = fund["fund_name"]
    sections = sections_for_fund(fund)
    page_texts = page_texts_for(fund["source_file"])

    ids, docs, metadatas = [], [], []
    for s in sections:
        ids.append(f"{fund_name}::{s['section']}")
        docs.append(s["text"])
        metadatas.append({
            "page": page_for_snippets(page_texts, s["snippets"]) or "",
            "fund_name": fund_name,
            "document_code": fund["document_code"] or "",
            "source_file": fund["source_file"],
            "section": s["section"],
            "summary_risk_indicator": fund["summary_risk_indicator"] or -1,
        })

    n_replaced = replace_chunks_for(collection, "fund_name", fund_name, ids, docs, metadatas)
    return {"fund_name": fund_name, "n_chunks": len(sections), "replaced_existing_chunks": n_replaced}


def query_fund_factsheets(question: str, n_results: int = 3, fund_name: str | None = None) -> list[dict]:
    """See vector_store.keyword_boosted_query for why this isn't plain vector
    search - found by testing: "which fund has a minimum investment of USD
    250,000" didn't return the APEX note (the fund that actually has that
    minimum) in the top 4 at all, returning four *other* funds' minimum-
    investment rows instead - embedding similarity alone doesn't reliably
    match on a specific number."""
    collection = get_chroma_collection(COLLECTION_NAME)
    where = {"fund_name": fund_name} if fund_name else None
    # Keyword-boosted vector search into a wider pool, then cross-encoder
    # re-rank down to n_results (see rerank.py). The numeric-token boost
    # keeps exact figures like "USD 250,000" high; the cross-encoder
    # re-orders by joint query-chunk relevance.
    candidates = keyword_boosted_query(collection, question, n_results=n_results, where=where)

    out = []
    for c in candidates:
        meta = c["meta"]
        out.append({
            "fund_name": meta["fund_name"],
            "section": meta["section"],
            "summary_risk_indicator": meta["summary_risk_indicator"],
            "page": meta.get("page") or None,
            "similarity": c["similarity"],
            "text": c["doc"],
        })
    return rerank(question, out, keep=n_results)


def main() -> int:
    if not FUNDS_JSON.exists():
        print(f"error: {FUNDS_JSON} not found - run ingest_factsheet.py / fund_factsheet_ingestion.ipynb first")
        return 1

    with open(FUNDS_JSON, encoding="utf-8") as f:
        funds = json.load(f)

    collection = get_chroma_collection(COLLECTION_NAME)
    for fund in funds:
        r = ingest_fund(fund, collection)
        replaced = f", replaced {r['replaced_existing_chunks']} old chunks" if r["replaced_existing_chunks"] else ""
        print(f"{r['fund_name']}: {r['n_chunks']} chunks{replaced}")

    print(f"\ncollection '{COLLECTION_NAME}' now has {collection.count()} chunks total")
    return 0


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--query":
        results = query_fund_factsheets(sys.argv[2], n_results=int(sys.argv[3]) if len(sys.argv) > 3 else 3)
        for r in results:
            print(f"[{r['similarity']}] {r['fund_name']} (SRI {r['summary_risk_indicator']}) — {r['section']}")
            print(f"  {r['text'][:200].replace(chr(10), ' ')}...\n")
        raise SystemExit(0)

    raise SystemExit(main())
