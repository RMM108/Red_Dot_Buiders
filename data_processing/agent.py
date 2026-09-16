"""Agent orchestrator: wires 6 tools (the 5 from PLAN.md §4.2, plus
query_portfolio_exposure for cross-client compliance screening) into a
tool-calling loop over OpenAI's Responses API (PLAN.md §5 steps 6-7).

Design for citations + abstention (§4.2 points 4-5, §4.5's "never re-derived
by the LLM"): every tool call appends evidence entries to an EvidencePool,
each tagged with a ref_id and a locator string built from the tool's own
metadata (source_file, section, client_id, date, ...) - never written by the
model. The model's final answer is a structured AgentAnswer that cites by
ref_id only; this module resolves each ref_id back to its locator, so the
citation text the user sees is always pulled from real metadata, and an
invalid ref_id (one the model didn't actually receive) is caught rather than
silently trusted.

The model is instructed to set abstained=True (with a reason) when evidence
is missing, conflicting, or - per the CL011 case in PLAN.md - when a policy's
scope is genuinely ambiguous as applied; in that last case the right answer
is to surface the ambiguity, not force a yes/no.

Usage:
    python agent.py "Is Robert Chua's structured note holding suitable given his risk profile?"
"""

from __future__ import annotations

import json
import re
import sys
from typing import Optional

from dotenv import load_dotenv
from langsmith import traceable
from pydantic import BaseModel

import db
from tracing import traced_openai_client
from entity_crosswalk import resolve_client
from ingest_ack_forms import query_ack_forms
from ingest_call_notes import query_call_notes
from ingest_clients import normalize_product_name
from ingest_complaints import query_complaints
from ingest_correspondence import query_correspondence
from ingest_fund_vectors import query_fund_factsheets
from ingest_policies import query_policy
from persona import PERSONA_DISPATCH, PERSONA_TOOLS

load_dotenv()

MODEL = "gpt-5.5"
MAX_TOOL_TURNS = 8
# Reasoning-tier models (gpt-5.x, o-series) reject an explicit temperature.
SUPPORTS_TEMPERATURE = not (MODEL.startswith("gpt-5") or MODEL.startswith("o1")
                             or MODEL.startswith("o3") or MODEL.startswith("o4"))

REWRITE_MODEL = "gpt-5.4-mini"

REWRITE_SYSTEM_PROMPT = """Rewrite the user's latest question into a standalone question that makes \
sense with no prior context. Resolve pronouns and references (e.g. "him", "her", "that client", "the \
same fund", "it") into the explicit name, client ID, or product they refer to, using the conversation \
history. Do not answer the question. Do not change its meaning or add information the conversation \
doesn't support. If the question is already standalone, return it unchanged. Return ONLY the rewritten \
question, with no explanation or quotation marks."""

SYSTEM_PROMPT = """You are a compliance-aware AI copilot for relationship managers and compliance \
staff at a wealth management firm. Answer client-specific questions using ONLY the tools provided - \
never rely on outside knowledge about specific clients, products, or policies.

Rules:
- Call tools to gather evidence before answering. Use as many tool calls as needed to cover every \
part of a multi-part question (e.g. a question about a client's holding may need query_client_db, \
get_fund_factsheet, AND get_policy_section together).
- Cite every factual claim by the ref_id(s) shown in the tool results. Never write a citation or \
locator yourself - only use ref_ids you actually received from a tool.
- Tool results show each source's text as numbered paragraphs, e.g. "[2] ...". When a specific \
paragraph is what actually supports a claim, set that citation's paragraph to that exact number so \
the user can jump straight to it. Only use a number you actually saw printed next to that ref_id's \
text - never estimate or invent one. Leave paragraph unset when a source has no numbering shown (a \
single short paragraph) or when the claim draws on the source as a whole rather than one specific part.
- If get_fund_factsheet returns an error (no fact sheet on file), do not guess the product's risk \
rating or terms from its name - report that the fact sheet is missing.
- If the question references a specific conversation, discussion, call, or email (e.g. "as of his \
July discussion with his RM", "following her complaint"), call search_documents to find it before \
answering - the narrative sources (call notes, correspondence, complaints) often state a figure or \
decision directly (e.g. an already-calculated remittance headroom), and that stated figure must be \
used as-is rather than re-derived or estimated from unrelated structured records like the \
transaction ledger, which can use a different basis and give a wrong answer.
- Before applying any policy rule, check who it explicitly says it covers (e.g. a rule stated for \
"a Retail Investor's liquid net worth" does not plainly cover an Accredited Investor just because \
both are complex-product buyers - read the rule's stated scope literally, don't assume it extends \
further). If the client's investor classification differs from the rule's stated scope, that is a \
genuine ambiguity, not a basis for confidently asserting either a breach or a clean bill of health. \
In that situation, set abstained=true, state both readings (the literal-scope reading and the \
precautionary reading), and say explicitly that this needs a human compliance judgment call - do \
not pick one reading and present it as the firm's policy position. Mechanical check before you \
finalize: if your own answer text contains a hedge like "however", "does not breach... but/although", \
"may require review", or "judgment call", that hedge IS the ambiguity - set abstained=true to match \
it. abstained=false is reserved for answers you would state without any such hedge. When \
abstained=true, the answer text itself must not open with a flat "does breach" / "does not breach" \
assertion that the rest of the answer then contradicts - lead with the ambiguity itself (e.g. "Whether \
this breaches policy is ambiguous:") so the prose and the abstained flag agree.
- Set abstained=true (with abstention_reason) whenever the tools don't return enough evidence to \
answer confidently, or when retrieved evidence conflicts. A partial, calibrated answer with an \
explicit gap is correct; a confident answer built on missing evidence is not.
- If a client name is ambiguous (query_client_db returns candidates), ask which client was meant \
instead of guessing - set abstained=true and list the candidates in abstention_reason.
- Persona tools (lookup_client_persona, generate_client_persona, generate_personas) summarize \
observed client records and comparable-client portfolio patterns. Use lookup_client_persona to \
retrieve one client's stored persona by exact client_id, and generate_client_persona to surface \
comparable clients for a set of characteristics. Treat persona output as discussion context and \
review prompts only - it is NOT an investment recommendation and must never be presented as one. \
If generate_client_persona reports that no comparable clients were found, treat that as missing \
evidence and set abstained=true rather than inventing a recommendation. Persona output does not \
replace policy evaluation: still call get_policy_section and get_fund_factsheet before any \
suitability conclusion.
"""


class Citation(BaseModel):
    ref_id: str
    paragraph: Optional[int] = None


class AgentAnswer(BaseModel):
    answer: str
    citations: list[Citation]
    abstained: bool
    abstention_reason: Optional[str] = None


# --------------------------------------------------------------------------
# Evidence pool: every tool result becomes one or more ref_id -> locator entries
# --------------------------------------------------------------------------

class EvidencePool:
    def __init__(self):
        self.entries: dict[str, dict] = {}

    def add(self, ref_id: str, locator: str, text: str) -> dict:
        entry = {"ref_id": ref_id, "locator": locator, "text": text}
        self.entries[ref_id] = entry
        return entry

    def resolve(self, ref_id: str) -> Optional[dict]:
        return self.entries.get(ref_id)


# --------------------------------------------------------------------------
# Tool implementations - each returns (evidence_entries, text_for_model)
# --------------------------------------------------------------------------

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
FIRM_NAME = "Meridian Peak Wealth Partners"


def _apa_citation(locator: str, quoted_text: Optional[str]) -> str:
    """APA-style in-text citation built only from data already in `locator` -
    these are internal firm records rather than published works, so there's
    no separate "author"/"title" metadata to draw on; the firm name stands in
    for the author, and the locator's own document/section/page/paragraph
    detail (already validated, never model-authored) fills the rest, the
    way a page number normally would. `quoted_text` is the model-cited
    paragraph's real text, so a claim is shown as an actual excerpt, not a
    paraphrase - never freehand quoted by the model itself."""
    year_match = _YEAR_RE.search(locator)
    year = year_match.group(0) if year_match else "2026"
    attribution = f"{FIRM_NAME}, {year}, {locator}"
    return f'"{quoted_text}" ({attribution})' if quoted_text else f"({attribution})"


def _with_page(locator: str, page: Optional[str]) -> str:
    """Append a page locator (e.g. "p.4" or "pp.4-5") to a citation string
    when the ingest pipeline recovered one for this chunk - see
    chunking.page_range_for_span (regex-chunked PDFs) and
    ingest_fund_vectors.page_for_snippets (VLM-extracted fact sheet fields)."""
    return f"{locator}, {page}" if page else locator


_BULLET_LINE_RE = re.compile(r"^\s*[\x7f•*]\s+|^\s*-\s+")
_LABEL_LINE_RE = re.compile(r"^\s*[A-Za-z][\w /()&]{1,40}:\s")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _is_structural_anchor(line: str) -> bool:
    """A line that reliably starts a new citable unit on its own: a bullet
    ("- Product Name...", the policy PDFs' "\x7f " glyph bullets) or a
    "Label: value" field (e.g. "Date: 2026-01-20", "FLAG: ..."). Distinct
    from a plain wrapped continuation line, which starts mid-sentence."""
    return bool(_BULLET_LINE_RE.match(line) or _LABEL_LINE_RE.match(line))


def split_paragraphs(text: str) -> list[str]:
    """Split a piece of evidence text into citable paragraphs. Used both to
    number paragraphs for the model to cite by (format_evidence_text) and,
    later, to check a claimed paragraph number is real (run_agent's citation
    resolution) - the same function drives both sides so the numbering
    always matches.

    Three cases, in order:
    1. Blank-line-separated paragraphs, where the source actually has them.
    2. A source with structural anchors (bullets / "Label: value" rows, e.g.
       a policy section's numbered sub-points or a call note's "FLAG:" /
       "Summary:" fields): group each anchor with any wrapped continuation
       lines that follow it, so e.g. the policy's "20%" bullet stays one
       clean unit instead of being cut at the PDF's line-wrap point.
    3. Plain prose with no structural markers at all (a complaint letter's
       body): pypdf's extract_text() wraps lines at the page width, not at
       sentence boundaries - "I do\\nnot recall" is one sentence split
       mid-word, not two paragraphs. Rejoin everything into one blob and
       split on sentence boundaries instead, the finest unit that's still
       safe to cite without cutting a claim in half. (Known limitation: a
       short abbreviation like "Pte." before a capitalized word can trigger
       a spurious split - acceptable here since it mainly affects
       letterhead/header text, not the substantive sentences that actually
       get cited.)"""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text.strip()) if b.strip()]
    if len(blocks) > 1:
        return blocks

    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    if not lines:
        return []

    anchor_fraction = sum(1 for l in lines if _is_structural_anchor(l)) / len(lines)
    if anchor_fraction < 0.15:
        rejoined = re.sub(r"\s+", " ", text).strip()
        sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(rejoined) if s.strip()]
        return sentences or [rejoined]

    groups: list[str] = []
    buf = ""
    for line in lines:
        if not buf or _is_structural_anchor(line):
            if buf:
                groups.append(buf)
            # Strip a pure bullet glyph (not a "Label:" prefix, which is
            # real content) so a quoted citation reads as clean prose
            # instead of carrying the PDF's raw bullet character along.
            buf = _BULLET_LINE_RE.sub("", line, count=1)
        else:
            buf = f"{buf} {line}"
    if buf:
        groups.append(buf)
    return groups


def format_evidence_text(text: str) -> str:
    """Render `text` as numbered paragraphs (e.g. "[2] ...") for the model to
    optionally cite a specific paragraph from - see Citation.paragraph and
    the SYSTEM_PROMPT rule on when to set it. A single-paragraph source is
    left unnumbered since there's nothing to disambiguate."""
    paragraphs = split_paragraphs(text)
    if len(paragraphs) <= 1:
        return text
    return "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(paragraphs))


def _resolve_client_id(client_id_or_name: str) -> dict:
    s = client_id_or_name.strip().upper()
    if s.startswith("CL") and s[2:].isdigit():
        conn = db.get_connection()
        row = conn.execute("SELECT client_id FROM clients WHERE client_id = ?", (s,)).fetchone()
        conn.close()
        return {"client_id": s} if row else {"found": False, "query": client_id_or_name}
    return resolve_client(client_id_or_name)


@traceable(name="tool_query_client_db", run_type="tool")
def tool_query_client_db(pool: EvidencePool, client_id_or_name: str) -> str:
    resolution = _resolve_client_id(client_id_or_name)
    if resolution.get("ambiguous"):
        return json.dumps({"error": "ambiguous client name", **resolution})
    if resolution.get("found") is False:
        return json.dumps({"error": f"no client found matching {client_id_or_name!r}"})

    client_id = resolution["client_id"]
    conn = db.get_connection()
    client = conn.execute("SELECT * FROM clients WHERE client_id = ?", (client_id,)).fetchone()
    holdings = conn.execute("SELECT * FROM holdings WHERE client_id = ?", (client_id,)).fetchall()
    conn.close()
    if not client:
        return json.dumps({"error": f"no client found matching {client_id_or_name!r}"})

    client = dict(client)
    holdings_text = "\n".join(
        f"  - {h['product_name']} ({h['asset_class']}): {h['allocation_pct']}% / {h['value_sgd']} SGD"
        for h in holdings
    )
    text = (
        f"Client {client_id} ({client['name']}): residency={client['residency_country']}, "
        f"investor_status={client['investor_status']}, risk_profile={client['risk_profile']}"
        + (f" [note: {client['risk_profile_note']}]" if client["risk_profile_note"] else "")
        + f", risk_score={client['risk_score_1_to_10']}/10, aum_sgd={client['aum_sgd']}, "
        f"kyc_status={client['kyc_status']} (refreshed {client['kyc_refresh_date']})"
        + (f" [note: {client['kyc_note']}]" if client["kyc_note"] else "")
        + f", pep_status={client['pep_status']}"
        + (f" [note: {client['pep_note']}]" if client["pep_note"] else "")
        + f", suitability_flag={client['suitability_flag']}\nHoldings:\n{holdings_text}"
    )
    ref_id = f"client_{client_id}"
    pool.add(ref_id, f"clients_portfolio.json ({client_id})", text)
    return json.dumps({"ref_id": ref_id, "content": format_evidence_text(text)})


@traceable(name="tool_query_transactions", run_type="tool")
def tool_query_transactions(
    pool: EvidencePool,
    client_id_or_name: str,
    status: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> str:
    resolution = _resolve_client_id(client_id_or_name)
    if resolution.get("ambiguous") or resolution.get("found") is False:
        return json.dumps({"error": f"could not resolve client {client_id_or_name!r}", **resolution})

    client_id = resolution["client_id"]
    conn = db.get_connection()
    query = "SELECT * FROM transactions WHERE client_id = ?"
    params: list = [client_id]
    if status:
        query += " AND status = ?"
        params.append(status)
    if date_from:
        query += " AND date >= ?"
        params.append(date_from)
    if date_to:
        query += " AND date <= ?"
        params.append(date_to)
    rows = [dict(r) for r in conn.execute(query, params)]
    conn.close()

    if not rows:
        return json.dumps({"error": f"no transactions found for {client_id}" + (f" with status={status}" if status else "")})

    text = "\n".join(
        f"  - {r['transaction_id']} {r['date']} {r['transaction_type']}: {r['product_name']} "
        f"({r['amount']} {r['currency']}, {r['status']})" + (f" - {r['notes']}" if r["notes"] else "")
        for r in rows
    )
    ref_id = f"transactions_{client_id}"
    locator = f"transactions.csv ({client_id}"
    if date_from:
        locator += f", from={date_from}"
    if date_to:
        locator += f", to={date_to}"
    pool.add(ref_id, locator + ")", text)
    return json.dumps({"ref_id": ref_id, "content": format_evidence_text(text)})


@traceable(name="tool_query_portfolio_exposure", run_type="tool")
def tool_query_portfolio_exposure(
    pool: EvidencePool,
    product_type: str = "Complex Product",
    threshold_pct: float = 20.0,
    investor_status: Optional[str] = None,
) -> str:
    """Find holdings whose allocation exceeds a portfolio concentration threshold -
    the compliance/supervisor cross-client screening use case (PRD.md user story 4),
    e.g. "which clients hold Complex Products above the 20% concentration guideline?"."""

    conn = db.get_connection()
    query = """
        SELECT c.client_id, c.name, c.investor_status, h.product_name,
               h.asset_class, h.allocation_pct
        FROM holdings AS h
        JOIN clients AS c ON c.client_id = h.client_id
        WHERE h.allocation_pct > ?
          AND (
              lower(h.asset_class) LIKE '%complex%'
              OR lower(h.asset_class) LIKE '%structured%'
              OR lower(h.asset_class) LIKE '%specified investment%'
          )
    """
    params: list[object] = [threshold_pct]
    if investor_status:
        query += " AND lower(c.investor_status) = lower(?)"
        params.append(investor_status)
    rows = [dict(row) for row in conn.execute(query, params)]
    conn.close()

    if not rows:
        return json.dumps({
            "content": (
                f"No {product_type} holdings exceed {threshold_pct:.2f}%"
                + (f" for investor_status={investor_status}" if investor_status else "")
                + "."
            )
        })

    text = "\n".join(
        f"  - {row['client_id']} ({row['name']}), investor_status={row['investor_status']}: "
        f"{row['product_name']} [{row['asset_class']}] at {row['allocation_pct']}%"
        for row in rows
    )
    ref_id = f"portfolio_exposure_{threshold_pct:g}"
    pool.add(ref_id, "clients_portfolio.csv (complex-product concentration)", text)
    return json.dumps({
        "ref_id": ref_id,
        "content": (
            f"{product_type} holdings above {threshold_pct:.2f}% allocation:\n{format_evidence_text(text)}"
        ),
    })


@traceable(name="tool_get_fund_factsheet", run_type="tool")
def tool_get_fund_factsheet(pool: EvidencePool, product_name: str) -> str:
    conn = db.get_connection()
    norm = normalize_product_name(product_name)
    all_funds = conn.execute("SELECT fund_name, source_file FROM funds").fetchall()
    match = next((f for f in all_funds if normalize_product_name(f["fund_name"]) == norm), None)

    if not match:
        conn.close()
        return json.dumps({"error": f"no fact sheet on file for {product_name!r} - do not infer its SRI or terms"})

    fund_name, source_file = match["fund_name"], match["source_file"]
    fund = dict(conn.execute("SELECT * FROM funds WHERE fund_name = ?", (fund_name,)).fetchone())
    key_facts = conn.execute("SELECT label, value FROM fund_key_facts WHERE fund_name = ?", (fund_name,)).fetchall()
    allocation = conn.execute("SELECT asset_type, pct_of_fund FROM fund_asset_allocation WHERE fund_name = ?", (fund_name,)).fetchall()
    conn.close()

    facts_text = "\n".join(f"  {r['label']}: {r['value']}" for r in key_facts)
    alloc_text = "\n".join(f"  {r['asset_type']}: {r['pct_of_fund']}" for r in allocation) or "  (none)"
    sql_text = (
        f"{fund_name} - SRI {fund['summary_risk_indicator']}/7 ({fund['summary_risk_indicator_label']}), "
        f"minimum investment {fund['minimum_investment']}, base currency {fund['base_currency']}, "
        f"as of {fund['as_of_date']}.\nKey facts:\n{facts_text}\nAsset allocation:\n{alloc_text}"
    )
    sql_ref_id = f"factsheet_sql_{fund_name}"
    pool.add(sql_ref_id, f"{source_file} (structured fields)", sql_text)

    contents = [f"[{sql_ref_id}] {format_evidence_text(sql_text)}"]
    for r in query_fund_factsheets(product_name, n_results=3, fund_name=fund_name):
        vec_ref_id = f"factsheet_vec_{fund_name}_{r['section']}"
        pool.add(vec_ref_id, _with_page(f"{source_file} ({r['section']})", r.get("page")), r["text"])
        contents.append(f"[{vec_ref_id}] {format_evidence_text(r['text'])}")

    return json.dumps({"content": "\n\n".join(contents)})


@traceable(name="tool_get_policy_section", run_type="tool")
def tool_get_policy_section(pool: EvidencePool, query: str, document_code: Optional[str] = None) -> str:
    results = query_policy(query, n_results=3, document_code=document_code)
    if not results:
        return json.dumps({"error": "no matching policy section found"})

    contents = []
    for r in results:
        section_num = r["section"].split(":")[0].replace("Section ", "").strip() if "Section" in r["section"] else "x"
        ref_id = f"policy_{r['document_code']}_s{section_num}"
        pool.add(ref_id, _with_page(f"{r['policy_name']} {r['section']}", r.get("page")), r["text"])
        contents.append(f"[{ref_id}] {r['policy_name']} {r['section']}:\n{format_evidence_text(r['text'])}")

    return json.dumps({"content": "\n\n".join(contents)})


@traceable(name="tool_search_documents", run_type="tool")
def tool_search_documents(pool: EvidencePool, query: str, doc_types: Optional[list[str]] = None,
                           client_id: Optional[str] = None) -> str:
    doc_types = doc_types or ["call_notes", "complaints", "correspondence", "ack_forms", "fund_factsheets"]
    contents = []

    if "call_notes" in doc_types:
        for r in query_call_notes(query, n_results=2, client_id=client_id):
            ref_id = f"call_note_{r['client_id']}_{r['date']}"
            pool.add(ref_id, _with_page(f"rm_call_notes_log.pdf ({r['client_id']}, {r['date']})", r.get("page")), r["text"])
            contents.append(f"[{ref_id}] {format_evidence_text(r['text'])}")

    if "complaints" in doc_types:
        for r in query_complaints(query, n_results=2, client_id=client_id):
            ref_id = f"complaint_{r['complaint_ref']}"
            pool.add(ref_id, _with_page(f"client_complaint_letters.pdf ({r['complaint_ref']})", r.get("page")), r["text"])
            contents.append(f"[{ref_id}] {format_evidence_text(r['text'])}")

    if "correspondence" in doc_types:
        for r in query_correspondence(query, n_results=2, client_id=client_id):
            ref_id = f"correspondence_{r['thread_id']}"
            pool.add(ref_id, f"client_correspondence.json ({r['thread_id']}: {r['subject']})", r["text"])
            contents.append(f"[{ref_id}] {format_evidence_text(r['text'])}")

    if "ack_forms" in doc_types:
        for r in query_ack_forms(query, n_results=2, client_id=client_id):
            ref_id = f"ack_form_{r['client_id'] or 'unknown'}"
            pool.add(ref_id, _with_page(f"complex_product_risk_acknowledgement_forms.pdf ({r['client_id']})", r.get("page")), r["text"])
            contents.append(f"[{ref_id}] {format_evidence_text(r['text'])}")

    if "fund_factsheets" in doc_types:
        for r in query_fund_factsheets(query, n_results=2):
            ref_id = f"factsheet_vec_{r['fund_name']}_{r['section']}"
            pool.add(ref_id, _with_page(f"fund_factsheet ({r['fund_name']}, {r['section']})", r.get("page")), r["text"])
            contents.append(f"[{ref_id}] {format_evidence_text(r['text'])}")

    if not contents:
        return json.dumps({"error": "no matching documents found"})
    return json.dumps({"content": "\n\n".join(contents)})


TOOLS = [
    {
        "type": "function",
        "name": "query_client_db",
        "description": "Look up a client's KYC/risk profile and portfolio holdings by client_id (e.g. 'CL002') or by name (e.g. 'Robert Chua').",
        "parameters": {
            "type": "object",
            "properties": {"client_id_or_name": {"type": "string"}},
            "required": ["client_id_or_name"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "query_transactions",
        "description": "Look up a client's transaction ledger (subscriptions, redemptions, coupons, pending items) by client_id or name.",
        "parameters": {
            "type": "object",
            "properties": {
                "client_id_or_name": {"type": "string"},
                "status": {"type": ["string", "null"], "description": "Optional filter, e.g. 'Pending' or 'Settled'"},
                "date_from": {"type": ["string", "null"], "description": "Optional inclusive ISO date lower bound"},
                "date_to": {"type": ["string", "null"], "description": "Optional inclusive ISO date upper bound"},
            },
            "required": ["client_id_or_name", "status", "date_from", "date_to"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "query_portfolio_exposure",
        "description": "Find client holdings above a concentration threshold for Complex or Structured Products, optionally filtered by investor status.",
        "parameters": {
            "type": "object",
            "properties": {
                "product_type": {"type": "string", "description": "Product category, normally 'Complex Product'"},
                "threshold_pct": {"type": "number", "description": "Strict allocation threshold, e.g. 20"},
                "investor_status": {"type": ["string", "null"], "description": "Optional exact investor-status filter"},
            },
            "required": ["product_type", "threshold_pct", "investor_status"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_fund_factsheet",
        "description": "Get a fund/product's fact sheet: SRI, minimum investment, key facts, asset allocation, "
                        "objective, who it's for, and key risks. Returns an error if no fact sheet exists for "
                        "this product - report that as missing evidence, never infer the terms from the name.",
        "parameters": {
            "type": "object",
            "properties": {"product_name": {"type": "string"}},
            "required": ["product_name"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_policy_section",
        "description": "Search the firm's policy documents (KYC/onboarding policy POL-KYC-004, investment "
                        "suitability policy POL-INV-011) for the section relevant to a question.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "document_code": {"type": ["string", "null"], "description": "Optional: 'POL-KYC-004' or 'POL-INV-011' to restrict to one policy"},
            },
            "required": ["query", "document_code"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "search_documents",
        "description": "Semantic search across call notes, complaint letters, client correspondence, "
                        "risk-acknowledgement forms, and fund fact sheet narrative text.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "doc_types": {
                    "type": ["array", "null"],
                    "items": {"type": "string", "enum": ["call_notes", "complaints", "correspondence", "ack_forms", "fund_factsheets"]},
                    "description": "Optional: restrict to these document types. Omit/null to search all.",
                },
                "client_id": {"type": ["string", "null"], "description": "Optional: restrict to one client's documents"},
            },
            "required": ["query", "doc_types", "client_id"],
            "additionalProperties": False,
        },
    },
    # Persona tools (defined in persona.py) - expose them to the model.
    *PERSONA_TOOLS,
]

DISPATCH = {
    "query_client_db": tool_query_client_db,
    "query_transactions": tool_query_transactions,
    "query_portfolio_exposure": tool_query_portfolio_exposure,
    "get_fund_factsheet": tool_get_fund_factsheet,
    "get_policy_section": tool_get_policy_section,
    "search_documents": tool_search_documents,
    **PERSONA_DISPATCH,
}


# --------------------------------------------------------------------------
# Orchestrator loop
# --------------------------------------------------------------------------

@traceable(name="rewrite_query", run_type="chain")
def rewrite_query(question: str, history: Optional[list[dict]] = None) -> str:
    """Resolve conversational references in `question` using prior turns, so the
    retrieval loop below always works from a standalone question. `history` is a
    list of {"role": "user"|"assistant", "content": str} from earlier turns in
    the same conversation (most recent last); pass None/[] for a fresh question."""
    if not history:
        return question

    client = traced_openai_client()
    convo = "\n".join(f"{h['role']}: {h['content']}" for h in history[-6:])
    resp = client.responses.create(
        model=REWRITE_MODEL,
        input=[
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Conversation so far:\n{convo}\n\nLatest question: {question}"},
        ],
    )
    return resp.output_text.strip() or question


MAX_HISTORY_MESSAGES = 20  # ~10 exchanges of real conversation context fed to the main loop


@traceable(name="run_agent", run_type="chain")
def run_agent(question: str, history: Optional[list[dict]] = None, pool: Optional[EvidencePool] = None,
              max_turns: int = MAX_TOOL_TURNS, verbose: bool = False) -> dict:
    """`history` is the real prior conversation (list of {"role", "content"}, most
    recent last) - fed to the model as-is so it can handle corrections, follow-up
    nuance, and reasoning that spans turns, not just pronoun resolution (that part
    is still handled separately by rewrite_query, which produces a clean standalone
    final question on top of this).

    `pool` carries the EvidencePool forward across turns (pass the same instance
    the caller used on the previous call) so a ref_id cited in an earlier turn
    stays resolvable - e.g. the model can cite a fact it already retrieved instead
    of being forced to re-call a tool just to regenerate a valid citation for it.
    Pass None for a single-shot question with no conversation to carry."""
    rewritten_question = rewrite_query(question, history)
    if verbose and rewritten_question != question:
        print(f"  rewritten query: {rewritten_question!r}")

    client = traced_openai_client()
    pool = pool if pool is not None else EvidencePool()
    input_list: list = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in (history or [])[-MAX_HISTORY_MESSAGES:]:
        input_list.append({"role": h["role"], "content": h["content"]})
    input_list.append({"role": "user", "content": rewritten_question})
    tool_call_log = []
    n_evidence_before = len(pool.entries)

    for _ in range(max_turns):
        kwargs = {"temperature": 0} if SUPPORTS_TEMPERATURE else {}
        resp = client.responses.parse(model=MODEL, input=input_list, tools=TOOLS, text_format=AgentAnswer, **kwargs)
        input_list += resp.output

        function_calls = [o for o in resp.output if o.type == "function_call"]
        if not function_calls:
            parsed: AgentAnswer = resp.output_parsed
            break

        for call in function_calls:
            args = json.loads(call.arguments)
            args = {k: v for k, v in args.items() if v is not None}
            fn = DISPATCH[call.name]
            if verbose:
                print(f"  tool call: {call.name}({args})")
            result = fn(pool, **args)
            tool_call_log.append({"tool": call.name, "args": args})
            input_list.append({"type": "function_call_output", "call_id": call.call_id, "output": result})
    else:
        return {
            "answer": "Could not complete within the tool-call budget.",
            "citations": [], "abstained": True,
            "abstention_reason": f"exceeded {max_turns} tool-call turns",
            "tool_calls": tool_call_log, "n_evidence_retrieved": len(pool.entries) - n_evidence_before,
            "n_evidence_total": len(pool.entries),
            "original_question": question, "rewritten_question": rewritten_question,
        }

    resolved_citations = []
    for c in parsed.citations:
        entry = pool.resolve(c.ref_id)
        if entry:
            locator = entry["locator"]
            quoted_text = None
            # A paragraph number is only meaningful (and only ever shown to
            # the model) when the source had more than one; validate against
            # the real count rather than trusting the model's number as-is -
            # same "never re-derived by the LLM" principle as ref_id itself.
            if c.paragraph is not None:
                paragraphs = split_paragraphs(entry["text"])
                if len(paragraphs) > 1 and 1 <= c.paragraph <= len(paragraphs):
                    locator = f"{locator}, para {c.paragraph}"
                    quoted_text = paragraphs[c.paragraph - 1]
            resolved_citations.append({
                "ref_id": c.ref_id,
                "locator": locator,
                "text": quoted_text or entry["text"],
                "apa": _apa_citation(locator, quoted_text),
            })
        else:
            resolved_citations.append({
                "ref_id": c.ref_id,
                "locator": "INVALID ref_id (model cited a source it never received)",
                "text": None,
                "apa": None,
            })

    return {
        "answer": parsed.answer,
        "citations": resolved_citations,
        "abstained": parsed.abstained,
        "abstention_reason": parsed.abstention_reason,
        "tool_calls": tool_call_log,
        "n_evidence_retrieved": len(pool.entries) - n_evidence_before,
        "n_evidence_total": len(pool.entries),
        "original_question": question,
        "rewritten_question": rewritten_question,
    }


def print_result(result: dict) -> None:
    if result.get("rewritten_question") and result["rewritten_question"] != result.get("original_question"):
        print(f"\nInterpreted as: {result['rewritten_question']}")
    print(f"\nAnswer: {result['answer']}")
    print(f"\nAbstained: {result['abstained']}" + (f" ({result['abstention_reason']})" if result["abstention_reason"] else ""))
    print(f"\nCitations ({len(result['citations'])}):")
    for c in result["citations"]:
        print(f"  [{c['ref_id']}] {c.get('apa') or c['locator']}")
    print(f"\nTool calls: {[t['tool'] for t in result['tool_calls']]}")
    print(f"Evidence retrieved: {result['n_evidence_retrieved']}")


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What is the minimum investment for the APAC Stable Income Money Market Fund?"
    print(f"Question: {q}")
    result = run_agent(q, verbose=True)
    print_result(result)
