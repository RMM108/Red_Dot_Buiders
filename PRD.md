# Product Requirements Document

## Meridian Peak Wealth Copilot — Agentic RAG Assistant for Relationship Managers

*(Dataset is branded "Meridian Peak Wealth Partners," a fictitious APAC private wealth division.)*

**Status:** Consolidated PRD — reconciles the original product vision (formerly this file) with the
system actually implemented and tested (`PLAN.md`, `agent.py`, `db.py`, `vector_store.py`,
`ingest_*.py`, `run_evaluation.py`). The **built system is the base**; scope below reflects what
exists and runs today, with the earlier, larger architecture carried forward as a deliberate
Phase 2 rather than re-scoped into the MVP.

---

## 1. Executive Summary

Relationship Managers (RMs) at a private wealth management division spend significant time before
every client conversation manually cross-referencing scattered sources: portfolio holdings, fund
fact sheets, internal policy, past call notes, email correspondence, complaint records, and
risk-acknowledgement forms — just to answer questions like "is this holding still suitable?" or
"what's actually happened on this account since the client asked to de-risk?"

This product is an **agentic RAG copilot** that unifies two knowledge domains — (1) generic
product/policy knowledge (fund fact sheets, risk-acknowledgement forms, suitability/KYC policy) and
(2) client-specific context (portfolios, transactions, call notes, correspondence, complaints) —
behind a single tool-calling agent. The agent plans retrieval steps across a SQL store (exact
lookups, filters, aggregations) and a vector store (semantic search over narrative content), assesses
whether it has enough grounded evidence, synthesizes a cited answer, and **abstains explicitly** when
evidence is missing, conflicting, or a policy's scope is genuinely ambiguous — rather than guessing.

**MVP goal (achieved):** ingest the full provided dataset (15 client profiles, 10 fund fact sheets, 2
policy documents, a 55-row transaction ledger, RM call notes, email correspondence, complaint
letters, and risk-acknowledgement forms), correctly answer the 5 golden benchmark questions with
grounded, cited, calibrated answers, and abstain cleanly on the deliberately ambiguous case — measured
via an LLM-judge evaluation harness against the provided golden dataset workbook. **Current result:
4 of 5 golden questions factually correct**, including the ambiguous-policy case, with known,
documented gaps (§5, §11) rather than unverified claims of completeness.

---

## 2. Mission

**Mission statement:** Give every relationship manager instant, trustworthy, cited answers about
their clients' portfolios and the products available to them — so client conversations are better
prepared, suitability risks are caught before they become complaints, and no answer is ever more
confident than the evidence supports.

**Core principles:**

1. **Grounded over fluent** — every answer must trace to retrieved evidence; a wrong answer is worse
   than no answer.
2. **Cite everything** — every claim carries a `ref_id` resolved back to real source metadata
   (document, section/date/row), never freehand-written by the model.
3. **Abstain, don't guess** — ambiguous policy or missing evidence produces a calibrated "this is
   unclear" / "I don't have enough evidence," never a fabricated resolution.
4. **Right store for the shape of the data** — structured/numeric facts (SRI vs. risk score,
   concentration %, presence/absence of a signed form) are computed via SQL, not inferred by the LLM
   from retrieved text; narrative facts are retrieved via vector search.
5. **Build the simplest thing that is honestly measured** — ship a working, evaluated system on the
   actual dataset before adding orchestration-framework or infrastructure complexity that the current
   scale doesn't yet require (see §13 for what that complexity buys later).

---

## 3. Target Users

**Primary — Relationship Managers.** Non-technical, time-pressured, need answers in the minutes
before a client call. Need trustworthy, compliance-aware synthesis — not a raw document dump they
still have to interpret.

**Secondary — Compliance / Supervisor users.** Need audit visibility into what was asked and answered,
and the ability to run cross-client aggregate queries (e.g. "which clients hold Complex Products above
the 20% concentration guideline?"). *Not yet supported by a dedicated tool — see §4 Out of Scope.*

**Technical comfort:** Low-to-moderate. The interface should eventually be a plain chat surface; today
it is a CLI (`python agent.py "<question>"`) suitable for the developer/evaluator, not yet an RM.

**Key pain points observed in the dataset:**
- Manually cross-referencing 4–5 documents (portfolio, call note, email thread, complaint letter,
  policy) to answer one suitability question.
- Risk of missing that a Complex Product sale lacks a signed risk acknowledgement (an *absence*,
  easy to miss when reading documents linearly).
- Genuinely ambiguous policy wording (does the 20% concentration cap apply to Accredited Investors?)
  that requires presenting both readings, not a false-confident verdict.

---

## 4. MVP Scope

### In Scope ✅ (built and tested)

**Core Functionality**
- ✅ Deterministic ingestion/cleaning pipeline per source type (PDF, CSV, JSON) — one script per
  document type (`ingest_clients.py`, `ingest_factsheet.py`, `ingest_policies.py`,
  `ingest_call_notes.py`, `ingest_complaints.py`, `ingest_ack_forms.py`, `ingest_correspondence.py`,
  `ingest_fund_vectors.py`), applying the specific cleaning rules found by inspecting the actual data
  (denormalized fields, embedded free text in categorical columns, overloaded `product_name`, missing
  fact sheet for one product, multi-record PDFs needing header-based segmentation — full detail in
  `PLAN.md` §3).
- ✅ Two backing stores, split by field type, not by document: SQLite for exact/structured facts
  (`clients`, `holdings`, `transactions`, `funds`, `fund_key_facts`, `fund_asset_allocation`,
  `product_crosswalk`), Chroma for narrative/prose content (`policies`, `fund_factsheets`,
  `call_notes`, `complaints`, `correspondence`, `ack_forms` collections).
- ✅ Entity resolution (`entity_crosswalk.py`) — client name → `client_id`, returning an explicit
  `ambiguous` result (with candidates) rather than guessing when a surname match is not unique.
- ✅ Product crosswalk (`ingest_clients.py`) — normalizes fund-name punctuation variants (em dash vs.
  hyphen) across holdings/transactions/fact sheets, and flags the one product with no fact sheet on
  file (Singapore Government Bond Fund) so the agent abstains on SRI/fee questions about it instead of
  inferring from the name.
- ✅ Tool-calling agent orchestrator (`agent.py`, OpenAI Responses API, `gpt-4o-mini`) implementing a
  retrieve → assess sufficiency → retrieve more (or answer) loop, capped at 8 tool-call turns.
- ✅ Five typed tools (`query_client_db`, `query_transactions`, `get_fund_factsheet`,
  `get_policy_section`, `search_documents`) — see §7 for the actual signatures.
- ✅ RAG responses grounded strictly in tool-returned evidence — every tool call appends entries to an
  `EvidencePool` keyed by `ref_id`; the model's structured `AgentAnswer` cites by `ref_id` only, and
  each citation is resolved back to a real locator (source file + section/date/client) — an invalid
  `ref_id` the model never actually received is caught, not trusted.
- ✅ Explicit abstention (`abstained: bool`, `abstention_reason: str | None`) driven by system-prompt
  rules refined against observed failures (§5): missing/conflicting evidence, no fact sheet on file,
  ambiguous client name, and policy scope genuinely ambiguous as applied to the client's investor
  classification (the CL011 case) — including a mechanical self-check that flags hedge words
  ("however", "may require review") as a signal the answer must set `abstained=true`.
- ✅ Keyword-boosted policy retrieval (`ingest_policies.py`'s `query_policy`) — a numeric token
  verbatim in the query (e.g. `"20%"`) promotes the matching chunk to the front regardless of
  embedding rank, fixing an observed failure where pure vector search ranked the operative section
  last out of 14 chunks.
- ✅ Automated evaluation harness (`run_evaluation.py`) — runs all 5 golden queries through the live
  agent, judges each response against the expected answer with an LLM-judge call, and writes results
  into `data/processed/golden_dataset_for_RAG_evaluation_completed.xlsx` alongside the standard
  Context Precision / Context Recall / Completeness / Faithfulness ratio computation.

**Technical**
- ✅ SQLite structured store (`db.py`, `data/processed/wealth_management.db`)
- ✅ Chroma persistent vector store (`vector_store.py`, `./chroma_db`), `text-embedding-3-small`
  embeddings
- ✅ OpenAI Responses API tool-calling loop (`gpt-4o-mini`), Pydantic-typed structured output
- ✅ `.env`-based config (`python-dotenv`) for the OpenAI API key

### Out of Scope for MVP ❌ (not built; see §13 Future Considerations for the larger architecture already drafted for these)

- ❌ Chat UI (Streamlit is listed in `requirements.txt` but unused — the only interface today is the
  CLI / `run_agent()` function)
- ❌ HTTP API layer (no FastAPI backend; `agent.run_agent()` is called directly)
- ❌ RM identity / row-level access control (no `rm_id` concept; every tool call sees the full dataset)
- ❌ Structured audit logging of queries/answers beyond the evaluation workbook output
- ❌ Reranking stage (retrieval relies on Chroma's native similarity + the policy keyword boost, not a
  cross-encoder/reranker model)
- ❌ Full observability/tracing platform (debugging today is `verbose=True` console output of tool
  calls, not a hosted tracing tool)
- ❌ Multi-node graph orchestration framework (the agent is a single OpenAI tool-calling loop, not a
  LangGraph router/aggregator/critic graph)
- ❌ Separate deterministic "compliance rule-check" node outside the LLM (SRI-vs-risk-score and
  concentration checks are currently reasoned about by the LLM over retrieved SQL facts, not computed
  by a standalone rule engine — flagged as a known gap, not a design endorsement, in §11/§14)
- ❌ Docker Compose / containerized deployment
- ❌ Query rewriting, cross-document comparison mode, persistent multi-turn memory
- ❌ Enterprise SSO, PII redaction pipeline, data-residency controls

---

## 5. User Stories

1. **As an RM**, I want to ask "Is [Client]'s holding of [Product] suitable given their risk
   profile?" so I can catch a suitability mismatch before it becomes a complaint.
   *Example: Robert Chua (CL002) — Conservative profile holding a 7/7 SRI structured note with no
   signed risk acknowledgement on file. The agent answers correctly from `query_client_db` +
   `get_fund_factsheet` + `get_policy_section`, though it does not yet reliably also surface the call
   note / complaint letter / internal email that give the full escalation trail — a documented,
   partial gap (§11).*

2. **As an RM**, I want a plain-language summary of a fund fact sheet's risk rating and minimum
   investment before a call, so I don't have to re-read the PDF each time.
   *Verified working end-to-end via `get_fund_factsheet`, combining the SQL structured fields with the
   vector-retrieved narrative sections in one tool response.*

3. **As an RM**, I want to reconstruct a timeline of what's happened on an account across emails, call
   notes, complaints, and transactions, so I can give an accurate status update.
   *Example: James Sullivan (CL013) — needs the original email, follow-up call note, complaint letter,
   and the transaction ledger's pending entry to agree nothing has executed yet. Requires the agent to
   chain `search_documents` and `query_transactions` — the multi-hop case §4.3 of `PLAN.md` is built
   around.*

4. **As a Compliance/Supervisor user**, I want to query suitability or concentration exceptions across
   the client book, so I can prioritize review.
   *Example: Park Ji-hoon (CL011) — 35% concentration in one Complex Product; the 20% cap is worded
   for "Retail Investors" and he's an Accredited Investor, with no documented exception either way.
   **Verified working**: the agent sets `abstained=True`, cites `POL-INV-011 §4`, and opens with
   "Whether this breaches policy is ambiguous" rather than asserting either side — this took three
   rounds of prompt strengthening to make reliable (§5 of `PLAN.md`) and should be re-tested with
   phrasing variants before being trusted as fully solved.*

5. **As an RM**, I want the assistant to clearly say when it doesn't have enough evidence, rather than
   give me a false-confident answer I might repeat to a client.
   *General abstention behavior, verified for: unknown client (ambiguous name → asks rather than
   guesses), unknown product (no fact sheet → reports missing, doesn't infer from name), and
   ambiguous policy scope (story 4).*

6. **As an RM**, I want every answer to show exactly which document and section it came from, so I can
   verify it myself before a client conversation.
   *Every citation is a `ref_id` resolved to a real locator string (e.g. `"policy_POL-INV-011_s4"` →
   `"Investment Suitability Policy Section 4"`) — never freehand-generated.*

7. **(Technical) As a developer**, I want an automated evaluation suite that scores retrieval and
   generation quality against a golden dataset, so regressions are caught before they reach an RM.
   *`run_evaluation.py`, built and run — current baseline: 4/5 factually correct (§11).*

8. **(Not yet supported) As an RM**, I want to ask which of my clients would be eligible for a newly
   launched product. *No `screen_clients_for_product`-equivalent tool exists yet — carried into §13.*

---

## 6. Core Architecture & Monorepo Patterns

**High-level approach (as built):** a flat, single-package Python project — no backend/frontend
split, no monorepo workspaces. `agent.py` is the orchestration entry point; every ingestion script is
independently runnable (`python ingest_X.py`) and idempotent (re-ingesting a revised source replaces
its old rows/chunks by ID rather than duplicating them). This is intentionally the simplest structure
that supports the MVP's actual interface (CLI + evaluation script) — see §13 for the layered
backend/frontend structure planned once a real UI/API consumer exists.

**Current repository layout:**

```
/                         # flat project root — no backend/frontend split yet
  db.py                   # SQLite schema + connection (clients, holdings, transactions,
                           #   funds, fund_key_facts, fund_asset_allocation, product_crosswalk)
  vector_store.py          # shared Chroma embedding/upsert/replace-by-id helper
  entity_crosswalk.py      # client name/ID resolution, ambiguous-match handling
  ingest_clients.py        # clients_portfolio.* + transactions.csv -> SQLite, product crosswalk
  ingest_factsheet.py      # VLM-based fact sheet PDF -> structured JSON (grounding-checked)
  ingest_fund_vectors.py   # fact sheet narrative fields -> Chroma `fund_factsheets`
  ingest_policies.py       # policy PDFs -> Chroma `policies` (+ query_policy keyword boost)
  ingest_call_notes.py     # call notes log -> Chroma `call_notes` (per-client/date chunking)
  ingest_complaints.py     # complaint letters -> Chroma `complaints` (per-CPL-ID chunking)
  ingest_ack_forms.py      # ack forms -> Chroma `ack_forms` (blank template tagged/excluded)
  ingest_correspondence.py # email threads -> Chroma `correspondence` (per-thread chunking)
  agent.py                 # tool definitions, EvidencePool, orchestrator loop, CLI entry point
  run_evaluation.py        # golden-dataset runner + LLM-judge + xlsx report writer
  data/                    # raw source files (PDFs, CSV, JSON) + data/processed/ (db, chroma, reports)
  PLAN.md                  # technical build log — architecture rationale, what's built, findings
  PRD.md                   # this document
```

**Shared dependencies strategy:** none needed yet — a single process imports the ingestion modules
and `db`/`vector_store` helpers directly; there is no API boundary to keep in sync across a
frontend/backend split. This is a direct consequence of not yet having a second consumer of the
agent (§13 introduces the contract once one exists).

**Key design pattern — tools as the single seam:** every retrieval capability is a plain Python
function with a JSON-schema tool definition and an entry in `DISPATCH`, called through the model's
function-calling turn and independently callable/unit-testable outside the agent loop (e.g.
`query_policy()` is used directly by `get_policy_section` and is also spot-checked standalone via
`ingest_policies.py --query`). Adding a knowledge source is: write an `ingest_X.py` with a `query_X()`
function, then add one tool + one `DISPATCH` entry — not a rewrite of the loop.

---

## 7. Tools/Features

| Tool | Purpose | Backing store | Key inputs | Key outputs |
|---|---|---|---|---|
| `query_client_db` | Look up a client's KYC/risk profile and holdings by `client_id` or name | SQLite `clients`, `holdings` (+ `entity_crosswalk.py` resolution) | `client_id_or_name` | client text block + `ref_id`; ambiguous-name or not-found reported as errors, not guessed |
| `query_transactions` | Look up a client's transaction ledger, optionally filtered by status | SQLite `transactions` | `client_id_or_name`, `status` (optional) | transaction rows text block + `ref_id` |
| `get_fund_factsheet` | Get a fund's SRI, minimum investment, key facts, asset allocation, objective, who-for, key risks | SQLite `funds`/`fund_key_facts`/`fund_asset_allocation` (exact fields) + Chroma `fund_factsheets` (narrative) | `product_name` | combined SQL + vector content, each with its own `ref_id`; explicit error (not inference) if no fact sheet exists |
| `get_policy_section` | Retrieve the policy section relevant to a question, with numeric-token keyword boost | Chroma `policies` | `query`, `document_code` (optional: `POL-KYC-004` / `POL-INV-011`) | matching section text(s) + `ref_id`s |
| `search_documents` | Semantic search across call notes, complaints, correspondence, ack forms, fact sheet narrative | Chroma `call_notes`, `complaints`, `correspondence`, `ack_forms`, `fund_factsheets` | `query`, `doc_types` (optional filter), `client_id` (optional filter) | matching chunks + `ref_id`s per source type |

Every tool call is logged (`tool_call_log`) and every returned fact is added to the `EvidencePool`
before the model sees it, so the final answer's citations are always resolvable back to a real
locator (`source_file`/`section`/`client_id`/`date`) — an invalid `ref_id` is caught explicitly rather
than silently trusted (§4, §6).

**Not built (see §13):** reranking of retrieved candidates, a standalone `evidence_registry_lookup`
presence/absence tool (currently folded into `search_documents` + `get_fund_factsheet`'s explicit
"no fact sheet on file" errors), and `screen_clients_for_product` cross-client eligibility screening.

---

## 8. Technology Stack

| Layer | Choice (as built) |
|---|---|
| Orchestration | Single-loop tool-calling agent in `agent.py` (no graph framework) |
| LLM | OpenAI `gpt-4o-mini` via the Responses API (`client.responses.parse`, structured `AgentAnswer` output) |
| Embeddings | OpenAI `text-embedding-3-small` (via `vector_store.py` / Chroma's OpenAI embedding function) |
| Reranking | None — Chroma similarity + a deterministic numeric-token keyword boost in `query_policy()` |
| Vector DB | ChromaDB, local persistent client (`./chroma_db`) |
| Structured DB | SQLite (`data/processed/wealth_management.db`) |
| Backend/API | None — direct function calls (`agent.run_agent()`), CLI entry point |
| Frontend | None built (Streamlit present in `requirements.txt`, unused) |
| Observability | Console `verbose=True` tool-call logging; no hosted tracing |
| Evaluation | Custom `run_evaluation.py` (LLM-as-judge) + `openpyxl` I/O against the provided golden workbook — not the `ragas` library, though `requirements.txt` lists it as available |
| Ingestion | `pypdf`/`pymupdf` (PDF text extraction), `pandas` (CSV/JSON), regex/header-based deterministic chunkers, one VLM-assisted extraction pass for fact sheets (`ingest_factsheet.py`) |
| Containerization | None |
| Testing | Manual spot-checks documented in `PLAN.md` §5 (e.g. `--query` flags on ingestion scripts); no `pytest` suite yet |

`requirements.txt` also lists `langchain`, `langgraph`, `llama-index`, `faiss-cpu`,
`sentence-transformers`, and `tensorflow` — none of these are imported by the current codebase. They
were pulled in for the exploration documented in the original PRD/architecture (§13) and are left in
place as the dependency surface for that future phase, not because the MVP uses them; trimming them is
a reasonable cleanup but out of scope for this consolidation, which is about the PRD, not the
requirements file.

---

## 9. Security & Configuration

- **Auth/access:** none implemented. There is no RM identity concept; every tool call has unrestricted
  access to all 15 clients' data. Acceptable for a single-user, local, fictitious-data evaluation
  exercise; explicitly **not** acceptable to carry into any multi-user or real-data deployment (§13,
  §14).
- **Configuration:** `.env` (via `python-dotenv`) holds the OpenAI API key; no `.env.example` currently
  committed — worth adding as a small follow-up so a fresh clone's missing-key error is self-explanatory.
- **Security scope — in place:** parameterized SQL throughout (`db.py`'s queries use `?` placeholders,
  never string interpolation); retrieved document text (emails, call notes, complaint letters) is
  passed to the LLM strictly as tool-result content, not as instructions, limiting (but not
  eliminating) prompt-injection risk from free-text sources.
- **Security scope — explicitly out of scope for MVP:** access control, encryption-at-rest, PII
  handling/redaction, audit logging, any of it. The dataset is synthetic and fictitious by design
  (`PLAN.md` §2) specifically so these gaps are safe to defer during evaluation.
- **Deployment:** none — local script execution only (`python agent.py "..."`,
  `python run_evaluation.py`).

---

## 10. API Specification

**Not applicable to the current MVP** — there is no HTTP API. The programmatic contract today is the
Python function signature:

```python
# agent.py
def run_agent(question: str, max_turns: int = 8, verbose: bool = False) -> dict:
    """
    Returns:
      {
        "answer": str,
        "citations": [{"ref_id": str, "locator": str, "text": str | None}],
        "abstained": bool,
        "abstention_reason": str | None,
        "tool_calls": [{"tool": str, "args": dict}],
        "n_evidence_retrieved": int,
      }
    """
```

`run_evaluation.py` is the reference caller. §13 carries forward the original `POST /query` /
`/feedback` / `/health` / `/admin/ingest` FastAPI design for whenever a UI or external consumer needs
this over HTTP — that design is preserved (not lost) but not re-scoped into this MVP.

---

## 11. Success Criteria

**MVP success definition (met, with documented gaps below):** the assistant answers the 5 golden
benchmark questions with grounded, cited answers and correctly abstains/hedges on the ambiguous-policy
case, evaluated automatically rather than by eyeballing.

**Functional requirements — actual status**
- ✅ Ingests and chunks all provided PDFs/CSV/JSON without manual preprocessing (spot-checked per
  source type in `PLAN.md` §5)
- ✅ 4 of 5 golden questions scored `factually_correct` by the LLM-judge harness, including the CL011
  ambiguity case
- ✅ Every generated answer's citations resolve to real evidence metadata (never freehand); an invalid
  `ref_id` is caught, not silently accepted
- ✅ System abstains rather than fabricates on missing/conflicting evidence (verified for unknown
  client, missing fact sheet, ambiguous policy scope)
- ❌ Row-level RM access scoping — not applicable yet, no RM identity concept exists
- ⚠️ **Known, open gap (row 4 / CL002):** the mis-sale suitability answer is directionally correct
  from structured tools + policy alone but doesn't reliably pull the call note / complaint letter /
  internal email the golden dataset expects as supporting citations — answer is right, evidence trail
  is incomplete. A stronger system-prompt nudge is identified as the likely fix but not yet applied.
- ⚠️ **Known, open gap (row 5 / CL015, now fixed):** initially computed a wrong LRS headroom figure
  from raw transactions instead of using the RM's already-stated figure in correspondence — fixed via
  an explicit prompt rule to check correspondence first when a question references a specific
  conversation. Left here as a reminder this class of failure exists, not as an unresolved item.
- ⚠️ **Metric caveat:** Context Precision is mechanically low (0.25–0.60) because
  `n_evidence_retrieved` counts every chunk a tool call returns, not just the ones cited (e.g.
  `get_fund_factsheet` always returns all 3 narrative sections). This under-states retrieval quality
  rather than reflecting a real defect — noted so the number isn't over-read.

**Quality indicators (not yet instrumented):** no p95 latency budget, token-cost tracking, or
audit-log coverage exists yet — all deferred to §13 alongside the API/observability layer they depend
on.

**UX goals:** not yet applicable — there is no UI to evaluate against a "usable answer in under 20
seconds" bar; the CLI's latency is whatever `gpt-4o-mini` + Chroma + SQLite round-trips take
(unmeasured).

---

## 12. Implementation Phases

Phases 1–3 below are **complete and tested**, reframed as a record of what was actually built (from
`PLAN.md` §5) rather than a forward-looking plan. Phase 4 is the next real increment.

### Phase 1 — Structured + vector ingestion, and entity resolution (✅ done)
- ✅ SQLite schema + load scripts (`db.py`, `ingest_clients.py`) with cleaning applied at load time
  (risk profile/KYC/PEP split into category + narrative + date; non-product transaction rows flagged)
- ✅ Product crosswalk with dash-normalization fix (3 false "missing fact sheet" flags caught and
  fixed before they became silent data-quality bugs)
- ✅ Fact sheet structured extraction (`ingest_factsheet.py`) + narrative vector collection
  (`ingest_fund_vectors.py`), spot-checked against a cross-fund semantic query
- ✅ Remaining document ingestion (policies, call notes, complaints, ack forms, correspondence), each
  with the correct per-document-type chunk boundary (§3.2 of `PLAN.md`)
- ✅ Entity crosswalk (`entity_crosswalk.py`) with tested ambiguous-surname handling

**Validation (met):** every ingestion script's row/chunk counts match the source data
(15 clients, 70 holdings, 55 transactions, 10 funds, 127 key facts, 27 allocation rows, 14 policy
chunks, 30 fact-sheet chunks, 10 call-note chunks, 2 complaint chunks, 7 correspondence chunks, 2 ack
form chunks with the blank template correctly excluded from default results).

### Phase 2 — Agent orchestrator, tools, citations, abstention (✅ done)
- ✅ Five tools wired into a single tool-calling loop (`agent.py`)
- ✅ `EvidencePool`/`AgentAnswer` citation design — `ref_id`s resolved to real locators, never
  model-authored
- ✅ Abstention behavior, iteratively hardened against three specific observed failure modes on the
  CL011 case (confidently asserted "does breach" → confidently asserted "does not breach" →
  flag/prose mismatch → fixed)
- ✅ Retrieval defect found and fixed: policy keyword boost for numeric-token queries

**Validation (met):** CLI runs against arbitrary questions; verbose tool-call trace confirms the
retrieve → assess → retrieve-more pattern actually occurs (e.g. the James Sullivan timeline question
pulls email + call note + complaint + transactions across multiple tool calls).

### Phase 3 — Evaluation harness (✅ done)
- ✅ `run_evaluation.py` runs all 5 golden queries live, LLM-judges each against the expected answer,
  writes a completed workbook
- ✅ Ratio metrics (Context Precision/Recall, Completeness, Faithfulness) computed and printed

**Validation (met):** `python run_evaluation.py` runs green end-to-end and produces
`golden_dataset_for_RAG_evaluation_completed.xlsx` with a 4/5 correctness summary — this is the
system's actual, current, honestly-measured baseline, not a target.

### Phase 4 — Close the known gaps, then add a UI (next)
**Goal:** fix the two documented, still-open issues from §11 before layering on any new
infrastructure, then give the CLI a usable front end.
- ☐ CL002 evidence-trail nudge: extend the system prompt so a suitability question also checks
  `search_documents` for prior escalation history (call note / complaint / internal email), not just
  the structured + policy tools
- ☐ Re-test the CL011 abstention behavior with 2–3 rephrased variants of the question — flagged in
  `PLAN.md` as "still-fragile, not solved," so this should be confirmed, not assumed
- ☐ Address the Context Precision accounting caveat (§11) — separate "chunks returned to the model"
  from "chunks presented as citable," at minimum in the evaluation script's own summary
- ☐ Minimal Streamlit front end over `run_agent()` (already in `requirements.txt`) — a text box, the
  answer, the citation list, and the abstained flag; no auth, no session state beyond one query

**Validation:** re-run `run_evaluation.py` and confirm no regression on the 4 currently-correct rows
while CL002's evidence trail improves; a fresh `streamlit run` demonstrates the same golden questions
answered through a UI instead of the CLI.

---

## 13. Future Considerations

This section preserves the larger architecture from the original product vision **as a deliberate
next phase**, not as unbuilt MVP scope. Nothing here should be read as "should have been done
already" — it's what the current SQLite+Chroma+single-loop-agent system would need to become if usage
moves beyond a single local evaluator (multiple concurrent RMs, real client data, a compliance audit
requirement, or measurable latency/cost pressure at scale):

- **HTTP API + real frontend:** FastAPI backend (`/query`, `/feedback`, `/health`, `/admin/ingest`)
  fronted by a Streamlit (or richer) chat UI — the API contract sketched in §10's original form,
  reintroduced once a second consumer of the agent actually exists.
- **RM identity + row-level access control:** every structured and vector query filtered
  server-side by `relationship_manager` unless the caller holds a compliance/supervisor role —
  necessary before any real (non-fictitious) client data touches this system.
- **Full agentic graph (LangGraph or similar):** router/planner → parallel tool execution →
  aggregator with entity resolution → faithfulness critic → **separate deterministic compliance
  rule-check node** (SRI vs. risk score, 20% concentration, CAR/CKA) that cross-verifies numeric/policy
  claims independent of the LLM — today this reasoning happens inside the single LLM loop, which is
  the biggest architectural gap between this PRD's principle #4 (§2) and the current implementation.
- **Reranking:** cross-encoder rerank (e.g. Voyage `rerank-2`) of a wider top-N candidate set before
  synthesis, to directly address the Context Precision caveat in §11.
- **Structured/queryable vector store upgrade:** Qdrant or another server-backed vector DB if
  concurrent access or scale outgrows a local persistent Chroma client; MySQL/Postgres if SQLite's
  single-writer model becomes a constraint.
- **Full observability:** LangSmith (or equivalent) tracing on every node, tagged with
  `trace_id`/`rm_id`/`client_id`, replacing today's console-only tool-call log.
- **Structured audit logging:** every query, tool call, retrieved evidence, and final answer
  persisted (not just the evaluation workbook), for compliance traceability.
- **`screen_clients_for_product`:** cross-client eligibility screening tool (user story 8, §5) — no
  backing tool exists yet.
- **RAGAS-based evaluation:** swap the custom LLM-judge script for the `ragas` library already listed
  in `requirements.txt`, for standardized metric definitions if this needs to be defensible outside
  this project.
- **Docker Compose** one-command local environment once there's more than one process to coordinate.
- Query rewriting, cross-document comparison mode, persistent multi-turn memory, enterprise SSO, PII
  redaction, data-residency controls — all as originally scoped, all still genuinely future work.

---

## 14. Risks & Mitigations

1. **Risk:** LLM gives a false-confident answer on ambiguous policy questions (CL011-style).
   **Current mitigation:** prompt-level abstention rules, hardened over three observed failure
   rounds (§12 Phase 2). **Residual risk:** this is prompt-based, not a structurally independent
   check — `PLAN.md` explicitly calls it "the single hardest behavior in the whole system... worth
   treating as still-fragile, not solved." A different phrasing should be re-tested (§12 Phase 4)
   before trusting it further; the deterministic compliance rule-check node (§13) is the eventual
   structural fix.

2. **Risk:** Retrieval misses evidence of a *missing* document (e.g., no signed risk acknowledgement),
   since vector search can't prove absence.
   **Mitigation (in place):** the blank acknowledgement-form template is explicitly tagged
   (`is_blank_template=True`) and excluded from default `query_ack_forms` results, so it can't be
   mistaken for a signed form; `get_fund_factsheet` and `query_client_db` return explicit "not found"
   errors rather than empty strings, so the agent can distinguish "checked and absent" from "didn't
   check."

3. **Risk:** Context Precision reads artificially low, which could be misread as a retrieval-quality
   problem when it's an accounting artifact (§11).
   **Mitigation:** documented explicitly in this PRD and in `PLAN.md` rather than left to be
   rediscovered; fix scoped into §12 Phase 4.

4. **Risk:** No access control means this cannot safely handle real client data or multiple RMs today.
   **Mitigation:** explicitly scoped out of the MVP (§9) rather than silently assumed; §13 names the
   exact control (`relationship_manager`-scoped queries) required before that changes.

5. **Risk:** Single-loop agent reasoning (no independent compliance rule-check) means a numeric
   suitability judgment (SRI vs. risk score, concentration %) is currently made by the LLM reading SQL
   facts, not computed deterministically — a plausible-but-wrong LLM arithmetic error wouldn't be
   caught by anything in the current pipeline.
   **Mitigation:** flagged here as the top architectural gap versus this PRD's own principle #4 (§2);
   §13's compliance rule-check node is the intended fix, prioritized ahead of unrelated new features.

6. **Risk:** Scope creep — reintroducing the full §13 architecture before the two known Phase-4 gaps
   (§11/§12) are closed would repeat the original PRD's mismatch between aspiration and what's tested.
   **Mitigation:** §12 orders Phase 4 (close known gaps, ship a minimal UI) strictly before any §13
   item is started.

---

## 15. Appendix

**Related documents**
- `PLAN.md` — technical build log: architecture rationale, data-quality findings, what's built and
  tested, and the honest findings section (§5 step 8) this PRD's §11/§14 draw from directly
- `Task.txt` — original one-line use case statement
- `data/README.md` — dataset description and suggested RAG usage
- `data/golden_dataset_for_RAG_evaluation.xlsx` — golden Q&A set with RAGAS-shaped scoring columns
- `data/processed/golden_dataset_for_RAG_evaluation_completed.xlsx` — the evaluation harness's actual
  output (generated by `run_evaluation.py`)

**Source dataset**
- 15 client profiles: `clients_portfolio.json` / `clients_portfolio.csv`
- 55-row transaction ledger: `transactions.csv`
- 10 fund fact sheets + 2 policy PDFs (`policy_kyc_onboarding.pdf`, `policy_investment_suitability.pdf`)
- `rm_call_notes_log.pdf` — RM call/meeting notes, Jan–Aug 2026
- `client_correspondence.json` — 7 email threads (client↔RM, internal compliance escalations)
- `client_complaint_letters.pdf`, `complex_product_risk_acknowledgement_forms.pdf`

**Key dependencies (as actually used):** OpenAI API (Responses API + `text-embedding-3-small`),
ChromaDB, SQLite, `pypdf`/`pymupdf`, `pandas`, `openpyxl`, `python-dotenv`.
**Listed but not currently used** (retained in `requirements.txt` for the §13 future phase):
`langchain`, `langgraph`, `llama-index`, `faiss-cpu`, `sentence-transformers`, `tensorflow`, `ragas`,
`streamlit`, `anthropic`.

**Repository structure:** see §6.
