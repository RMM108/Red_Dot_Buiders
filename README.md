# Meridian Peak Wealth Copilot

An **agentic RAG copilot** for relationship managers (RMs) at a private wealth management
division. It unifies two knowledge domains — generic product/policy knowledge (fund fact
sheets, risk-acknowledgement forms, suitability/KYC policy) and client-specific context
(portfolios, transactions, call notes, correspondence, complaints) — behind a single
tool-calling agent that plans retrieval steps, assesses whether it has enough grounded
evidence, synthesizes a cited answer, and **abstains explicitly** when evidence is missing,
conflicting, or a policy's scope is genuinely ambiguous, rather than guessing.

*(Dataset is branded "Meridian Peak Wealth Partners," a fictitious APAC private wealth
division — see `data/README.md`. All client names, correspondence, and figures are
synthetic.)*

**Status:** MVP built and evaluated. **4 of 5 golden benchmark questions answered correctly**
(including the deliberately ambiguous policy-scope case), via an automated LLM-judge +
mechanical-check evaluation harness. Known, documented gaps are listed in
[§ Known limitations](#known-limitations) rather than left implicit.

---

## Contents

- [Quick start](#quick-start)
- [Architecture](#architecture)
- [Repository layout](#repository-layout)
- [Tools](#tools)
- [Models & infrastructure](#models--infrastructure)
- [Running the pipeline](#running-the-pipeline)
- [Evaluation](#evaluation)
- [Data quality findings](#data-quality-findings)
- [Known limitations](#known-limitations)
- [Roadmap](#roadmap)
- [Related documents](#related-documents)

---

## Quick start

```bash
# 1. Install dependencies into a venv
python -m venv venv
venv/Scripts/pip install -r requirements.txt        # Windows
# venv/bin/pip install -r requirements.txt           # macOS/Linux

# 2. Set your OpenAI key
cp .env.example .env
# edit .env: OPENAI_API_KEY=sk-...

# 3. Build the stores (run once, from the project root)
python data_processing/ingest_clients.py
python data_processing/ingest_fund_vectors.py
python data_processing/ingest_policies.py
python data_processing/ingest_call_notes.py
python data_processing/ingest_complaints.py
python data_processing/ingest_ack_forms.py
python data_processing/ingest_correspondence.py

# 4. Ask it something
python data_processing/agent.py "Is Robert Chua's (CL002) holding of the APEX Global Multi-Asset Autocallable Note Series 7 suitable given his risk profile?"

# 5. Run the tests / the golden evaluation
python -m unittest discover data_processing -v
python data_processing/run_evaluation.py
```

All ingestion scripts are idempotent — re-running one replaces its own rows/chunks by ID
rather than duplicating them, so re-running step 3 after a data change is always safe.

---

## Architecture

Two backing stores, split **by field type, not by document** — structured/numeric facts
(SRI vs. risk score, concentration %, presence/absence of a signed form) are computed via
SQL, not inferred by the LLM from retrieved text; narrative facts are retrieved via vector
search. A few sources (client records, fund fact sheets) are deliberately split across both.

```
RAW SOURCES (data/)
  clients_portfolio.json/csv, transactions.csv
  policy_kyc_onboarding.pdf, policy_investment_suitability.pdf
  fund_factsheet_*.pdf x10
  rm_call_notes_log.pdf, client_complaint_letters.pdf
  complex_product_risk_acknowledgement_forms.pdf
  client_correspondence.json (7 threads)
        |
        v
INGESTION PIPELINES (data_processing/ingest_*.py)
  structured (deterministic):     ingest_clients.py
  document, VLM extraction:       ingest_factsheet.py, ingest_fund_vectors.py
  document, deterministic chunks: ingest_policies.py, ingest_call_notes.py,
                                   ingest_complaints.py, ingest_ack_forms.py,
                                   ingest_correspondence.py
        |
        v
   +-------------------------+       +----------------------------------+
   |  SQL STORE (SQLite)     |       |   VECTOR STORE (ChromaDB)        |
   |  clients, holdings,     |       |   policies, fund_factsheets,     |
   |  transactions, funds,   |       |   call_notes, complaints,        |
   |  fund_key_facts,        |       |   correspondence, ack_forms      |
   |  fund_asset_allocation, |       |   (text-embedding-3-small)       |
   |  product_crosswalk      |       |                                  |
   +-------------------------+       +----------------------------------+
        |                                          |
        +--------------------+---------------------+
                             v
              AGENT ORCHESTRATOR (data_processing/agent.py)
              query_client_db, query_transactions, query_portfolio_exposure,
              get_fund_factsheet, get_policy_section, search_documents
              loop: retrieve -> assess sufficiency -> retrieve more | answer
                    (structured AgentAnswer, cites by ref_id, abstained flag)
                             |
                             v
              SYNTHESIS + CITATION + ABSTENTION
              every claim's ref_id resolves to a real locator
              (never freehand-written); abstains on missing/
              conflicting/ambiguous evidence
                             |
                             v
                    RM / Compliance user
              question in -> cited, calibrated answer out
```

### Design principles

1. **Grounded over fluent** — every answer must trace to retrieved evidence; a wrong answer
   is worse than no answer.
2. **Cite everything** — every claim carries a `ref_id` resolved back to real source
   metadata (document, section/date/row), never freehand-written by the model.
3. **Abstain, don't guess** — ambiguous policy or missing evidence produces a calibrated
   "this is unclear" / "I don't have enough evidence," never a fabricated resolution.
4. **Right store for the shape of the data** — see above.
5. **Build the simplest thing that is honestly measured** — ship a working, evaluated
   system on the actual dataset before adding orchestration-framework or infrastructure
   complexity the current scale doesn't yet require.

---

## Repository layout

```
hackathon/
├── README.md                  <- you are here (consolidates PLAN.md + PRD.md + the old
│                                   data_processing/README.md into one entry point)
├── PLAN.md                    technical build log: architecture rationale, data-quality
│                                   findings, what's built, and the detailed findings from
│                                   testing (retrieval defects found/fixed, prompt-tuning
│                                   rounds, etc.)
├── PRD.md                     product requirements doc: mission, user stories, MVP scope,
│                                   success criteria, risks, future phases
├── Task.txt                   original one-line use case statement
├── docker-compose.yml         spins up a local Qdrant instance (prepared, not yet wired
│                                   into any ingestion script — see qdrant_store.py below)
├── requirements.txt
├── .env / .env.example        OPENAI_API_KEY (+ optional QDRANT_URL)
│
├── data/                      raw source files (see data/README.md)
│   └── processed/             pipeline outputs
│       ├── wealth_management.db                          SQLite (ingest_clients.py)
│       ├── fund_factsheets_structured.json/.csv           (ingest_factsheet.py)
│       └── golden_dataset_for_RAG_evaluation_completed.xlsx  (run_evaluation.py output)
│
├── chroma_db/                 persistent Chroma vector store (all ingest_*.py write here)
│
└── data_processing/           <- the entire pipeline + agent + tests live here
    ├── db.py                      SQLite schema + connection helper
    ├── vector_store.py            Chroma connection/embedding helper (shared)
    ├── qdrant_store.py            Qdrant counterpart to vector_store.py — prepared for a
    │                               future server-backed vector DB migration, not currently
    │                               used by any ingest script (see PRD.md §13)
    ├── chunking.py                shared fixed-size fallback chunker, used by
    │                               ingest_call_notes.py/ingest_complaints.py when their
    │                               primary header-based split finds no matches
    ├── entity_crosswalk.py        client name -> client_id resolver (ambiguity-aware)
    ├── persona.py                 standalone client-persona matching utility (characteristics
    │                               -> comparable clients' portfolio patterns); not currently
    │                               wired into agent.py's tool set — see below
    │
    ├── ingest_clients.py          clients_portfolio.* + transactions.csv -> SQLite
    ├── ingest_factsheet.py        new fund fact sheet PDF -> structured JSON (VLM)
    ├── ingest_fund_vectors.py     fact sheet narrative fields -> Chroma fund_factsheets
    ├── ingest_policies.py         policy PDFs -> Chroma policies
    ├── ingest_call_notes.py       call notes log -> Chroma call_notes
    ├── ingest_complaints.py       complaint letters -> Chroma complaints
    ├── ingest_ack_forms.py        risk acknowledgement forms -> Chroma ack_forms
    ├── ingest_correspondence.py   email threads -> Chroma correspondence
    │
    ├── agent.py                   tool definitions, EvidencePool, orchestrator loop, CLI
    ├── run_evaluation.py          golden-dataset runner: LLM-judge + mechanical checks
    │
    ├── test_agent_tools.py        26 unittest cases: agent tools, ingestion boundaries,
    ├── test_rag_copilot.py            DB inventory, evaluation contracts, persona matching
    │
    ├── eda.ipynb                  exploratory analysis of the raw structured data
    └── fund_factsheet_ingestion.ipynb   original 10-fund fact sheet extraction run
```

**A note on `persona.py`:** it's a fully working, independently tested (9/9 tests) module
that matches a set of supplied characteristics (age, net worth band, investment objective,
...) against the 15 client records to surface comparable clients and their portfolio
patterns — useful for "what would a client like this typically hold" discussion prep. It
reads directly from `clients_portfolio.json`/`transactions.csv`/`client_correspondence.json`
and is deliberately **not** wired into `agent.py`'s tool set yet (no `generate_persona` tool
exists) — its docstring is explicit that persona output is discussion context, never a
product recommendation, which is a design stance worth deciding on deliberately before
exposing it to the agent loop, not defaulting into by omission.

---

## Tools

`agent.py`'s orchestrator loop has 6 tools:

| Tool | Purpose | Backing store | Key inputs |
|---|---|---|---|
| `query_client_db` | Client KYC/risk profile + holdings, by `client_id` or name | SQLite `clients`, `holdings` (+ `entity_crosswalk.py` name resolution) | `client_id_or_name` |
| `query_transactions` | Client transaction ledger, optionally date/status filtered | SQLite `transactions` | `client_id_or_name`, `status`, `date_from`, `date_to` |
| `query_portfolio_exposure` | Cross-client concentration screening (compliance/supervisor use case) | SQLite `clients` join `holdings` | `product_type`, `threshold_pct`, `investor_status` |
| `get_fund_factsheet` | Fund SRI, minimum investment, key facts, allocation, objective, risks | SQLite `funds`/`fund_key_facts`/`fund_asset_allocation` (exact) + Chroma `fund_factsheets` (narrative) | `product_name` |
| `get_policy_section` | Relevant policy section, with numeric-token keyword boost | Chroma `policies` | `query`, `document_code` (optional) |
| `search_documents` | Semantic search across call notes, complaints, correspondence, ack forms, fact-sheet narrative | Chroma (5 collections) | `query`, `doc_types`, `client_id` (optional filters) |

Every tool call appends its result to an `EvidencePool` keyed by `ref_id` before the model
sees it. The model's structured final answer (`AgentAnswer`) cites by `ref_id` only; each
citation is resolved back to a real locator (source file + section/date/client) — an invalid
`ref_id` the model never actually received is caught explicitly, not silently trusted.

---

## Models & infrastructure

| Purpose | Choice |
|---|---|
| Agent reasoning / tool-calling loop | OpenAI `gpt-4o-mini`, Responses API (`client.responses.parse`, structured `AgentAnswer` output, `temperature=0`) |
| Fund fact sheet extraction (reads PDF pages as **images**) | OpenAI `gpt-4o-mini` (vision), Responses API, structured output |
| Evaluation grading (LLM-as-judge) | OpenAI `gpt-4o-mini`, Responses API |
| Embeddings | OpenAI `text-embedding-3-small` |
| Structured store | SQLite, `data/processed/wealth_management.db`, no server |
| Vector store | ChromaDB, local persistent client, `./chroma_db` |
| Vector store (prepared, unused) | Qdrant — `qdrant_store.py` + `docker-compose.yml`, not wired into any ingest script |
| Orchestration | Single-loop tool-calling agent (no graph framework) |
| Reranking | None — Chroma similarity + a deterministic numeric-token keyword boost (`vector_store.keyword_boosted_query`) |
| Backend/API | None — direct function calls (`agent.run_agent()`), CLI entry point |
| Frontend | None built (Streamlit listed in `requirements.txt`, unused) |

`requirements.txt` also lists `langchain`, `langgraph`, `llama-index`, `faiss-cpu`,
`sentence-transformers`, `tensorflow`, `ragas`, `anthropic` — none of these are imported by
the current codebase; they're retained as the dependency surface for future phases (see
[Roadmap](#roadmap)), not because the MVP uses them.

---

## Running the pipeline

Run everything from the **project root**, not from inside `data_processing/` — every script
resolves `../data`, `../chroma_db`, and `../.env` via `Path(__file__).parent.parent`.

### 1. Structured + vector ingestion (run once, or after a data change)

```bash
python data_processing/ingest_clients.py        # clients, holdings, transactions, funds -> SQLite
python data_processing/ingest_fund_vectors.py   # fact sheet narrative fields -> Chroma
python data_processing/ingest_policies.py       # policy PDFs -> Chroma
python data_processing/ingest_call_notes.py     # call notes log -> Chroma
python data_processing/ingest_complaints.py     # complaint letters -> Chroma
python data_processing/ingest_ack_forms.py      # risk ack forms -> Chroma
python data_processing/ingest_correspondence.py # email threads -> Chroma
```

`ingest_factsheet.py` is separate — it ingests **one new fund fact sheet PDF** at a time
(`python data_processing/ingest_factsheet.py path/to/new_factsheet.pdf`), not part of the
batch above. The 10 fact sheets already in this dataset were extracted this way; see
`data_processing/fund_factsheet_ingestion.ipynb` for that original batch run.

Current row/chunk counts (verified by `test_agent_tools.py`'s `DatabaseInventoryTests`):

| Store | Contents |
|---|---|
| SQLite | 15 clients, 70 holdings, 55 transactions, 10 funds, 127 key facts, 27 allocation rows, 11 product-crosswalk entries (1 gap: Singapore Government Bond Fund has no fact sheet) |
| Chroma | `policies` 14, `fund_factsheets` 157, `call_notes` 10, `complaints` 2, `correspondence` 7, `ack_forms` 2 — 192 chunks total |

### 2. Ask a question

```bash
python data_processing/agent.py "Does Park Ji-hoon's (CL011) 35% allocation to the APEX Autocallable Note breach firm policy?"
python data_processing/agent.py "Which clients hold Complex Products above the 20% concentration guideline?"
```

### 3. Tests

```bash
python -m unittest discover data_processing -v
```

26 tests: agent tool contracts (client lookup, transactions, portfolio exposure, fact
sheets), ingestion boundary correctness (call notes split into exactly 10 client records,
complaints tagged to the right client, the blank ack-form template excluded from signed
evidence, policy sections include the 20% rule), a database-inventory check against the
counts above, evaluation-contract checks (`mechanical_checks`), and 9 persona-matching
tests.

---

## Evaluation

`data_processing/run_evaluation.py` runs the live agent against the 5 golden queries in
`data/golden_dataset_for_RAG_evaluation.xlsx`, scores each response two ways, and writes
`data/processed/golden_dataset_for_RAG_evaluation_completed.xlsx`:

- **Mechanical checks** (`mechanical_checks()`) — deterministic, no LLM: every source the
  golden dataset expects is actually cited, no citation resolves to an invalid `ref_id`, and
  the answer text contains the specific figures/terms the golden answer hinges on (e.g. the
  exact LRS headroom numbers for the CL015 question).
- **LLM-as-judge** — `factually_correct` (Yes/No), key-points-covered, and claims-supported
  vs. claims-total, all graded by a `gpt-4o-mini` call against the expected answer and cited
  evidence. Explicitly marked `[LLM-judge auto-grade, review recommended]` in the output —
  the workbook's own instructions call column J "your judgement," so this is a bootstrap for
  human review, not a final grade.

**Current baseline: 4 of 5 factually correct**, including the deliberately ambiguous CL011
policy-scope question (agent sets `abstained=True`, cites `POL-INV-011 §4`, opens with
"Whether this breaches policy is ambiguous" rather than asserting either side). See
[Known limitations](#known-limitations) for the one open gap (row 4 / CL002) and
`PLAN.md` §5 for the full blow-by-blow of what testing found and fixed along the way.

---

## Data quality findings

Found by inspecting the actual files, not assumed generically (full detail in `PLAN.md` §3):

- **Denormalized/embedded fields**: `clients_portfolio.json`'s `risk_profile`, `kyc_status`,
  and `pep_status` mix a clean category with embedded free text (e.g.
  `"Conservative (revised down from Growth on 2026-08-10; portfolio not yet rebalanced)"`) —
  split into category + narrative + parsed date at load time in `ingest_clients.py`.
- **Overloaded `product_name`**: 2 of 55 `transactions.csv` rows aren't actual product
  trades (e.g. `"Portfolio Rebalancing (Growth to Conservative)"`) — flagged via
  `is_product_transaction` rather than treated as fund lookups.
- **Product-name punctuation drift**: fund names extracted from PDFs use an em dash
  (`"... Note — Series 7"`); the hand-entered `product_name` in holdings/transactions uses a
  plain hyphen or no separator at all. A naive exact-match crosswalk produced 3 **false**
  "missing fact sheet" flags before this was caught and fixed with dash-normalized matching.
- **One genuine gap**: "Singapore Government Bond Fund" has no fact sheet in the corpus —
  the agent reports this as missing evidence rather than inferring its SRI from the name.
- **Multi-record PDFs**: `rm_call_notes_log.pdf` (10 entries) and
  `client_complaint_letters.pdf` (2 letters) each need header-based segmentation, not
  page-based chunking, to avoid blending unrelated clients into one retrieval hit.
- **A blank template masquerading as evidence**: `complex_product_risk_acknowledgement_forms.pdf`
  contains one filled record (CL001) and one blank template — the template is explicitly
  tagged `is_blank_template=True` and excluded from `query_ack_forms`'s default results so
  it can never be mistaken for proof of a signature.

---

## Known limitations

- **Row 4 (CL002) evidence trail is incomplete.** The mis-sale suitability answer is
  directionally correct from `query_client_db` + `get_fund_factsheet` + `get_policy_section`
  alone, but doesn't reliably also pull the call note / complaint letter / internal email the
  golden dataset expects as supporting citations. The answer's substance is right; the
  evidence trail is partial. Likely fix: a stronger system-prompt nudge to check
  `search_documents` for prior escalation history on suitability questions — identified, not
  yet applied.
- **CL011 abstention is prompt-based, not structurally independent.** Getting the
  deliberately-ambiguous policy-scope question to reliably abstain took three rounds of
  prompt strengthening (see `PLAN.md` §5) — it is explicitly flagged there as "still-fragile,
  not solved." A different phrasing of the same question should be re-tested before trusting
  this further.
- **Context Precision reads artificially low (0.25–0.60).** `n_evidence_retrieved` counts
  every chunk a tool call returns, not just the ones cited — e.g. `get_fund_factsheet` always
  returns all 3 narrative sections even if only one is relevant. This is an accounting
  artifact, not a retrieval-quality defect, but it under-states the number if read at face
  value.
- **No access control.** There is no RM identity concept; every tool call sees all 15
  clients' data. Acceptable for a single-user, local, fictitious-data evaluation; not
  acceptable for any multi-user or real-data deployment.
- **No independent compliance rule-check.** Numeric suitability judgments (SRI vs. risk
  score, 20% concentration) are currently reasoned about by the LLM reading SQL facts, not
  computed by a standalone deterministic rule engine. A plausible-but-wrong LLM arithmetic
  error wouldn't be caught by anything else in the pipeline today.
- **`persona.py` is not agent-integrated** (see [Repository layout](#repository-layout)) and
  **`qdrant_store.py`/`docker-compose.yml` are prepared but unused** — both are
  fully-functional, independently tested pieces of infrastructure, not yet wired into the
  live retrieval path. Documented here so neither is mistaken for "built and working end to
  end" the way the rest of the pipeline is.

---

## Roadmap

Not built, and not pretended to be — see `PRD.md` §12/§13 for the full phased breakdown:

- **Next (Phase 4):** close the CL002 evidence-trail gap; re-test CL011 abstention with
  rephrased variants; fix the Context Precision accounting; ship a minimal Streamlit front
  end over `run_agent()`.
- **Later:** HTTP API (FastAPI) + real chat UI; RM identity + row-level access control; a
  full agentic graph (LangGraph) with a **separate deterministic compliance rule-check node**
  that cross-verifies numeric/policy claims independent of the LLM (the biggest architectural
  gap versus this project's own "right store for the shape of the data" principle); a
  cross-encoder reranking stage; the Qdrant migration `qdrant_store.py` is already prepared
  for; full observability/tracing; structured audit logging; a `screen_clients_for_product`
  eligibility-screening tool; swapping the custom LLM-judge for `ragas` (already in
  `requirements.txt`); Docker Compose for the whole stack, not just Qdrant.

---

## Related documents

- **`PLAN.md`** — technical build log: architecture rationale, the data-quality findings
  above in full, and a step-by-step account of what was built, tested, and found along the
  way (including every retrieval defect found and fixed, and the three-round CL011
  prompt-tuning story).
- **`PRD.md`** — product requirements: mission, target users, user stories, MVP scope
  (in/out), success criteria, risks & mitigations, and the full future-phases architecture.
- **`Task.txt`** — the original one-line use case statement this project started from.
- **`data/README.md`** — dataset description, why it's synthetic, and suggested RAG usage.
- **`data/golden_dataset_for_RAG_evaluation.xlsx`** — the golden Q&A benchmark.
