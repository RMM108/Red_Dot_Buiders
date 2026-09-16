# Wealth Management AI Copilot — Task & Approach

## 1. Use case

Build an AI copilot for a wealth management firm (fictitious "Meridian Peak
Wealth Partners") that **retrieves and synthesizes relevant supplied content
to answer client-specific questions**, with:

- **Source citations** — every claim traceable to a specific document (and
  ideally passage) in the corpus.
- **Explicit abstention** — when the evidence is missing, contradictory, or
  the policy is genuinely ambiguous, the copilot says so instead of guessing.

The target users are relationship managers (RMs) and compliance staff who
need fast, defensible answers that combine structured client data
(portfolios, transactions) with unstructured knowledge (policies, fact
sheets, call notes, correspondence, complaints).

## 2. Dataset overview (`data/`)

All data is synthetic, prepared for this exercise (see `data/README.md`).
Nothing is sourced from the internet — real client/KYC records and issuer
fact sheets are confidential/copyrighted, so a hand-built dataset lets the
scenarios (and "correct" answers) be known in advance for evaluation.

| Category | Files | Format | Notes |
|---|---|---|---|
| **Client & portfolio data** | `clients_portfolio.json`, `clients_portfolio.csv`, `transactions.csv` | JSON (15 client objects: KYC attributes, risk profile, holdings) / CSV (70 rows, one per holding) / CSV (55-row transaction ledger, Jan–Aug 2026) | Structured. Natural fit for SQL/dataframe queries rather than embedding-based retrieval. |
| **Policy documents** | `policy_kyc_onboarding.pdf`, `policy_investment_suitability.pdf` | PDF, short (MAS-style policy text with numbered sections) | The suitability policy's §4 (Complex Product concentration limits, worded around "Retail Investor") is a deliberately ambiguous test case. |
| **Product fact sheets** | 10 `fund_factsheet_*.pdf` files, SRI 1/7 (safe) to 7/7 (exotic/unsafe) | PDF, short, semi-structured "Key Facts" tables + risk banners | Each fund has a name, SRI rating, minimum investment, and risk profile fit — used to check suitability against a client's risk score. |
| **Operational / unstructured** | `rm_call_notes_log.pdf` (10 notes), `client_correspondence.json` (7 email threads), `client_complaint_letters.pdf` (2 letters), `complex_product_risk_acknowledgement_forms.pdf` (1 filled + 1 blank) | PDF / JSON | Narrative evidence — comprehension flags, de-risking requests, complaint timelines, signed/missing acknowledgements. |
| **Evaluation** | `golden_dataset_for_RAG_evaluation.xlsx` | Excel workbook (Instructions tab + Golden Dataset tab) | 5 golden Q&A pairs with expected answer, cited sources, supporting paragraphs, and blank columns for scoring a candidate system's retrieval/generation quality (context precision/recall, answer completeness, faithfulness). |

### Reference scenarios embedded in the data

The dataset deliberately encodes traceable, multi-document storylines to
stress-test retrieval + reasoning (full detail in `data/README.md`):

- **CL002 (Robert Chua)** — Conservative client holding a 7/7 SRI structured
  note with no signed risk acknowledgement → mis-sale complaint. Evidence
  spans portfolio JSON, suitability policy, call notes, an internal
  compliance email, and the complaint letter.
- **CL008 (Carlos Bautista)** — Retail investor sold an FX-linked DCI with no
  CAR/CKA on file; includes a realised FX loss.
- **CL011 (Park Ji-hoon)** — Accredited Investor at 35% concentration in one
  Complex Product; the 20% cap is worded for "Retail Investors" — genuinely
  ambiguous, a test of over-/under-flagging.
- **CL013 (James Sullivan)** — Asked to de-risk before retirement; portfolio
  not yet rebalanced despite two weeks passing — timeline reconstruction
  across email, call note, complaint, and a "Pending" transaction row.
- **CL014 (Zhang Wei)** — PEP-adjacent, "cleared but monitored" status (not a
  simple PEP yes/no).
- **CL015 (Arjun Mehta)** — Cross-border LRS remittance headroom, a numeric
  reasoning check across an email and the client record.
- **CL001 / CL009** — Clean "positive control" cases (correctly documented,
  no breach) to check the system doesn't over-flag.

## 3. Data quality findings & cleaning plan

The core tables are clean by construction — verified via a pandas/pypdf pass
(`eda.ipynb`): no duplicate rows in `clients_portfolio.json/csv` or
`transactions.csv`, no client-ID mismatches across files, and every client's
holdings sum to exactly 100%. The issues below are real and specific, found
by inspecting the actual files rather than assumed generically.

### 3.1 Structured data (`clients_portfolio.*`, `transactions.csv`)

- **Denormalized client fields on every holding row** —
  `clients_portfolio.csv` repeats `investor_status`, `risk_profile`,
  `suitability_flag`, etc. on all 70 rows (one per holding). Not wrong, but
  don't re-merge them back in from `clients_portfolio.json` — doing exactly
  that in `eda.ipynb` produced a `KeyError` from duplicate `_x`/`_y`
  columns — and don't aggregate them across a client's holdings as if they
  were independent observations.
- **Categorical fields with embedded free text**:
  - `risk_profile` for CL013 is
    `"Conservative (revised down from Growth on 2026-08-10; portfolio not yet rebalanced)"`
    instead of a clean category.
  - `kyc_status` mixes a Verified/Not-Verified state with an embedded refresh
    date and, for several clients (CL004, CL005, CL009, CL014, CL015),
    extra EDD / source-of-funds clauses.
  - `pep_status` is a clean `"Not a PEP"` for 14 clients but a full sentence
    for CL014 (`"PEP-adjacent - former member of a municipal government
    economic advisory committee..."`).

  Clean by splitting each into **(a)** a normalized category value, **(b)** a
  parsed date where present, and **(c)** the narrative remainder kept as a
  separate free-text field — don't discard the narrative; it's the actual
  evidence for scenarios like CL014's PEP monitoring, just don't let it block
  exact-match filtering on the category.
- **Overloaded `product_name` in `transactions.csv`** — 53 of 55 rows are
  real fund names, but 2 `Pending` rows use the field for non-product
  activity labels (`"Portfolio Rebalancing (Growth to Conservative)"`,
  `"Cash (Investment Top-Up)"`). A product-lookup tool built on this column
  must skip/handle these rather than trying to fetch a fact sheet for them.
- **Product catalog gap** — CL002 holds "Singapore Government Bond Fund"
  (25% of portfolio) but there is no matching `fund_factsheet_*.pdf` in the
  corpus for it. Build a product master list cross-referencing every
  distinct `product_name` in `clients_portfolio.csv`/`transactions.csv`
  against the 10 fact sheets, and flag orphans — the copilot must **abstain**
  on SRI/fee/minimum-investment questions about these (no fact sheet =
  no evidence), not infer from the product name.
- Dates and currency are already well-formed (ISO `YYYY-MM-DD` throughout,
  only SGD/USD appear) — just parse to proper datetime/enum types on load
  rather than treating them as free text.
- `transactions.notes` is null on ~51% of rows — legitimate sparsity (not
  every transaction has a note), not something to impute.

### 3.2 Unstructured PDFs

- **Three PDFs are multi-record logs, not one record per document**:
  `rm_call_notes_log.pdf` (10 entries across 3 pages), `client_complaint_letters.pdf`
  (2 letters across 3 pages), `complex_product_risk_acknowledgement_forms.pdf`
  (1 filled example + 1 blank template across 2 pages). Naive page- or
  file-level chunking will blend unrelated clients into one retrieval hit.
  Each has a reliable, regex-splittable header:
  - call notes: `"CL0XX — <Name>"` followed by `"Date: YYYY-MM-DD"`
  - complaint letters: `"Complaint Letter — CPL-YYYY-NNN"` +
    `"Client Account: CL0XX"`
  - ack form: the second record is explicitly labelled `"BLANK TEMPLATE"`

  Segment on these markers into one chunk per client record, tagging each
  chunk with `client_id` and `date`/`complaint_id` metadata.
- The acknowledgement form's **blank template must be excluded from (or
  clearly tagged as not) evidence of a signed acknowledgement** — otherwise
  a naive retriever could return the template as if it were proof of a
  signature for a client other than CL001.
- Text extraction itself is clean — verified `pypdf` correctly extracts an
  em dash as `U+2014`, not mojibake; the `�` seen in a raw terminal dump was
  a console codepage rendering artifact, not corrupted source data. Still,
  the ingestion pipeline should read/write everything as UTF-8 throughout to
  avoid introducing real corruption downstream.
- Each fact sheet/policy/log PDF carries its own header metadata (Document
  Code, e.g. `FS-MA-SG-012`; Classification) — worth extracting into chunk
  metadata rather than left as undifferentiated body text, since RM/compliance
  users may reference documents by code.

### 3.3 Cross-document entity resolution

- Build one canonical crosswalk: `client_id → {name, name variants}` and
  `product_name → {fact sheet file | "no fact sheet on file"}`. Unstructured
  docs (call notes, complaints) reference clients by **both** name and
  `CL0XX` ID; retrieval/filtering should key on `client_id`, but chunk text
  will often only contain the name, so name variants need to resolve to the
  same ID before metadata filtering works.
- `client_correspondence.json` is already clean in this respect — every
  thread carries `related_client_id` — and needs no repair.

## 4. Suggested approach: Agentic RAG

A single flat vector index over everything is a poor fit here: half the
corpus is *structured* (needs exact filtering/aggregation, e.g. "clients
with >20% concentration"), half is *narrative* (needs semantic retrieval
across PDFs/emails), and several golden questions require **multi-hop,
cross-document synthesis** plus a judgment call on **when to abstain**. That
combination is exactly what an agentic RAG pattern — an LLM that plans
retrieval steps, calls tools, inspects intermediate results, and decides
when it has (or lacks) sufficient evidence — is designed for, rather than a
single-shot "embed the query, fetch top-k, answer" pipeline.

### 4.0 Architecture diagram

Everything below is implemented and tested (§5 walks through how). The only
piece from the original design not built is an optional chat UI.

```
+--------------------------------------------------------------+
|                     RAW SOURCES  (data/)                     |
| clients_portfolio.json/csv   transactions.csv                |
| policy_kyc_onboarding.pdf   policy_investment_suitability.pdf|
| fund_factsheet_*.pdf  x10                                    |
| rm_call_notes_log.pdf   client_complaint_letters.pdf         |
| complex_product_risk_acknowledgement_forms.pdf               |
| client_correspondence.json  (7 threads)                      |
+--------------------------------------------------------------+
                                |
                                v
+-----------------------------------------------------------------------+
|                    INGESTION PIPELINES  (all built)                   |
|                                                                       |
| structured (deterministic)                                            |
|   ingest_clients.py   [BUILT]  -> clean fields, product crosswalk     |
|                                                                       |
| document, VLM/LLM extraction                                          |
|   ingest_factsheet.py [BUILT]  -> structured record + grounding check |
|   ingest_fund_vectors.py [BUILT] -> narrative sections -> vector store|
|                                                                       |
| document, deterministic segmentation                                  |
|   ingest_policies.py      [BUILT]  -> section-numbered chunks         |
|   ingest_call_notes.py    [BUILT]  -> one chunk per CLxxx / Date entry|
|   ingest_correspondence.py[BUILT]  -> one chunk per email thread      |
|   ingest_complaints.py    [BUILT]  -> one chunk per CPL-xxxx letter   |
|   ingest_ack_forms.py     [BUILT]  -> filled vs BLANK TEMPLATE tagged |
+-----------------------------------------------------------------------+
                                    |
                                    v
+------------------------------------------------------+    +---------------------------------------------------+
|         SQL STORE  (SQLite, db.py)  [BUILT]          |    |    VECTOR STORE  (Chroma ./chroma_db)  [BUILT]    |
|                                                      |    |                                                   |
| clients       15 rows, KYC + risk profile (cleaned)  |    | policies          14 chunks                       |
| holdings      70 rows                                |    | fund_factsheets   30 chunks (objective, who-for,  |
| transactions  55 rows (2 flagged non-product)        |    |                   key_risks, extra_info per fund) |
| funds         10 rows: SRI, min investment, base_ccy,|    | call_notes        10 chunks                       |
|               doc_code  <- structured half of        |    | complaints        2 chunks                        |
|               fund_factsheets_structured.json        |    | correspondence    7 chunks                        |
| fund_key_facts (127) / fund_asset_allocation (27)    |    | ack_forms         2 chunks                        |
| product_crosswalk  11 products, 1 gap found          |    |   (blank template tagged is_blank_template=True,  |
|   (Singapore Government Bond Fund - no fact sheet)   |    |    excluded from 'signed acknowledgement' evidence|
|                                                      |    |    by default)                                    |
+------------------------------------------------------+    +---------------------------------------------------+
                                                        |
                                                        v
+--------------------------------------------------------------------------------+
|                    AGENT ORCHESTRATOR  (agent.py)  [BUILT]                     |
|                                                                                |
| query_client_db(client_id_or_name)  -> SQL  clients, holdings                  |
|                                         (+ entity_crosswalk.py name resolution)|
| query_transactions(client_id, status) -> SQL  transactions                     |
| get_fund_factsheet(product_name)    -> SQL + Vector  funds, fund_factsheets    |
| get_policy_section(query, doc_code) -> Vector  policies (+ numeric-token       |
|                                         keyword boost for exact thresholds)    |
| search_documents(query, doc_types)  -> Vector  fund_factsheets, call_notes,    |
|                                         correspondence, complaints, ack_forms  |
|                                                                                |
| loop: retrieve -> assess sufficiency -> retrieve more | answer (structured,    |
|       cites by ref_id resolved from real tool metadata, abstained flag)        |
+--------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------+
|       SYNTHESIS + CITATION + ABSTENTION  [BUILT, in agent.py]       |
| AgentAnswer{answer, citations[ref_id], abstained, abstention_reason}|
| citations resolved to locators from EvidencePool, never LLM-authored|
+---------------------------------------------------------------------+
                                   |
                                   v
+----------------------------------------------+
|             RM / Compliance user             |
| question in  ->  cited, calibrated answer out|
+----------------------------------------------+
```

### 4.1 Ingestion & indexing layer

- **Structured store** (SQLite): `clients_portfolio.json/csv` and
  `transactions.csv` loaded as queryable tables (`clients`, `holdings`,
  `transactions`) — apply the §3.1 cleaning (split `risk_profile`/
  `kyc_status`/`pep_status` into category + narrative, normalize the
  overloaded `product_name` in `transactions`) during this load, not after.
  Enables exact filters and aggregations (concentration %, date ranges,
  product matches) that embedding search does poorly.
- **Fund fact sheets: both stores, split by field type — not one or the
  other.** `data/processed/fund_factsheets_structured.json` (built via
  `ingest_factsheet.py`) already separates the fact sheet into two kinds of
  content, and each kind wants a different store:
  - *Structured/numeric fields* (`summary_risk_indicator`, `minimum_investment`,
    `base_currency`, `document_code`, `key_facts` rows like tenor/barrier
    levels) → a `funds` **SQL table**. These are what `get_fund_factsheet`
    needs for exact lookups and what a suitability check needs to *join*
    against `clients.risk_score` / `holdings.allocation_pct` — that's a
    join and a threshold comparison, not a semantic search, and doing it in
    SQL means the answer to "does this client's SRI exceed their risk
    score" is computed, not inferred by the LLM from retrieved text.
  - *Narrative fields* (`fund_objective_or_product_description`,
    `who_is_this_for`, `key_risks`, `extra_information`) → a
    `fund_factsheets` **vector collection**, embedded the same way as the
    policies (one chunk per section). These are what a fuzzy or comparative
    question needs ("which funds carry FX risk?", "why is this note
    unsuitable for a conservative investor?") and what citations need
    verbatim, quotable passages rather than a computed field.
  - Both are populated from the *same* `fund_factsheets_structured.json` —
    no second PDF pass needed, since `ingest_factsheet.py` already did the
    grounded extraction. This dual-store split is the general pattern for
    any document that mixes lookup-table fields with narrative content, and
    is exactly what §4.2's `get_fund_factsheet` (SQL) vs. `search_documents`
    (vector) tool split assumes.
- **Document store + vector index** for the remaining unstructured content:
  policy PDFs (`ingest_policies.py`, built), call notes, complaint letters,
  risk-acknowledgement forms, and correspondence — chunked per section /
  per entry / per email message (see §3.2 for the per-document-type
  boundary markers already identified) with metadata (`doc_id`, `client_id`
  if applicable, `date`, `source_file`, `section`) attached to every chunk
  for citation and filtering. The acknowledgement form's blank template is
  tagged and excluded from "signed acknowledgement" evidence per §3.2.
- **Cross-references**: tag chunks with `client_id` where derivable (email
  threads reference `related_client_id`; call notes and complaints
  reference client names/IDs directly — see §3.2) so retrieval can be
  scoped to one client without relying purely on semantic similarity.

**Implemented so far:**
- `ingest_factsheet.py` — VLM-based ingestion of new fund fact sheets into
  `data/processed/fund_factsheets_structured.{json,csv}` (the exact-match
  table behind `get_fund_factsheet`), with grounding + dataset-match
  verification. `fund_factsheet_ingestion.ipynb` did the initial 10-fund
  extraction this pipeline maintains going forward.
- `ingest_policies.py` — deterministic (regex, not LLM) section-based
  chunking of policy PDFs into a persistent Chroma vector store
  (`./chroma_db`, `text-embedding-3-small` embeddings), one chunk per
  numbered policy section (falls back to overlapping paragraph chunks for a
  future policy PDF without numbered sections). Re-ingesting a revised
  policy replaces its old chunks by `document_code` rather than
  duplicating them. Exposes `query_policy(question, document_code=None)` —
  this is the `get_policy_section` / policy half of `search_documents` from
  §4.2. Retrieval was spot-checked against known section content (e.g. a
  concentration-limit question correctly surfaces POL-INV-011 §4) — see the
  `--query` flag.

### 4.2 Agent architecture

A planner/orchestrator agent with a small set of tools, following a
retrieve → verify → synthesize → cite (or abstain) loop:

1. **Query understanding** — classify the question (single-doc lookup,
   client-specific suitability check, numeric/temporal reasoning, policy
   interpretation, multi-hop timeline) and extract entities (client name/ID,
   product name, date range).
2. **Tool-calling retrieval** (agent decides which tools to invoke, and can
   call several in sequence based on what it finds):
   - `query_client_db(client_id | name)` → structured KYC/risk/holdings from
     the SQL store.
   - `query_transactions(client_id, filters)` → ledger rows.
   - `search_documents(query, filters={doc_type, client_id, date_range})` →
     vector search over policies / fact sheets / call notes / complaints /
     correspondence, returning chunks with source metadata.
   - `get_fund_factsheet(product_name)` → direct lookup by product name
     (exact match beats semantic search for a known entity).
   - `get_policy_section(policy_name, section)` → direct section lookup once
     a policy citation is known (e.g. "suitability policy §4").
3. **Evidence assembly & sufficiency check** — after each retrieval step the
   agent evaluates: *do I have enough grounded evidence to answer every part
   of the question?* If not, it issues another targeted retrieval (this is
   the "agentic" loop — e.g. for the James Sullivan timeline question it
   must pull the email, the call note, the complaint, *and* the transaction
   ledger before it has the full picture). If evidence remains insufficient
   or contradictory after reasonable retries, it **abstains explicitly**
   rather than filling gaps with assumption.
4. **Synthesis with citations** — the answer generator is constrained to
   only assert facts traceable to retrieved chunks/rows, and must attach a
   citation (source file + section/date/row) to each claim, mirroring the
   `Citation sources` / `Relevant paragraph(s)` columns in the golden
   dataset.
5. **Abstention policy** — the agent should abstain (partially or fully)
   when: no matching client/product/policy chunk is retrieved; sources
   conflict; the policy text is ambiguous as applied (CL011 case — the
   correct behavior is to *surface* the ambiguity, not force a yes/no); or
   required documentation is missing (e.g. no signed risk acknowledgement
   found — that absence is itself an answer, not a retrieval failure).

### 4.3 Why "agentic" rather than static RAG

- **Multi-hop retrieval**: several golden questions (CL002, CL013, CL015)
  require pulling from 3–5 different sources across structured and
  unstructured stores — a single top-k vector search over one index won't
  surface all of them reliably. The agent needs to plan and chain retrievals.
- **Tool routing**: numeric/exact questions (LRS headroom, concentration %,
  minimum investment) are better answered via structured lookup/calculation
  than semantic search; the agent must choose the right tool per
  sub-question.
- **Self-assessed sufficiency → abstention**: the agent needs an explicit
  "do I have enough evidence" checkpoint before answering, which is a
  reasoning step, not a retrieval mechanic. This is precisely what
  distinguishes agentic RAG from fixed-pipeline RAG.
- **Ambiguity handling**: for CL011, the *correct* answer is "this is
  ambiguous" — the agent must recognize conflicting interpretive scope in
  the policy text itself, not just retrieve and paraphrase.

### 4.4 Evaluation

Use `golden_dataset_for_RAG_evaluation.xlsx` directly:

- Run the copilot against the 5 golden queries (plus the worked example
  row), fill in the yellow "chatbot response" columns (G–J, and the
  paragraph/claim counts in K–Q).
- The workbook auto-computes **Context Precision** (M/L), **Context Recall**
  (M/K), **Answer Completeness** (O/N), and **Faithfulness** (P/Q) —
  standard RAG metrics for retrieval quality and groundedness.
- Extend with additional held-out questions per the `README.md` "Suggested
  use" section (e.g. cross-document suitability checks, concentration-limit
  edge cases) to test generalization beyond the 5 seeded golden rows.
- Track abstention correctness separately: for questions where the "right"
  answer is to abstain or flag ambiguity (e.g. CL011), score whether the
  copilot does so instead of forcing a confident-sounding wrong answer.

### 4.5 Suggested stack (indicative, not prescriptive)

- Orchestration: an agent framework with tool-calling (e.g. Claude Agent
  SDK / LangGraph) driving the retrieve→verify→synthesize loop.
- Structured store: SQLite or DuckDB loaded from the JSON/CSV files.
- Vector store: any embedding index (e.g. Chroma, FAISS, pgvector) over
  chunked PDF/JSON text with metadata filters.
- PDF parsing: straightforward text extraction (the fact sheets/policies are
  short, text-based PDFs — no OCR needed based on inspection).
- Citation formatting: every answer segment tagged with `source_file` +
  locator (section heading, date, or row ID) pulled through from chunk
  metadata, not re-derived by the LLM.

## 5. Build status and evaluation findings

Steps 1–8 below are implemented and tested end-to-end. Only step 9 (UI)
remains.

1. **Structured store** — `db.py` (schema) + `ingest_clients.py` load
   `clients_portfolio.json`, `transactions.csv`, and
   `fund_factsheets_structured.json` into SQLite (`data/processed/
   wealth_management.db`), applying the §3.1 cleaning at load time
   (`risk_profile`/`kyc_status`/`pep_status` split into category + narrative
   + parsed date; `transactions`' 2 non-product rows flagged via
   `is_product_transaction`). 15 clients, 70 holdings, 55 transactions, 10
   funds, 127 key facts, 27 asset-allocation rows loaded and spot-checked.
2. **Product crosswalk** — built inside `ingest_clients.py`. Found a real
   bug while building it: 3 of 10 fund names disagreed between the
   PDF-extracted `fund_name` (em dash, e.g. `"... Note — Series 7"`) and the
   hand-entered `product_name` in holdings/transactions (plain hyphen or no
   separator at all), which would have produced 3 false "missing fact
   sheet" flags. Fixed with a `normalize_product_name()` matching key that
   strips dash variants before comparing (display names untouched). After
   the fix, exactly 1 genuine gap remains: **Singapore Government Bond
   Fund** has no fact sheet, as predicted in §3.1.
3. **Fund fact sheet vector collection** — `ingest_fund_vectors.py` embeds
   the narrative fields from `fund_factsheets_structured.json` into a
   `fund_factsheets` Chroma collection (30 chunks, 3 per fund). Spot-checked
   with a cross-fund semantic query ("which fund has FX currency risk") —
   correctly top-ranked the DCI (the actual FX-linked product).
4. **Remaining document ingestion** — `ingest_call_notes.py` (10 chunks,
   split on the verified `"CLxxx — Name"` / `"Date:"` header),
   `ingest_complaints.py` (2 chunks, split on `"Complaint Letter —
   CPL-YYYY-NNN"`), `ingest_ack_forms.py` (2 chunks; the blank template is
   tagged `is_blank_template=True` and confirmed excluded from
   `query_ack_forms`'s default results), `ingest_correspondence.py` (7
   chunks, one per thread). All reuse the shared `vector_store.py` helper
   (extracted from `ingest_policies.py` in this pass) for embedding +
   revision/replace logic.
5. **Entity crosswalk** — `entity_crosswalk.py`'s `resolve_client()`.
   Found a real ambiguity while testing it: CL005 "Siti Rahman" and CL006
   "Ahmad Faisal bin Rahman" share a surname, so a naive surname-match
   would have silently picked the wrong client half the time. It now
   returns `{"ambiguous": True, "candidates": [...]}` for a shared-surname
   query instead of guessing, and the agent is instructed to ask for
   clarification rather than proceed on an ambiguous match.
6. **Agent orchestrator** — `agent.py`, OpenAI Responses API tool-calling
   loop (`gpt-4o-mini`) over the 5 §4.2 tools, each backed by the stores
   above. While testing it, found and fixed a real retrieval defect: pure
   vector search ranked `POL-INV-011` §4 (the section stating the 20%
   Complex Product concentration cap) **last** out of 14 chunks for the
   query "20% concentration guideline" — below generic sections like
   "Purpose" — because embedding similarity under-weights a short,
   numerically-specific section against longer topical ones. Fixed by
   adding a keyword-match boost to `query_policy()` in `ingest_policies.py`:
   any chunk containing a numeric token from the query verbatim (e.g.
   `"20%"`) is promoted to the front regardless of embedding rank.
7. **Citation + abstention layer** — built into `agent.py`'s
   `EvidencePool`/`AgentAnswer` design: every tool call appends evidence
   tagged with a `ref_id`, the model's final answer must cite by `ref_id`
   only, and every citation is resolved back to its locator from the
   pool's own metadata — never written freehand by the model (an invalid
   `ref_id` the model didn't actually receive is caught, not trusted).
8. **Evaluation** — `run_evaluation.py` runs all 5 golden queries through
   the agent, judges each response against the expected answer with an
   LLM-judge call (clearly marked "review recommended" in the output — the
   workbook's own instructions call column J "your judgement", so treat
   this as a bootstrap, not a final grade), and writes
   `data/processed/golden_dataset_for_RAG_evaluation_completed.xlsx`.
   Final run: **4/5 factually correct**, including the CL011 ambiguity
   case (`policy_POL-INV-011_s4` cited, `abstained=True`, answer opens
   with "Whether this breaches policy is ambiguous" rather than asserting
   either side).

   **What testing this revealed, kept honest rather than smoothed over:**
   - Getting CL011 (the deliberately-ambiguous case) to reliably abstain
     took three rounds of prompt strengthening — first pass confidently
     asserted "does breach", second pass confidently asserted "does not
     breach", third pass set `abstained=True` but the answer text still
     opened with a flat assertion contradicting its own flag. Each fix
     targeted a specific observed failure (explicit scope-checking
     instruction → mechanical hedge-word self-check → prose/flag
     consistency instruction) rather than being guessed in advance. Even
     with a stronger model (`gpt-4o`) tried mid-way, the calibration was
     not reliable until the prose-consistency instruction landed. This is
     the single hardest behavior in the whole system and is worth treating
     as still-fragile, not solved — a different phrasing of the same
     question should be re-tested before trusting this in production.
   - The CL002 mis-sale question (row 4) answers correctly from
     `query_client_db` + `get_fund_factsheet` + `get_policy_section` alone,
     but doesn't reliably pull the call note / complaint letter /
     internal email that the golden dataset's expected citation set
     includes (`D`: `rm_call_notes_log.pdf`, `client_correspondence.json`
     `EML-006`, `client_complaint_letters.pdf` `CPL-2026-014`) — the answer
     is directionally right without the full expected evidence trail. A
     stronger system prompt nudge ("for a suitability question, also check
     search_documents for prior escalation history") would likely close
     this; not yet done.
   - The CL015 LRS question (row 5) initially failed outright: the agent
     computed a wrong headroom figure from `transactions.csv` instead of
     using the RM's already-calculated figure stated directly in
     `client_correspondence.json` (`EML-004`) — it never called
     `search_documents` at all. Fixed with an explicit prompt rule: when a
     question references a specific conversation, check correspondence
     before computing anything from structured records.
   - Context precision (`S`) is consistently low (0.25–0.60) because
     `n_evidence_retrieved` counts every chunk a tool call returned, not
     just the ones cited — e.g. `get_fund_factsheet` always returns all 3
     narrative sections plus the SQL facts even if only 1 is relevant to
     the question. This mechanically penalizes precision without
     reflecting a real quality problem; if this metric matters for
     reporting, retrieval-count accounting should separate "returned to
     the model" from "presented as a citable source."
9. **(Optional, not built) thin UI**: a Streamlit chat front-end (already
   in `requirements.txt`, unused so far) over `agent.run_agent()`, for
   demoing to non-technical stakeholders.
