"""Run the agent against the 5 golden queries in
data/golden_dataset_for_RAG_evaluation.xlsx and fill in the "after running
your chatbot" columns (PLAN.md §5 step 8).

Columns G, H, I, L are filled mechanically from the agent's own run:
  G = the agent's answer
  H = source files/policies referenced (from citation locators)
  I = the specific passages referenced (full locator text)
  L = total evidence chunks/rows retrieved across all tool calls

Columns J, O, P, Q require judging the response against the expected answer
(F) and are NOT purely mechanical - this script uses an LLM-as-judge call
for them (clearly marked as such in Remarks) as a first pass. The workbook's
own instructions describe J as "your judgement", so treat these as a
starting point for human review, not a final grade - M (relevant paragraphs
retrieved) is approximated as the number of sources the agent actually
cited, which undercounts if the agent retrieved something relevant but
failed to cite it.

Writes a new file (golden_dataset_for_RAG_evaluation_completed.xlsx)
alongside the original rather than overwriting it.

Usage:
    python run_evaluation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import openpyxl
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from agent import run_agent

load_dotenv()

DATA_DIR = Path(__file__).parent.parent / "data"
SOURCE_XLSX = DATA_DIR / "golden_dataset_for_RAG_evaluation.xlsx"
OUTPUT_XLSX = DATA_DIR / "processed" / "golden_dataset_for_RAG_evaluation_completed.xlsx"

JUDGE_MODEL = "gpt-4o-mini"
GOLDEN_ROWS = range(3, 8)  # rows 3-7; row 2 is the worked example, left untouched

COL = {"query": 3, "expected_answer": 6, "n_relevant_in_kb": 11, "n_key_points_expected": 14,
       "response": 7, "sources": 8, "paragraphs": 9, "factually_correct": 10,
       "n_retrieved": 12, "n_relevant_retrieved": 13, "n_key_points_generated": 15,
       "n_claims_supported": 16, "n_claims_total": 17, "remarks": 18}


class EvalJudgment(BaseModel):
    factually_correct: bool
    key_points_covered: int
    n_claims_in_response: int
    n_claims_supported: int
    remarks: str


JUDGE_SYSTEM_PROMPT = """You are grading a RAG chatbot's response against a golden expected answer, for \
a wealth-management compliance evaluation. Judge ONLY from the text given - the expected answer, the \
generated response, and the cited evidence snippets.

- factually_correct: does the generated response's substance match the expected answer? For a \
question whose expected answer says the situation is ambiguous (not a firm yes/no), a response that \
also correctly identifies the ambiguity counts as correct even if worded differently - a response \
that forces a confident yes/no where the expected answer says "ambiguous" is NOT correct.
- key_points_covered: of the {n_expected} distinct factual points in the expected answer, how many \
does the generated response also state (count overlap, not the response's own total).
- n_claims_in_response: total distinct factual claims made in the generated response.
- n_claims_supported: of those claims, how many are directly backed by the cited evidence snippets \
provided (not just plausible-sounding - actually stated in the evidence text).
- remarks: one or two sentences on failure modes, hallucination, or missed nuance, if any.
"""


def judge_response(query: str, expected_answer: str, n_key_points_expected: int,
                    generated_answer: str, citations: list[dict]) -> EvalJudgment:
    client = OpenAI()
    evidence_text = "\n\n".join(f"[{c['ref_id']}] {c['text']}" for c in citations if c.get("text"))

    user_content = (
        f"Query: {query}\n\nExpected answer ({n_key_points_expected} key points):\n{expected_answer}\n\n"
        f"Generated response:\n{generated_answer}\n\nCited evidence:\n{evidence_text}"
    )
    resp = client.responses.parse(
        model=JUDGE_MODEL,
        input=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT.format(n_expected=n_key_points_expected)},
            {"role": "user", "content": user_content},
        ],
        text_format=EvalJudgment,
        temperature=0,
    )
    return resp.output_parsed


def run_row(ws, row: int) -> dict:
    query = ws.cell(row, COL["query"]).value
    expected_answer = ws.cell(row, COL["expected_answer"]).value
    n_key_points_expected = ws.cell(row, COL["n_key_points_expected"]).value

    print(f"\n--- row {row}: {query[:80]}...")
    result = run_agent(query, verbose=False)

    locators = [c["locator"] for c in result["citations"]]
    sources = sorted({loc.split(" (")[0] for loc in locators})

    judgment = judge_response(query, expected_answer, n_key_points_expected, result["answer"], result["citations"])

    remarks = judgment.remarks
    if result["abstained"]:
        remarks = f"[agent abstained: {result['abstention_reason']}] {remarks}"
    remarks = f"[LLM-judge auto-grade, review recommended] {remarks}"

    ws.cell(row, COL["response"]).value = result["answer"]
    ws.cell(row, COL["sources"]).value = "; ".join(sources)
    ws.cell(row, COL["paragraphs"]).value = "; ".join(locators)
    ws.cell(row, COL["factually_correct"]).value = "Yes" if judgment.factually_correct else "No"
    ws.cell(row, COL["n_retrieved"]).value = result["n_evidence_retrieved"]
    ws.cell(row, COL["n_relevant_retrieved"]).value = len(result["citations"])
    ws.cell(row, COL["n_key_points_generated"]).value = judgment.key_points_covered
    ws.cell(row, COL["n_claims_supported"]).value = judgment.n_claims_supported
    ws.cell(row, COL["n_claims_total"]).value = judgment.n_claims_in_response
    ws.cell(row, COL["remarks"]).value = remarks

    print(f"  abstained={result['abstained']}  factually_correct={judgment.factually_correct}  "
          f"n_retrieved={result['n_evidence_retrieved']}  n_cited={len(result['citations'])}")

    return {
        "row": row, "query": query, "abstained": result["abstained"],
        "factually_correct": judgment.factually_correct,
        "n_retrieved": result["n_evidence_retrieved"], "n_cited": len(result["citations"]),
        "n_relevant_in_kb": ws.cell(row, COL["n_relevant_in_kb"]).value,
        "n_key_points_expected": n_key_points_expected, "key_points_covered": judgment.key_points_covered,
        "n_claims_total": judgment.n_claims_in_response, "n_claims_supported": judgment.n_claims_supported,
    }


def main() -> int:
    wb = openpyxl.load_workbook(SOURCE_XLSX)
    ws = wb["Golden Dataset"]

    summaries = [run_row(ws, row) for row in GOLDEN_ROWS]

    OUTPUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUTPUT_XLSX)
    print(f"\nsaved: {OUTPUT_XLSX}")

    def ratio(numerator, denominator) -> str:
        if not denominator:
            return "n/a"
        return f"{numerator / denominator:.2f}"

    print("\n=== Summary (computed the same way the workbook's S-V formulas would) ===")
    print(f"{'row':>3} {'abstained':>9} {'correct':>7} {'ctx_prec':>8} {'ctx_recall':>10} {'completeness':>12} {'faithfulness':>12}")
    for s in summaries:
        ctx_prec = ratio(s["n_cited"], s["n_retrieved"])
        ctx_recall = ratio(s["n_cited"], s["n_relevant_in_kb"])
        completeness = ratio(s["key_points_covered"], s["n_key_points_expected"])
        faithfulness = ratio(s["n_claims_supported"], s["n_claims_total"])
        print(f"{s['row']:>3} {str(s['abstained']):>9} {str(s['factually_correct']):>7} "
              f"{ctx_prec:>8} {ctx_recall:>10} {completeness:>12} {faithfulness:>12}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
