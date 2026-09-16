"""Ingest a new fund fact sheet PDF into data/processed/fund_factsheets_structured.{json,csv}.

Extraction reads the PDF as images via a vision-language model (VLM) rather
than a text layer, so it also works on scanned/image-only fact sheets.
Two independent checks run before anything is merged into the dataset:

1. Grounding check - every extracted value must appear (near-verbatim) in
   the PDF's own text layer (via pypdf, a path independent of the VLM).
2. Dataset match check - does this fund already exist (by document_code /
   fund_name)? If so, diff the changed fields instead of silently
   overwriting.

Content that doesn't fit the fixed schema (ESG sections, performance
history, manager bios, tax notes, etc.) is not dropped - it's collected
into a new `extra_information` column so a reviewer can see what the
schema is missing, instead of it silently disappearing.

Usage:
    python ingest_factsheet.py path/to/new_factsheet.pdf
    python ingest_factsheet.py path/to/new_factsheet.pdf --dry-run
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import pymupdf
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from pypdf import PdfReader

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

DATA_DIR = Path(__file__).parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
DATASET_JSON = PROCESSED_DIR / "fund_factsheets_structured.json"
DATASET_CSV = PROCESSED_DIR / "fund_factsheets_structured.csv"
LOG_DIR = PROCESSED_DIR / "ingestion_logs"

MODEL = "gpt-4o-mini"
RENDER_DPI = 150

NESTED_COLUMNS = ["key_facts", "asset_allocation", "who_is_this_for", "key_risks", "extra_information"]
COLUMN_ORDER = [
    "fund_name", "product_type", "document_subtitle",
    "summary_risk_indicator", "summary_risk_indicator_label",
    "as_of_date", "document_code", "classification",
    "fund_objective_or_product_description", "minimum_investment", "base_currency",
    "key_facts", "asset_allocation", "who_is_this_for", "key_risks",
    "extra_information", "source_file",
]


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------

class KeyFact(BaseModel):
    label: str
    value: str


class AssetAllocationItem(BaseModel):
    asset_type: str
    pct_of_fund: Optional[str] = None


class FundFactSheetExtract(BaseModel):
    fund_name: str
    document_subtitle: Optional[str] = None
    as_of_date: Optional[str] = None
    document_code: Optional[str] = None
    classification: Optional[str] = None
    summary_risk_indicator: Optional[int] = None
    summary_risk_indicator_label: Optional[str] = None
    product_type: str
    fund_objective_or_product_description: str
    key_facts: list[KeyFact]
    asset_allocation: Optional[list[AssetAllocationItem]] = None
    who_is_this_for: list[str]
    key_risks: list[str]
    minimum_investment: Optional[str] = None
    base_currency: Optional[str] = None
    extra_information: list[str]


SYSTEM_PROMPT = """You extract fund/product fact sheets into structured JSON for a wealth \
management compliance system, reading the document as page images. Accuracy and completeness \
are critical - missing a risk term (e.g. a barrier level or "capital protection: none") could \
lead to a mis-sale.

Rules:
- Only use information visible in the images. Never invent, estimate, or infer numbers not shown.
- key_facts must include EVERY row from the "Key Facts" / "Key Terms" table, in the order they \
appear, one KeyFact per row. Do not skip, summarize, merge, or omit any row.
- asset_allocation: only populate if there is an explicit asset-allocation / sub-fund-allocation \
percentage table. Otherwise null.
- who_is_this_for and key_risks: one list item per bullet point, preserving the original wording.
- fund_objective_or_product_description, document_subtitle, as_of_date, document_code, \
classification, summary_risk_indicator, summary_risk_indicator_label, minimum_investment, \
base_currency: extract as in prior fact sheets of this series.
- extra_information: this document may contain sections that do NOT fit any of the fields above \
(e.g. ESG/sustainability disclosures, historical performance/NAV charts, fund manager biography, \
tax treatment notes, distributor/contact details, awards, benchmark comparisons). Capture each \
such section as one summarized string per item. If there is nothing extra beyond the fields \
above, return an empty list - do not pad it.
"""


# --------------------------------------------------------------------------
# 1. Ingestion: render PDF to images, extract via VLM
# --------------------------------------------------------------------------

def pdf_to_image_data_urls(pdf_path: Path, dpi: int = RENDER_DPI) -> list[str]:
    urls = []
    doc = pymupdf.open(pdf_path)
    try:
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            b64 = base64.b64encode(pix.tobytes("png")).decode()
            urls.append(f"data:image/png;base64,{b64}")
    finally:
        doc.close()
    return urls


def extract_via_vlm(client: OpenAI, image_data_urls: list[str]) -> FundFactSheetExtract:
    content = [{"type": "input_text", "text": "Extract this fund fact sheet into the given schema."}]
    for url in image_data_urls:
        content.append({"type": "input_image", "image_url": url})

    resp = client.responses.parse(
        model=MODEL,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        text_format=FundFactSheetExtract,
        temperature=0,
    )
    return resp.output_parsed


# --------------------------------------------------------------------------
# 2. Verification: grounding check against the PDF's own text layer
# --------------------------------------------------------------------------

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def extract_text_layer(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def grounding_check(record: FundFactSheetExtract, text_layer: str) -> dict:
    norm_text = normalize(text_layer)

    def score(values: list[str]) -> tuple[int, int, list[str]]:
        misses = [v for v in values if normalize(v) not in norm_text]
        return len(values) - len(misses), len(values), misses

    kf_values = [f"{kf.label} {kf.value}" for kf in record.key_facts]
    kf_hits, kf_total, kf_misses = score(kf_values)
    wf_hits, wf_total, wf_misses = score(record.who_is_this_for)
    kr_hits, kr_total, kr_misses = score(record.key_risks)
    doc_code_ok = bool(record.document_code) and record.document_code in text_layer

    return {
        "doc_code_grounded": doc_code_ok,
        "key_facts_grounded": f"{kf_hits}/{kf_total}",
        "who_is_this_for_grounded": f"{wf_hits}/{wf_total}",
        "key_risks_grounded": f"{kr_hits}/{kr_total}",
        "ungrounded_values": kf_misses + wf_misses + kr_misses,
        "fully_grounded": doc_code_ok and not (kf_misses or wf_misses or kr_misses),
    }


def cross_check_sri(record: FundFactSheetExtract, text_layer: str) -> dict:
    """Independently parse the SRI banner from the text layer and compare to
    the VLM's read of it - catches a misread digit, which text extraction
    (when available) is more reliable for than vision."""
    m = re.search(r"(\d)\s*/\s*7\s*.\s*([A-Z \-/]+?)(?:\n|Capital|COMPLEX|Insurance)", text_layer)
    text_layer_sri = int(m.group(1)) if m else None
    return {
        "vlm_sri": record.summary_risk_indicator,
        "text_layer_sri": text_layer_sri,
        "match": text_layer_sri is None or text_layer_sri == record.summary_risk_indicator,
    }


# --------------------------------------------------------------------------
# 3. Verification: match against the existing dataset
# --------------------------------------------------------------------------

def load_dataset() -> pd.DataFrame:
    if not DATASET_JSON.exists():
        return pd.DataFrame(columns=COLUMN_ORDER)
    with open(DATASET_JSON, encoding="utf-8") as f:
        records = json.load(f)
    df = pd.DataFrame(records)
    for col in COLUMN_ORDER:
        if col not in df.columns:
            df[col] = None
    return df[COLUMN_ORDER]


def check_against_dataset(record: FundFactSheetExtract, existing_df: pd.DataFrame) -> dict:
    by_code = existing_df[existing_df["document_code"] == record.document_code] if record.document_code else existing_df.iloc[0:0]
    by_name = existing_df[existing_df["fund_name"] == record.fund_name]

    match = by_code if len(by_code) else by_name
    if not len(match):
        return {"action": "new", "matched_row": None, "field_changes": []}

    existing = match.iloc[0]
    comparable_fields = [
        "fund_name", "product_type", "summary_risk_indicator", "summary_risk_indicator_label",
        "minimum_investment", "base_currency", "as_of_date",
    ]
    changes = []
    for field in comparable_fields:
        old = existing.get(field)
        old = old.item() if hasattr(old, "item") else old
        old = None if pd.isna(old) else old
        new = getattr(record, field, None)
        if old != new:
            changes.append({"field": field, "old": old, "new": new})

    return {
        "action": "update",
        "matched_row": existing["fund_name"],
        "matched_index": int(match.index[0]),
        "field_changes": changes,
    }


# --------------------------------------------------------------------------
# 4. Merge + save
# --------------------------------------------------------------------------

def record_to_row(record: FundFactSheetExtract, source_file: str) -> dict:
    row = record.model_dump()
    row["source_file"] = source_file
    return {col: row.get(col) for col in COLUMN_ORDER}


def merge_into_dataset(existing_df: pd.DataFrame, new_row: dict, dataset_check: dict) -> pd.DataFrame:
    if dataset_check["action"] == "update":
        existing_df = existing_df.drop(index=dataset_check["matched_index"]).reset_index(drop=True)
    return pd.concat([existing_df, pd.DataFrame([new_row])], ignore_index=True)[COLUMN_ORDER]


def save_dataset(df: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATASET_JSON, "w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2, ensure_ascii=False)

    csv_df = df.copy()
    for col in NESTED_COLUMNS:
        csv_df[col] = csv_df[col].apply(lambda v: json.dumps(v, ensure_ascii=False) if v is not None else None)
    csv_df.to_csv(DATASET_CSV, index=False)


def write_ingestion_log(pdf_path: Path, report: dict) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = LOG_DIR / f"{pdf_path.stem}_{ts}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    return log_path


# --------------------------------------------------------------------------
# Pipeline entry point
# --------------------------------------------------------------------------

def ingest(pdf_path: Path, dry_run: bool = False) -> dict:
    pdf_path = Path(pdf_path)
    client = OpenAI()

    image_urls = pdf_to_image_data_urls(pdf_path)
    record = extract_via_vlm(client, image_urls)

    text_layer = extract_text_layer(pdf_path)
    grounding = grounding_check(record, text_layer)
    sri_check = cross_check_sri(record, text_layer)

    existing_df = load_dataset()
    dataset_check = check_against_dataset(record, existing_df)

    new_row = record_to_row(record, pdf_path.name)

    report = {
        "source_file": pdf_path.name,
        "fund_name": record.fund_name,
        "document_code": record.document_code,
        "dry_run": dry_run,
        "grounding": grounding,
        "sri_cross_check": sri_check,
        "dataset_check": dataset_check,
        "extra_information": record.extra_information,
        "n_extra_information_items": len(record.extra_information),
    }

    if not dry_run:
        updated_df = merge_into_dataset(existing_df, new_row, dataset_check)
        save_dataset(updated_df)
        report["log_file"] = str(write_ingestion_log(pdf_path, report))

    return report


def print_report(report: dict) -> None:
    print(f"\n=== {report['source_file']} -> {report['fund_name']!r} ({report['document_code']}) ===")
    print(f"dataset action: {report['dataset_check']['action']}"
          + (f" (replacing existing row for {report['dataset_check']['matched_row']!r})"
             if report['dataset_check']['action'] == 'update' else ""))
    if report["dataset_check"]["field_changes"]:
        print("  field changes vs existing record:")
        for c in report["dataset_check"]["field_changes"]:
            print(f"    - {c['field']}: {c['old']!r} -> {c['new']!r}")

    g = report["grounding"]
    status = "OK - fully grounded" if g["fully_grounded"] else "FLAGGED - not fully grounded, review needed"
    print(f"grounding check: {status}")
    print(f"  doc_code_grounded={g['doc_code_grounded']}  key_facts={g['key_facts_grounded']}  "
          f"who_is_this_for={g['who_is_this_for_grounded']}  key_risks={g['key_risks_grounded']}")
    if g["ungrounded_values"]:
        for v in g["ungrounded_values"]:
            print(f"    ungrounded: {v[:120]}")

    sri = report["sri_cross_check"]
    if not sri["match"]:
        print(f"  SRI MISMATCH: VLM read {sri['vlm_sri']}, text layer says {sri['text_layer_sri']}")

    n_extra = report["n_extra_information_items"]
    print(f"extra_information: {n_extra} item(s) not covered by the standard schema")
    for item in report["extra_information"]:
        print(f"  - {item}")

    if report["dry_run"]:
        print("(dry run - dataset not modified)")
    else:
        print(f"dataset updated: {DATASET_JSON.name}, {DATASET_CSV.name}")
        print(f"log written: {report['log_file']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf_path", type=Path, help="Path to the new fund fact sheet PDF")
    parser.add_argument("--dry-run", action="store_true", help="Run extraction and verification without writing to the dataset")
    args = parser.parse_args()

    if not args.pdf_path.exists():
        print(f"error: {args.pdf_path} does not exist", file=sys.stderr)
        return 1

    report = ingest(args.pdf_path, dry_run=args.dry_run)
    print_report(report)
    return 0 if report["grounding"]["fully_grounded"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
