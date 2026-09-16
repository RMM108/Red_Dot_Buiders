import json
import unittest
from pathlib import Path

import db
from agent import (
    EvidencePool,
    tool_query_client_db,
    tool_query_portfolio_exposure,
    tool_query_transactions,
    tool_get_fund_factsheet,
)
from ingest_ack_forms import extract_text as extract_ack_text
from ingest_ack_forms import split_records
from ingest_call_notes import extract_text as extract_call_text
from ingest_call_notes import split_entries
from ingest_complaints import extract_text as extract_complaint_text
from ingest_complaints import split_letters
from ingest_policies import extract_text as extract_policy_text
from ingest_policies import split_into_sections
from run_evaluation import mechanical_checks


ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"


class AgentToolTests(unittest.TestCase):
    def setUp(self):
        self.pool = EvidencePool()

    def test_client_tool_returns_grounded_cl002_record(self):
        result = json.loads(tool_query_client_db(self.pool, "CL002"))

        self.assertEqual(result["ref_id"], "client_CL002")
        self.assertIn("risk_score=2/10", result["content"])
        self.assertIn("APEX Global Multi-Asset Autocallable Note Series 7", result["content"])
        self.assertEqual(self.pool.resolve("client_CL002")["locator"], "clients_portfolio.json (CL002)")

    def test_transaction_tool_preserves_pending_rebalance_state(self):
        result = json.loads(tool_query_transactions(self.pool, "CL013", "Pending"))

        self.assertEqual(result["ref_id"], "transactions_CL013")
        self.assertIn("Pending", result["content"])
        self.assertIn("TXN-1047", result["content"])

    def test_exposure_tool_finds_complex_products_above_twenty_percent(self):
        result = json.loads(tool_query_portfolio_exposure(self.pool, threshold_pct=20))

        self.assertEqual(result["ref_id"], "portfolio_exposure_20")
        for client_id in ("CL001", "CL009", "CL011"):
            self.assertIn(client_id, result["content"])
        self.assertNotIn("CL002", result["content"])
        self.assertEqual(
            self.pool.resolve("portfolio_exposure_20")["locator"],
            "clients_portfolio.csv (complex-product concentration)",
        )

    def test_exposure_tool_can_filter_retail_investors(self):
        result = json.loads(
            tool_query_portfolio_exposure(
                self.pool, threshold_pct=20, investor_status="Retail Investor"
            )
        )

        self.assertIn("No Complex Product holdings exceed", result["content"])

    def test_ambiguous_surname_is_not_resolved_by_guessing(self):
        result = json.loads(tool_query_client_db(self.pool, "Rahman"))

        self.assertEqual(result["error"], "ambiguous client name")
        self.assertEqual(
            {candidate["client_id"] for candidate in result["candidates"]},
            {"CL005", "CL006"},
        )

    def test_missing_product_factsheet_is_explicit(self):
        result = json.loads(
            tool_get_fund_factsheet(self.pool, "Singapore Government Bond Fund")
        )

        self.assertIn("no fact sheet on file", result["error"])

    def test_cl008_record_preserves_dci_documentation_gap(self):
        result = json.loads(tool_query_client_db(self.pool, "CL008"))

        self.assertIn("Retail Investor", result["content"])
        self.assertIn("Dual Currency Investment", result["content"])
        self.assertIn("Customer Account Review / Customer Knowledge Assessment", result["content"])

    def test_cl011_record_preserves_ambiguous_investor_scope(self):
        result = json.loads(tool_query_client_db(self.pool, "CL011"))

        self.assertIn("Accredited Investor", result["content"])
        self.assertIn("35.0%", result["content"])
        self.assertIn("20% concentration guideline", result["content"])

    def test_cl014_record_preserves_pep_adjacent_narrative(self):
        result = json.loads(tool_query_client_db(self.pool, "CL014"))

        self.assertIn("PEP-adjacent", result["content"])
        self.assertIn("annual PEP monitoring", result["content"])

    def test_cl015_transaction_tool_preserves_pending_remittance(self):
        result = json.loads(tool_query_transactions(self.pool, "CL015", "Pending"))

        self.assertIn("TXN-1055", result["content"])
        self.assertIn("LRS remittance", result["content"])


class IngestionBoundaryTests(unittest.TestCase):
    def test_call_notes_are_split_into_ten_client_records(self):
        entries = split_entries(extract_call_text(DATA / "rm_call_notes_log.pdf"))

        self.assertEqual(len(entries), 10)
        self.assertEqual({entry["client_id"] for entry in entries}, {
            "CL001", "CL002", "CL004", "CL006", "CL008", "CL009", "CL011", "CL013", "CL014", "CL015"
        })
        self.assertTrue(all(entry["date"] for entry in entries))

    def test_complaints_are_split_and_client_tagged(self):
        letters = split_letters(extract_complaint_text(DATA / "client_complaint_letters.pdf"))

        self.assertEqual(len(letters), 2)
        self.assertEqual(
            {(letter["complaint_ref"], letter["client_id"]) for letter in letters},
            {("CPL-2026-014", "CL002"), ("CPL-2026-057", "CL013")},
        )

    def test_acknowledgement_template_is_not_signed_evidence(self):
        records = split_records(
            extract_ack_text(DATA / "complex_product_risk_acknowledgement_forms.pdf")
        )

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["client_id"], "CL001")
        self.assertFalse(records[0]["is_blank_template"])
        self.assertTrue(records[1]["is_blank_template"])
        self.assertIsNone(records[1]["client_id"])

    def test_policy_sections_include_the_twenty_percent_rule(self):
        text = extract_policy_text(DATA / "policy_investment_suitability.pdf")
        sections = split_into_sections(text)

        concentration_sections = [section for section in sections if "20%" in section["text"]]
        self.assertTrue(concentration_sections)
        self.assertTrue(any(section["section_number"] == "4" for section in concentration_sections))


class DatabaseInventoryTests(unittest.TestCase):
    def test_rebuilt_database_matches_plan_inventory(self):
        conn = db.get_connection()
        try:
            counts = {
                table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in (
                    "clients", "holdings", "transactions", "funds",
                    "fund_key_facts", "fund_asset_allocation", "product_crosswalk",
                )
            }
            gap = conn.execute(
                "SELECT product_name FROM product_crosswalk WHERE has_factsheet = 0"
            ).fetchall()
        finally:
            conn.close()

        self.assertEqual(counts, {
            "clients": 15,
            "holdings": 70,
            "transactions": 55,
            "funds": 10,
            "fund_key_facts": 127,
            "fund_asset_allocation": 27,
            "product_crosswalk": 11,
        })
        self.assertEqual([row[0] for row in gap], ["Singapore Government Bond Fund"])


class EvaluationContractTests(unittest.TestCase):
    def test_mechanical_checks_require_sources_and_answer_facts(self):
        result = {
            "answer": "The situation is ambiguous: USD 40,000 remains, and USD 60,000 would exceed it by USD 20,000.",
            "citations": [
                {"ref_id": "policy", "locator": "Investment Suitability Policy POL-INV-011 Section 4"},
                {"ref_id": "email", "locator": "client_correspondence.json (EML-004)"},
            ],
        }

        checks = mechanical_checks(
            result,
            "policy_investment_suitability.pdf (Section 4); client_correspondence.json (EML-004)",
            ("ambiguous", "40,000", "60,000", "20,000"),
        )

        self.assertTrue(checks["passed"])
        self.assertEqual(checks["missing_sources"], [])

    def test_mechanical_checks_reject_invalid_or_missing_evidence(self):
        result = {
            "answer": "It may require review.",
            "citations": [
                {"ref_id": "bad", "locator": "INVALID ref_id (model cited a source it never received)"},
            ],
        }

        checks = mechanical_checks(result, "policy_investment_suitability.pdf (Section 4)", ("ambiguous",))

        self.assertFalse(checks["passed"])
        self.assertEqual(checks["invalid_refs"], ["bad"])
        self.assertEqual(checks["missing_fragments"], ["ambiguous"])


if __name__ == "__main__":
    unittest.main()