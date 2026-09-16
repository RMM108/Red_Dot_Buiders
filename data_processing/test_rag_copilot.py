import unittest
from pathlib import Path

from persona import (
    generate_client_persona,
    generate_personas,
    lookup_client_persona,
)


ROOT = Path(__file__).parent.parent


class CopilotTests(unittest.TestCase):
    def test_persona_apis_accept_repository_or_data_root(self):
        repository_result = lookup_client_persona(ROOT, "CL015")
        data_result = lookup_client_persona(ROOT / "data", "CL015")

        self.assertEqual(repository_result, data_result)

    def test_generates_conservative_persona_from_client_characteristics(self):
        result = generate_client_persona(
            ROOT,
            {
                "age": 68,
                "marital_status": "Married",
                "net_worth_band": "Mass Affluent",
                "aum_sgd": 850000,
                "source_of_wealth": "Pension payout and CPF savings",
                "investment_objective": "Capital preservation and stable income for retirement",
            },
            top_k=3,
        )

        self.assertEqual(result["matched_clients"][0]["client_id"], "CL002")
        self.assertEqual(result["matched_client_risk_profiles"][0]["recorded_profile"], "Conservative")
        self.assertNotIn("expected_risk_profile", result)
        self.assertEqual(
            result["favoured_portfolios"][0]["name"],
            "APAC Stable Income Money Market Fund",
        )
        self.assertFalse(any("AUM in SGD" in item for item in result["limitations"]))

    def test_generates_aggressive_persona_from_client_characteristics(self):
        result = generate_client_persona(
            ROOT,
            {
                "age": 34,
                "marital_status": "Single",
                "net_worth_band": "High Net Worth (HNW)",
                "source_of_wealth": "Sale of equity stake in previous startup",
                "investment_objective": "Long-term capital growth with high volatility and illiquidity",
            },
            top_k=3,
        )

        self.assertEqual(result["matched_clients"][0]["client_id"], "CL001")
        self.assertEqual(result["matched_client_risk_profiles"][0]["recorded_score"], 9)
        self.assertNotIn("expected_risk_score", result)
        self.assertEqual(
            result["favoured_portfolios"][0]["name"],
            "Pacific Growth Equity Fund",
        )

    def test_looks_up_persona_by_exact_client_id(self):
        result = lookup_client_persona(ROOT, "CL001")

        self.assertEqual(result["client_id"], "CL001")
        self.assertEqual(result["expected_risk_profile"], "Aggressive Growth")
        self.assertEqual(result["expected_risk_score"], 9)
        self.assertEqual(result["characteristics"]["investor_status"], "Accredited Investor (SFA definition met)")
        self.assertEqual(result["client_record"]["pep_status"], "Not a PEP")
        self.assertEqual(result["current_profile"]["current_risk_profile"], "Aggressive Growth")
        self.assertEqual(
            result["favoured_portfolios"][0]["name"],
            "Pacific Growth Equity Fund",
        )
        self.assertIn("advisor_view", result)
        self.assertIn(
            "Do not infer a product recommendation",
            " ".join(result["advisor_view"]["next_steps"]),
        )

    def test_advisor_view_surfaces_client_specific_review_items(self):
        cl002 = lookup_client_persona(ROOT, "CL002")["advisor_view"]
        cl013 = lookup_client_persona(ROOT, "CL013")["advisor_view"]
        cl014 = lookup_client_persona(ROOT, "CL014")["advisor_view"]

        self.assertTrue(any(item["type"] == "suitability_review" for item in cl002["attention_items"]))
        self.assertTrue(any(item["type"] == "pending_action" for item in cl013["attention_items"]))
        self.assertTrue(any(item["type"] == "kyc_monitoring" for item in cl014["attention_items"]))
        self.assertIn("not investment", cl002["scope_note"])

    def test_current_profile_preserves_recorded_risk_and_surfaces_de_risk_signal(self):
        result = lookup_client_persona(ROOT, "CL013")

        current = result["current_profile"]
        self.assertEqual(current["current_risk_profile"], "Conservative")
        self.assertEqual(current["current_risk_score"], 3)
        self.assertIn("recorded values", current["risk_profile_basis"])
        self.assertTrue(current["pending_actions"])
        self.assertIn("retirement", current["current_objective"].casefold())

    def test_current_profile_includes_transaction_and_correspondence_activity(self):
        result = lookup_client_persona(ROOT, "CL015")

        current = result["current_profile"]
        self.assertEqual(current["pending_actions"][0]["transaction_id"], "TXN-1055")
        self.assertTrue(current["correspondence_events"])
        self.assertIn("Cash (Investment Top-Up)", current["product_activity"])

    def test_generated_personas_preserve_each_client_record(self):
        personas = generate_personas(ROOT)

        self.assertEqual(len(personas), 15)
        for persona in personas:
            self.assertEqual(persona["client_id"], persona["client_record"]["client_id"])
            self.assertEqual(
                persona["expected_risk_profile"],
                persona["client_record"]["risk_profile"].split("(", 1)[0].strip(),
            )

    def test_client_id_lookup_rejects_unknown_client(self):
        with self.assertRaisesRegex(ValueError, "Client ID not found: CL999"):
            lookup_client_persona(ROOT, "CL999")


if __name__ == "__main__":
    unittest.main()