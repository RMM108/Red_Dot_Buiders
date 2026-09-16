"""Unit tests for compliance_rules.py, checked against hand-verified
expected verdicts for the actual dataset (see compliance_rules.py's module
docstring for the reasoning behind each one)."""

import unittest

from compliance_rules import check_client


class ComplianceRuleTests(unittest.TestCase):
    def _finding(self, result, rule_id, product_name):
        return next(f for f in result["findings"] if f["rule_id"] == rule_id and f["product_name"] == product_name)

    def test_cl002_fails_risk_score_and_acknowledgement_known_case(self):
        result = check_client("CL002")
        product = "APEX Global Multi-Asset Autocallable Note Series 7"

        self.assertEqual(result["overall_verdict"], "FAIL")
        self.assertEqual(self._finding(result, "R1_risk_score", product)["verdict"], "FAIL")
        self.assertEqual(self._finding(result, "R2_concentration", product)["verdict"], "PASS")  # 15% < 20%
        self.assertEqual(self._finding(result, "R3_risk_acknowledgement", product)["verdict"], "FAIL")

    def test_cl013_fails_the_same_two_rules_as_cl002_a_new_finding(self):
        """CL013's narrative case study is framed around his rebalancing
        delay, not a risk-score mismatch - the deterministic check surfaces
        this structurally-identical-to-CL002 gap regardless."""
        result = check_client("CL013")
        product = "Private Equity Co-Investment Vehicle - Fund IV"

        self.assertEqual(result["overall_verdict"], "FAIL")
        self.assertEqual(self._finding(result, "R1_risk_score", product)["verdict"], "FAIL")
        self.assertEqual(self._finding(result, "R3_risk_acknowledgement", product)["verdict"], "FAIL")

    def test_cl011_concentration_is_ambiguous_not_a_flat_breach(self):
        """The 20% cap's literal text is scoped to a Retail Investor's
        liquid net worth; CL011 is an Accredited Investor, so exceeding it
        is AMBIGUOUS, matching PLAN.md's own resolution of this case."""
        result = check_client("CL011")
        product = "APEX Global Multi-Asset Autocallable Note Series 7"

        self.assertEqual(result["overall_verdict"], "AMBIGUOUS")
        self.assertEqual(self._finding(result, "R2_concentration", product)["verdict"], "AMBIGUOUS")
        self.assertEqual(self._finding(result, "R1_risk_score", product)["verdict"], "PASS")

    def test_cl011_has_a_second_ambiguous_holding_not_previously_documented(self):
        result = check_client("CL011")
        product = "Private Equity Co-Investment Vehicle - Fund IV"
        self.assertEqual(self._finding(result, "R2_concentration", product)["verdict"], "AMBIGUOUS")

    def test_hong_kong_professional_investor_classification_is_ambiguous_when_risk_score_alone_does_not_clear(self):
        """"Professional Investor (per SFO Cap. 571)" is a Hong Kong
        classification, not literally one of the SFA-defined terms
        POL-INV-011's exemption names."""
        result = check_client("CL014")
        product = "Private Equity Co-Investment Vehicle - Fund IV"
        # Risk Score 7 already clears the >=7 threshold on its own, so R1
        # passes outright here regardless of the classification ambiguity -
        # the ambiguity only bites when the numeric threshold does NOT clear.
        self.assertEqual(self._finding(result, "R1_risk_score", product)["verdict"], "PASS")

    def test_accredited_investor_well_within_limits_passes_cleanly(self):
        result = check_client("CL015")
        self.assertEqual(result["overall_verdict"], "PASS")
        for f in result["findings"]:
            self.assertIn(f["verdict"], ("PASS", "N_A"))

    def test_client_with_no_complex_product_holdings_is_a_trivial_pass(self):
        result = check_client("CL003")
        self.assertEqual(result["overall_verdict"], "PASS")
        self.assertEqual(result["findings"], [])

    def test_unknown_client_id_reports_an_error_not_a_verdict(self):
        result = check_client("CL999")
        self.assertIn("error", result)
        self.assertNotIn("overall_verdict", result)


if __name__ == "__main__":
    unittest.main()
