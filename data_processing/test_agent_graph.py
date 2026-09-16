"""Unit tests for agent_graph.py's deterministic nodes (compliance_gate,
critic) as pure functions - no live OpenAI calls, matching the style of
test_agent_tools.py. node_agent/node_rewrite/the compiled graph itself
call the API and are exercised manually (see agent_graph.py's __main__),
not here.
"""

import unittest

from agent import AgentAnswer, Citation, EvidencePool
from agent_graph import node_compliance_gate, node_critic


class ComplianceGateTests(unittest.TestCase):
    def _state(self, pool, **overrides):
        state = {
            "input_list": [], "pool": pool, "tool_call_log": [], "checked_clients": set(),
            "draft": None, "critic_retries": 0, "original_question": "", "rewritten_question": "",
            "n_evidence_before": 0, "history": [], "pending_calls": [], "needs_rerun": False,
        }
        state.update(overrides)
        return state

    def test_injects_a_compliance_result_for_a_client_not_yet_checked(self):
        pool = EvidencePool()
        pool.add("client_CL002", "clients_portfolio.json (CL002)", "Robert Chua record")

        result = node_compliance_gate(self._state(pool))

        self.assertEqual(result["checked_clients"], {"CL002"})
        self.assertTrue(result["needs_rerun"])
        self.assertIsNotNone(pool.resolve("compliance_rules_CL002"))
        self.assertIn("FAIL", pool.resolve("compliance_rules_CL002")["text"])
        # the injected message is appended for the model to see next turn
        self.assertEqual(result["input_list"][-1]["role"], "user")
        self.assertIn("CL002", result["input_list"][-1]["content"][0]["text"])

    def test_does_not_reinject_or_rerun_if_model_already_called_the_tool(self):
        pool = EvidencePool()
        pool.add("client_CL011", "clients_portfolio.json (CL011)", "Park Ji-hoon record")
        pool.add("compliance_rules_CL011", "compliance_rules.py (POL-INV-011 S4, computed)", "already computed")

        result = node_compliance_gate(self._state(pool))

        self.assertEqual(result["checked_clients"], {"CL011"})
        self.assertFalse(result["needs_rerun"])
        self.assertEqual(result["input_list"], [])  # nothing appended - no redundant round-trip
        # the pre-existing pool entry must be untouched, not overwritten
        self.assertEqual(pool.resolve("compliance_rules_CL011")["text"], "already computed")

    def test_skips_clients_already_checked_this_run(self):
        pool = EvidencePool()
        pool.add("client_CL001", "clients_portfolio.json (CL001)", "Tan Wei Ling record")

        result = node_compliance_gate(self._state(pool, checked_clients={"CL001"}))

        self.assertFalse(result["needs_rerun"])
        self.assertIsNone(pool.resolve("compliance_rules_CL001"))

    def test_no_client_ids_in_pool_is_a_no_op(self):
        pool = EvidencePool()
        pool.add("factsheet_sql_APAC Stable Income Money Market Fund", "fund_factsheet_safe.pdf", "fund facts")

        result = node_compliance_gate(self._state(pool))

        self.assertEqual(result["checked_clients"], set())
        self.assertFalse(result["needs_rerun"])


class CriticTests(unittest.TestCase):
    def _state(self, pool, draft, critic_retries=0):
        return {
            "input_list": [{"role": "user", "content": "prior turn"}], "pool": pool, "tool_call_log": [],
            "checked_clients": set(), "draft": draft, "critic_retries": critic_retries,
            "original_question": "", "rewritten_question": "", "n_evidence_before": 0,
            "history": [], "pending_calls": [], "needs_rerun": False,
        }

    def test_valid_citations_pass_through_unchanged(self):
        pool = EvidencePool()
        pool.add("client_CL001", "clients_portfolio.json (CL001)", "text")
        draft = AgentAnswer(answer="ok", citations=[Citation(ref_id="client_CL001")], abstained=False)

        result = node_critic(self._state(pool, draft))

        self.assertEqual(result, {})  # no correction needed

    def test_invalid_ref_id_triggers_a_correction_retry(self):
        pool = EvidencePool()
        draft = AgentAnswer(answer="ok", citations=[Citation(ref_id="made_up_ref")], abstained=False)

        result = node_critic(self._state(pool, draft))

        self.assertIsNone(result["draft"])  # cleared so route_after_critic sends it back to "agent"
        self.assertEqual(result["critic_retries"], 1)
        self.assertIn("made_up_ref", result["input_list"][-1]["content"][0]["text"])

    def test_gives_up_after_max_retries_rather_than_looping_forever(self):
        pool = EvidencePool()
        draft = AgentAnswer(answer="ok", citations=[Citation(ref_id="made_up_ref")], abstained=False)

        result = node_critic(self._state(pool, draft, critic_retries=2))  # == MAX_CRITIC_RETRIES

        self.assertEqual(result, {})  # ships the answer as-is rather than looping indefinitely


if __name__ == "__main__":
    unittest.main()
