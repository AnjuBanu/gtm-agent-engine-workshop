import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ.setdefault("LANGSMITH_TRACING", "false")

from gtm_agent import data_service
from gtm_agent import gtm_agent as agent_module
from gtm_agent.gtm_records import OFFERINGS, PROSPECTS


class UpdateProspectInfoTests(unittest.TestCase):
    prospect_id = "LEAD-39002"

    def setUp(self):
        self.original_tech_stack = list(PROSPECTS[self.prospect_id]["tech_stack"])
        self.had_cached_profile = self.prospect_id in data_service._PROFILES
        self.original_cached_profile = data_service._PROFILES.get(self.prospect_id)

    def tearDown(self):
        PROSPECTS[self.prospect_id]["tech_stack"] = self.original_tech_stack
        if self.had_cached_profile:
            data_service._PROFILES[self.prospect_id] = self.original_cached_profile
        else:
            data_service._PROFILES.pop(self.prospect_id, None)

    def test_update_persists_and_returns_technology(self):
        result = data_service.update_prospect_info(self.prospect_id, "Terraform")

        self.assertIn("Terraform", PROSPECTS[self.prospect_id]["tech_stack"])
        self.assertEqual(result["tech_stack"], PROSPECTS[self.prospect_id]["tech_stack"])

    def test_update_invalidates_cached_profile(self):
        original_profile = agent_module.build_prospect_profile.invoke(
            {"prospect_id": self.prospect_id}
        )["prospect_profile"]
        self.assertNotIn("Terraform", original_profile["tech_stack"])

        data_service.update_prospect_info(self.prospect_id, "Terraform")
        updated_profile = agent_module.build_prospect_profile.invoke(
            {"prospect_id": self.prospect_id}
        )["prospect_profile"]

        self.assertIn("Terraform", updated_profile["tech_stack"])

    def test_update_does_not_duplicate_existing_technology(self):
        data_service.update_prospect_info(self.prospect_id, "Terraform")

        result = data_service.update_prospect_info(self.prospect_id, "Terraform")

        self.assertEqual(result["tech_stack"].count("Terraform"), 1)

    def test_scoring_uses_updated_saved_technology(self):
        stale_profile = agent_module.build_prospect_profile.invoke(
            {"prospect_id": self.prospect_id}
        )["prospect_profile"]
        data_service.update_prospect_info(self.prospect_id, "Terraform")
        captured_messages = []

        def score(messages):
            captured_messages.extend(messages)
            return SimpleNamespace(
                model_dump=lambda: {
                    "score": 90,
                    "max_score": 100,
                    "justification": "Terraform is present in the prospect's tech stack.",
                    "rubric_breakdown": {
                        "revenue_fit": 30,
                        "tech_stack_match": 30,
                        "segment_fit": 30,
                        "component_max": 100,
                    },
                }
            )

        with patch.object(agent_module, "_scoring_llm") as scoring_llm:
            scoring_llm.invoke.side_effect = score
            result = agent_module.score_prospect.invoke(
                {
                    "prospect_profile": stale_profile,
                    "offering": OFFERINGS["OFFER-10004"],
                }
            )

        self.assertIn('"Terraform"', captured_messages[-1]["content"])
        self.assertNotIn("Terraform is missing", result["justification"])


if __name__ == "__main__":
    unittest.main()
