import json
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test")

from gtm_agent import data_service
from gtm_agent.gtm_agent import build_prospect_profile, get_prospect, score_prospect

os.environ["LANGSMITH_TRACING"] = "false"


class ProspectDataSafetyTests(unittest.TestCase):
    def setUp(self):
        self.record = {
            "name": "Taylor Example",
            "email": "taylor@example.com",
            "annual_revenue": 12000000,
            "enrichment_source": "ExampleData",
            "disqualified": False,
            "billing_qualification": {
                "tax_id": "secret-tax-id",
                "date_of_birth": "1990-01-01",
                "card_on_file": "secret-card",
                "credit_check_ref": "secret-ref",
            },
            "internal_notes": "unlisted source data",
            "engagement_history": [{"channel": "email"}],
            "account_details": [{"account": "Example Co"}],
            "tech_stack": ["PostgreSQL", "AWS"],
        }
        self.prospect_id = "LEAD-TEST"

    def test_get_prospect_returns_only_permitted_contact_fields(self):
        with patch.object(data_service, "get_prospect_record", return_value=self.record):
            result = get_prospect.invoke({"prospect_id": self.prospect_id})

        self.assertEqual(result, {
            "prospect": {
                "prospect_id": self.prospect_id,
                "name": "Taylor Example",
                "email": "taylor@example.com",
                "disqualified": False,
            },
            "found": True,
        })

    def test_build_profile_and_cache_use_only_permitted_fields(self):
        profile = {
            "prospect_id": self.prospect_id,
            "name": self.record["name"],
            "email": self.record["email"],
            "annual_revenue": self.record["annual_revenue"],
            "enrichment_source": self.record["enrichment_source"],
            "disqualified": False,
            "engagement_history": self.record["engagement_history"],
            "account_details": self.record["account_details"],
            "tech_stack": self.record["tech_stack"],
        }
        with (
            patch.object(data_service, "get_profile_from_db", return_value={"prospect_profile": None}),
            patch.object(data_service, "get_prospect_record", return_value=self.record),
            patch.object(data_service, "fetch_engagement_history", return_value=profile["engagement_history"]),
            patch.object(data_service, "fetch_account_details", return_value=profile["account_details"]),
            patch.object(data_service, "fetch_tech_stack", return_value=profile["tech_stack"]),
            patch.object(data_service, "save_profile_to_db") as save_profile,
        ):
            result = build_prospect_profile.invoke({"prospect_id": self.prospect_id})

        self.assertEqual(result["prospect_profile"], profile)
        self.assertEqual(save_profile.call_args.args[1], profile)

        polluted = {**profile, "billing_qualification": self.record["billing_qualification"], "internal_notes": "secret"}
        with (
            patch.object(data_service, "get_profile_from_db", return_value={"prospect_profile": polluted}),
            patch.object(data_service, "save_profile_to_db") as save_profile,
        ):
            cached = build_prospect_profile.invoke({"prospect_id": self.prospect_id})

        self.assertEqual(cached["prospect_profile"], profile)
        self.assertEqual(save_profile.call_args.args[1], profile)

    def test_scoring_prompt_keeps_allowed_profile_and_drops_unlisted_data(self):
        profile = {
            "prospect_id": self.prospect_id,
            "name": self.record["name"],
            "email": self.record["email"],
            "annual_revenue": self.record["annual_revenue"],
            "enrichment_source": self.record["enrichment_source"],
            "disqualified": False,
            "engagement_history": self.record["engagement_history"],
            "account_details": self.record["account_details"],
            "tech_stack": self.record["tech_stack"],
            "billing_qualification": self.record["billing_qualification"],
            "internal_notes": "secret",
        }
        offering = {
            "required_tech_stack": ["AWS"],
            "min_annual_revenue": 1000000,
            "description": "Example offering",
        }
        response = type("ScoreResult", (), {"model_dump": lambda self: {"score": 80}})()
        with (
            patch.object(data_service, "fetch_tech_stack", return_value=profile["tech_stack"]),
            patch("gtm_agent.gtm_agent._scoring_llm") as scoring_llm,
        ):
            scoring_llm.invoke.return_value = response
            result = score_prospect.invoke({"prospect_profile": profile, "offering": offering})

        prompt = scoring_llm.invoke.call_args.args[0][1]["content"]
        scored_profile = json.loads(prompt.split("\n\nProspect profile:\n", 1)[1])
        self.assertEqual(result, {"score": 80})
        self.assertEqual(scored_profile, {
            "prospect_id": self.prospect_id,
            "name": self.record["name"],
            "email": self.record["email"],
            "annual_revenue": self.record["annual_revenue"],
            "enrichment_source": self.record["enrichment_source"],
            "disqualified": False,
            "engagement_history": self.record["engagement_history"],
            "account_details": self.record["account_details"],
            "tech_stack": self.record["tech_stack"],
        })

    def test_lookup_profile_scoring_flow_never_exposes_billing(self):
        profile = {
            "prospect_id": self.prospect_id,
            "name": self.record["name"],
            "email": self.record["email"],
            "annual_revenue": self.record["annual_revenue"],
            "enrichment_source": self.record["enrichment_source"],
            "disqualified": False,
            "engagement_history": self.record["engagement_history"],
            "account_details": self.record["account_details"],
            "tech_stack": self.record["tech_stack"],
        }
        offering = {
            "required_tech_stack": ["AWS"],
            "min_annual_revenue": 1000000,
            "description": "Example offering",
        }
        response = type("ScoreResult", (), {"model_dump": lambda self: {"score": 80}})()
        with (
            patch.object(data_service, "get_prospect_record", return_value=self.record),
            patch.object(data_service, "get_profile_from_db", return_value={"prospect_profile": None}),
            patch.object(data_service, "fetch_engagement_history", return_value=profile["engagement_history"]),
            patch.object(data_service, "fetch_account_details", return_value=profile["account_details"]),
            patch.object(data_service, "fetch_tech_stack", return_value=profile["tech_stack"]),
            patch.object(data_service, "save_profile_to_db"),
            patch("gtm_agent.gtm_agent._scoring_llm") as scoring_llm,
        ):
            scoring_llm.invoke.return_value = response
            contact_result = get_prospect.invoke({"prospect_id": self.prospect_id})
            profile_result = build_prospect_profile.invoke({"prospect_id": self.prospect_id})
            score_prospect.invoke({"prospect_profile": profile_result["prospect_profile"], "offering": offering})

        self.assertNotIn("billing_qualification", json.dumps(contact_result))
        self.assertNotIn("billing_qualification", json.dumps(profile_result))
        self.assertNotIn("secret-tax-id", scoring_llm.invoke.call_args.args[0][1]["content"])
        self.assertIn("Taylor Example", scoring_llm.invoke.call_args.args[0][1]["content"])


if __name__ == "__main__":
    unittest.main()
