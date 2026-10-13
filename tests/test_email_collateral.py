import os
from types import SimpleNamespace
import unittest

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from gtm_agent.gtm_agent import SYSTEM_PROMPT, send_prospect_email


class EmailCollateralTests(unittest.TestCase):
    def send_email(self, body):
        return send_prospect_email.func(
            prospect={"name": "Alex Prospect", "email": "alex@example.com"},
            subject="Pricing",
            body=body,
            runtime=SimpleNamespace(config={"metadata": {}}),
            from_rep={"name": "Sam Rep", "email": "sam@example.com"},
        )

    def test_rejects_body_claiming_deck_is_attached(self):
        result = self.send_email("Attached is our pricing deck.")

        self.assertEqual(
            result,
            {
                "status": "failed",
                "error": "Email body references an attachment but this tool cannot attach files.",
            },
        )

    def test_rejects_attachment_claim_case_insensitively(self):
        result = self.send_email("I HAVE ATTACHED THE PDF.")

        self.assertEqual(
            result,
            {
                "status": "failed",
                "error": "Email body references an attachment but this tool cannot attach files.",
            },
        )

    def test_sends_ordinary_email_copy(self):
        body = "Thanks for your time. I look forward to our conversation."
        result = self.send_email(body)

        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["body"], body)

    def test_sends_rep_provided_link_without_attachment_claim(self):
        body = "Here is a link to the pricing deck: https://example.com/pricing.pdf"
        result = self.send_email(body)

        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["body"], body)

    def test_prompt_requests_link_or_truthful_limitation_for_collateral(self):
        self.assertIn("cannot attach files or retrieve collateral", SYSTEM_PROMPT)
        self.assertIn("Include a link only if the rep provides one", SYSTEM_PROMPT)
        self.assertIn(
            "ask the rep for a link or explain that the document cannot be included",
            SYSTEM_PROMPT,
        )
        self.assertIn("never say or imply it is attached", SYSTEM_PROMPT)
        self.assertIn("Accurately describe what the tool actually sent", SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
