"""Needs torch + transformers (and the cached NLI model). Run: python tests/test_nli_windows.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_guard.guardrails.grounding_check import GroundingChecker

CONTEXT = (
    "Question: wanna set up a password where can i do it\n"
    "Answer: I can help you set up a password for your account. To do that, you'll need to follow "
    "these steps:\n\n1. Visit our website.\n2. Click on the \"Login\" button located at the top right "
    "corner of the homepage.\n3. Look for the \"Sign Up\" option and click on it.\n4. Create a strong "
    "password that includes uppercase and lowercase letters.\n5. Confirm your password by retyping it."
)
SUPPORTED = [
    "Click on the \"Login\" button located at the top right corner of the homepage.",
    "Confirm your password by retyping it.",
]
UNSUPPORTED = [
    "A fee of $25 applies to this service.",
    "Click on the \"Logout\" button located at the bottom left corner of the homepage.",
    "Your username must be your social security number.",
    "You must visit a branch in person with two forms of ID.",
]


def test_verbatim_claims_supported_and_fabrications_not():
    nli = GroundingChecker()
    got = {r["claim"]: r["supported"] for r in nli.check(SUPPORTED + UNSUPPORTED, [CONTEXT])}
    assert all(got[c] for c in SUPPORTED), got
    assert not any(got[c] for c in UNSUPPORTED), got


if __name__ == "__main__":
    test_verbatim_claims_supported_and_fabrications_not()
    print("ok")
