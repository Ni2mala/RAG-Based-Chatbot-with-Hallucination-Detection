"""No Ollama or torch needed. Run: python tests/test_answer_filter.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_guard.guardrails.answer_filter import remove_claims


def test_removes_sentence_and_keeps_the_rest():
    a = "You can block the card in the app. A manager must approve it. Then confirm."
    assert remove_claims(a, ["A manager must approve it."]) == "You can block the card in the app. Then confirm."


def test_removed_step_is_dropped_and_steps_renumbered():
    a = "Follow these steps:\n\n1. Open the app.\n2. Send a notarized letter.\n3. Tap Cards.\n4. Confirm."
    out = remove_claims(a, ["Send a notarized letter."])
    assert out == "Follow these steps:\n1. Open the app.\n2. Tap Cards.\n3. Confirm."


def test_second_list_restarts_numbering():
    a = "1. One.\n2. Two.\n\nAlso:\n1. Three.\n2. Bogus step.\n3. Four."
    assert remove_claims(a, ["Bogus step."]) == "1. One.\n2. Two.\nAlso:\n1. Three.\n2. Four."


def test_everything_removed_leaves_empty():
    assert remove_claims("Only invented.", ["Only invented."]) == ""


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
