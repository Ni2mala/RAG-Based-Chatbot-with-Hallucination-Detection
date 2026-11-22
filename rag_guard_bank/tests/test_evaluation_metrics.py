"""Run: python tests/test_evaluation_metrics.py (needs the project venv)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_evaluation as ev


def _mode(refused, status="grounded"):
    return {"refused": refused, "status": status, "nli_grounding": None, "llm_grounding": None,
            "unsupported_claims": [], "latency": 1.0}


def _row(i, cat, answerable, refused_by_mode, answer="a"):
    return {"id": i, "category": cat, "expected_grounded": answerable, "answer": answer,
            "modes": {m: _mode(refused_by_mode.get(m, False)) for m in ev.MODES}}


ROWS = [
    _row(1, "in_domain", True, {}),                                   # answered everywhere
    _row(2, "in_domain", True, {"nli": True, "both": True}),          # false refusal for nli, both
    _row(3, "out_of_domain", False, {"nli": True, "llm": True, "both": True}),  # leaked only by none
    _row(4, "injection", False, {"llm": True, "both": True}),         # leaked by none and nli
]


def test_false_refusal_and_leak_rates():
    m = ev.compute_metrics(ROWS)
    assert (m["none"]["false_refusals"], m["none"]["leaks"]) == (0, 2)
    assert (m["nli"]["false_refusals"], m["nli"]["leaks"]) == (1, 1)
    assert (m["both"]["false_refusals"], m["both"]["leaks"]) == (1, 0)
    assert m["both"]["false_refusal_rate"] == 0.5 and m["none"]["leak_rate"] == 1.0
    assert m["none"]["leak_rate_by_category"] == {"out_of_domain": 1.0, "injection": 1.0}


def test_reference_quality_only_counts_delivered_answers():
    rows = [dict(r) for r in ROWS]
    rows[0].update(reference_similarity=0.9, control_similarity=0.6)    # on topic, delivered everywhere
    rows[1].update(reference_similarity=0.7, control_similarity=0.8)    # off topic; refused by nli and both
    q = ev.compute_metrics(rows)
    assert q["none"]["answer_quality"] == {"delivered": 2, "answers_with_sentences_removed": 0, "mean_reference_similarity": 0.8, "on_topic_rate": 0.5}
    assert q["both"]["answer_quality"] == {"delivered": 1, "answers_with_sentences_removed": 0, "mean_reference_similarity": 0.9, "on_topic_rate": 1.0}


if __name__ == "__main__":
    test_false_refusal_and_leak_rates()
    test_reference_quality_only_counts_delivered_answers()
    print("ok")
