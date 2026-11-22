"""Needs torch + transformers installed. Run: python tests/test_robustness.py (or pytest tests/)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient

from rag_guard.api.dependencies import get_retriever
from rag_guard.core.config import Settings
from rag_guard.guardrails.hallucination_detector import HallucinationDetector
from rag_guard.guardrails.refusal import refusal_message
from rag_guard.main import app


class FakeNLI:
    def check(self, claims, ctx):
        return [{"claim": c, "supported": True} for c in claims]


class FakeJudge:
    def judge(self, answer, ctx):
        return {"score": 0.9, "unsupported_claims": []}


def _detector(mode):
    d = HallucinationDetector(mode)
    d._nli, d._judge = FakeNLI(), FakeJudge()
    d._ensure_loaded = lambda m: None
    return d


def test_refusal_does_not_quote_model_text():
    msg = refusal_message(["Yer lookin' fer yer card limit, eh?", "Figures not found: 0"])
    assert "Yer lookin" not in msg and "Figures" not in msg


def test_invalid_detector_mode_fails_loudly():
    for bad in ("banana", "bth"):
        try:
            HallucinationDetector(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("HallucinationDetector accepted " + bad)
    try:
        Settings(DETECTOR_MODE="banana")
    except ValueError:
        pass
    else:
        raise AssertionError("Settings accepted a bad DETECTOR_MODE")
    assert Settings(DETECTOR_MODE="BOTH").DETECTOR_MODE == "both"


def test_empty_claim_list_not_scored_as_zero_support():
    short = "Yes."  # no claim longer than 8 characters
    both = _detector("both").detect(short, ["ctx"])
    assert both.nli_grounding is None and both.llm_grounding == 0.9 and both.status == "grounded"
    nli_only = _detector("nli").detect(short, ["ctx"])
    assert nli_only.status == "unsupported"  # nothing could verify it, so it is still refused


def test_model_backend_down_returns_503_not_500():
    class DownRetriever:
        def retrieve(self, *a, **k):
            raise ConnectionError("Failed to connect to Ollama")

    app.dependency_overrides[get_retriever] = lambda: DownRetriever()
    try:
        r = TestClient(app, raise_server_exceptions=False).post("/chat", json={"question": "hello bank"})
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 503 and "unavailable" in r.json()["detail"], (r.status_code, r.text)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
