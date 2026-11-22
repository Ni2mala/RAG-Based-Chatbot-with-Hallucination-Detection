"""End to end through ChatService with a stub retriever and generator and the real NLI model (no Ollama).
Run: python tests/test_chat_sentence_filter.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_guard.guardrails.hallucination_detector import HallucinationDetector
from rag_guard.rag.retriever import RetrievedChunk
from rag_guard.schemas.chat import ChatRequest
from rag_guard.services.chat_service import ChatService

DOC = (
    "Question: how do i block my card\nAnswer: To block your lost card, open the mobile app. "
    "Go to the Cards section and choose the card. Tap the Block option and confirm your choice. "
    "You can also contact customer support to block it."
)
INVENTED = "A courier will collect the documents from your home address."


class StubRetriever:
    def retrieve(self, q, top_k=None):
        return [RetrievedChunk(text=DOC, intent="block_card", category="CARD", score=0.9)]


class StubGenerator:
    def __init__(self, answer): self.answer = answer
    def generate(self, q, ctx): return self.answer


def ask(answer):
    svc = ChatService(StubRetriever(), StubGenerator(answer), HallucinationDetector("nli"))
    return svc.answer(ChatRequest(question="How do I block my card?"))


def test_invented_sentence_is_cut_not_refused():
    r = ask("To block your lost card, open the mobile app. Tap the Block option and confirm your choice. " + INVENTED)
    assert not r.refused and INVENTED not in r.answer and "Tap the Block option" in r.answer, r
    assert r.verification.removed_claims == [INVENTED]
    assert r.verification.status == "partially_supported"


def test_clean_answer_is_untouched():
    text = "To block your lost card, open the mobile app. Tap the Block option and confirm your choice."
    r = ask(text)
    assert not r.refused and r.answer == text and r.verification.removed_claims == [], r


def test_answer_that_is_only_invented_is_refused():
    r = ask(INVENTED)
    assert r.refused, r


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
