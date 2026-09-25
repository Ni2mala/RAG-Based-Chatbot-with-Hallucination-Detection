import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_guard.rag.retriever import Retriever
from rag_guard.rag.generator import Generator
from rag_guard.schemas.chat import ChatRequest
from rag_guard.services.chat_service import ChatService


def show(service: ChatService, question: str) -> None:
    t0 = time.time()
    response = service.answer(ChatRequest(question=question))
    seconds = time.time() - t0
    print("=" * 72)
    print(f"Q: {question}")
    print(f"A: {response.answer}")
    v = response.verification
    print(
        f"status={v.status} relevance={v.retrieval_relevance} "
        f"nli={v.nli_grounding} llm={v.llm_grounding} detector={v.detector}"
    )
    print(f"refused={response.refused} ({seconds:.1f}s)")


def main() -> None:
    service = ChatService(Retriever(), Generator())
    show(service, "How can I block my lost credit card?")
    show(service, "I want to send money to a friend, how long does a bank transfer take?")
    show(service, "What is the weather in Paris today?")


if __name__ == "__main__":
    main()