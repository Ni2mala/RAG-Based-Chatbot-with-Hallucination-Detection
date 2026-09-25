from rag_guard.core.config import settings
from rag_guard.rag.retriever import Retriever
from rag_guard.rag.generator import Generator
from rag_guard.schemas.chat import ChatRequest, ChatResponse, Source, Verification

REFUSAL_MESSAGE = (
    "I could not find enough information in the approved banking documents "
    "to answer that reliably. Please rephrase your question or contact our "
    "customer support."
)


class ChatService:
    def __init__(self, retriever: Retriever, generator: Generator) -> None:
        self._retriever = retriever
        self._generator = generator

    def answer(self, request: ChatRequest) -> ChatResponse:
        chunks = self._retriever.retrieve(request.question)
        max_relevance = max((c.score for c in chunks), default=0.0)
        sources = [
            Source(intent=c.intent, category=c.category, score=round(c.score, 3))
            for c in chunks
        ]

        if max_relevance < settings.RELEVANCE_MIN_SCORE:
            return ChatResponse(
                answer=REFUSAL_MESSAGE,
                sources=sources,
                verification=Verification(
                    retrieval_relevance=round(max_relevance, 3),
                    status="insufficient_evidence",
                ),
                refused=True,
            )

        answer = self._generator.generate(request.question, [c.text for c in chunks])
        return ChatResponse(
            answer=answer,
            sources=sources,
            verification=Verification(
                retrieval_relevance=round(max_relevance, 3),
                status="grounded_pending_check",
            ),
            refused=False,
        )
