from rag_guard.core.config import settings
from rag_guard.rag.retriever import Retriever
from rag_guard.rag.generator import Generator
from rag_guard.schemas.chat import ChatRequest, ChatResponse, Source, Verification
from rag_guard.guardrails.hallucination_detector import HallucinationDetector
from rag_guard.guardrails.refusal import refusal_message


class ChatService:
    def __init__(self, retriever: Retriever, generator: Generator) -> None:
        self._retriever = retriever
        self._generator = generator
        self._detector = HallucinationDetector()

    def answer(self, request: ChatRequest) -> ChatResponse:
        chunks = self._retriever.retrieve(request.question)
        max_relevance = max((c.score for c in chunks), default=0.0)
        sources = [
            Source(intent=c.intent, category=c.category, score=round(c.score, 3))
            for c in chunks
        ]

        if max_relevance < settings.RELEVANCE_MIN_SCORE:
            return ChatResponse(
                answer=refusal_message(),
                sources=sources,
                verification=Verification(
                    retrieval_relevance=round(max_relevance, 3),
                    status="insufficient_evidence",
                ),
                refused=True,
            )

        answer = self._generator.generate(request.question, [c.text for c in chunks])
        detection = self._detector.detect(
            answer, [c.text for c in chunks], mode=request.mode
        )

        refused = detection.status == "unsupported"
        return ChatResponse(
            answer=(
                refusal_message(detection.unsupported_claims)
                if refused
                else answer
            ),
            sources=sources,
            verification=Verification(
                retrieval_relevance=round(max_relevance, 3),
                status=detection.status,
                nli_grounding=(
                    round(detection.nli_grounding, 3)
                    if detection.nli_grounding is not None
                    else None
                ),
                llm_grounding=(
                    round(detection.llm_grounding, 3)
                    if detection.llm_grounding is not None
                    else None
                ),
                unsupported_claims=detection.unsupported_claims,
                detector=detection.detector,
            ),
            refused=refused,
        )