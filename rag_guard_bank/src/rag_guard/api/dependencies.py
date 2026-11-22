from functools import lru_cache

from rag_guard.guardrails.hallucination_detector import HallucinationDetector
from rag_guard.rag.generator import Generator
from rag_guard.rag.retriever import Retriever


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever()


@lru_cache(maxsize=1)
def get_generator() -> Generator:
    return Generator()


@lru_cache(maxsize=1)
def get_detector() -> HallucinationDetector:
    # Cached so the NLI model loads once per process, not once per request.
    return HallucinationDetector()
