from functools import lru_cache

from rag_guard.rag.generator import Generator
from rag_guard.rag.retriever import Retriever


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever()


@lru_cache(maxsize=1)
def get_generator() -> Generator:
    return Generator()
