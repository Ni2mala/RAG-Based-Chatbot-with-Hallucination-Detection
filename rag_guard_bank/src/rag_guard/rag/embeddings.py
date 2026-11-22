from typing import Sequence
import ollama

from rag_guard.core.config import settings
from rag_guard.core.retry import retry_once

DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "
BATCH_SIZE = 32


class Embedder:
    def __init__(self, model: str | None = None, host: str | None = None) -> None:
        self.model = model or settings.EMBEDDING_MODEL
        self._client = ollama.Client(
            host=host or settings.OLLAMA_HOST, timeout=settings.OLLAMA_TIMEOUT_S
        )

    def embed_query(self, text: str) -> list[float]:
        return self._embed([QUERY_PREFIX + text])[0]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            vectors.extend(self._embed([DOC_PREFIX + t for t in batch]))
        return vectors

    def _embed(self, prompts: list[str]) -> list[list[float]]:
        resp = retry_once(
            lambda: self._client.embed(model=self.model, input=prompts), "Ollama embed"
        )
        return resp["embeddings"]
