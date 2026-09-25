from dataclasses import dataclass

from rag_guard.core.config import settings
from rag_guard.rag.embeddings import Embedder
from rag_guard.rag.vector_store import VectorStore


@dataclass
class RetrievedChunk:
    text: str
    intent: str
    category: str
    score: float


class Retriever:
    def __init__(
        self,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
    ) -> None:
        self._embedder = embedder or Embedder()
        self._store = store or VectorStore()

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        k = top_k or settings.TOP_K
        qv = self._embedder.embed_query(query)
        res = self._store.collection.query(query_embeddings=[qv], n_results=k)
        chunks = []
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            chunks.append(
                RetrievedChunk(
                    text=doc,
                    intent=meta["intent"],
                    category=meta["category"],
                    score=1.0 - dist,
                )
            )
        return chunks
