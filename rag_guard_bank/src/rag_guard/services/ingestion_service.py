import json

from rag_guard.core.config import settings
from rag_guard.core.logging import get_logger
from rag_guard.rag.chunking import chunk_text
from rag_guard.rag.embeddings import Embedder
from rag_guard.rag.vector_store import VectorStore

logger = get_logger(__name__)


def _load_rows() -> list[dict]:
    path = settings.DATA_RAW_DIR / "bitext_banking.jsonl"
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _deduplicate(rows: list[dict], per_intent: int = 5) -> list[dict]:
    by_intent: dict[str, list[dict]] = {}
    for row in rows:
        by_intent.setdefault(row["intent"], []).append(row)

    kept: list[dict] = []
    for group in by_intent.values():
        group.sort(key=lambda r: r["id"])
        n = min(per_intent, len(group))
        step = len(group) / n
        for i in sorted({int(i * step) for i in range(n)}):
            kept.append(group[i])
    return kept


def _build_documents(rows: list[dict]):
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []
    for idx, row in enumerate(rows):
        doc = f"Question: {row['request']}\nAnswer: {row['response']}"
        for i, piece in enumerate(chunk_text(doc)):
            ids.append(f"{row['intent']}-{idx}-c{i}")
            documents.append(piece)
            metadatas.append({"intent": row["intent"], "category": row["category"]})
    return ids, documents, metadatas


def rebuild_index(per_intent: int = 5, reset: bool = True) -> dict:
    rows = _load_rows()
    logger.info("Raw rows: %d", len(rows))

    deduped = _deduplicate(rows, per_intent)
    intents = sorted({r["intent"] for r in deduped})
    logger.info("Compact KB rows: %d across %d intents", len(deduped), len(intents))

    ids, documents, metadatas = _build_documents(deduped)
    logger.info("Chunks to embed: %d", len(documents))

    embedder = Embedder()
    logger.info("Embedding compact KB on CPU ...")
    embeddings = embedder.embed_documents(documents)
    logger.info("Embedding done.")

    store = VectorStore()
    if reset:
        store.reset()
    store.upsert(ids, documents, embeddings, metadatas)
    logger.info("Indexed %d chunks into '%s'", store.count(), settings.COLLECTION_NAME)

    return {
        "store": store,
        "embedder": embedder,
        "raw_rows": len(rows),
        "kb_rows": len(deduped),
        "chunks": len(documents),
    }
