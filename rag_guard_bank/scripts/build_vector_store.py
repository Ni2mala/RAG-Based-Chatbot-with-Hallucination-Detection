import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_guard.services.ingestion_service import rebuild_index


def main() -> None:
    result = rebuild_index()
    store = result["store"]
    embedder = result["embedder"]

    probes = [
        "How can I block my lost credit card?",
        "What do I need to open a bank account?",
        "I want to send money to a friend.",
    ]
    for q in probes:
        qv = embedder.embed_query(q)
        res = store.collection.query(query_embeddings=[qv], n_results=3)
        print("\nQ:", q)
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            sim = 1.0 - dist
            snippet = doc[:80].replace("\n", " ")
            print(f"  [sim={sim:.3f}] {meta['intent']} :: {snippet}...")


if __name__ == "__main__":
    main()
