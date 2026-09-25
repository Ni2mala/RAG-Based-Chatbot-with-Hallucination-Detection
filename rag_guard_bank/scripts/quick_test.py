import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_guard.rag.retriever import Retriever
from rag_guard.rag.generator import Generator


def main() -> None:
    retriever = Retriever()
    generator = Generator()

    questions = [
        "How can I block my lost credit card?",
        "What documents do I need to open a savings account?",
        "I want to send money abroad, how do I do it?",
        "Tell me about the weather in Paris tomorrow.",
    ]

    for q in questions:
        print("\n\n==== QUESTION:", q)
        chunks = retriever.retrieve(q, top_k=3)
        for c in chunks:
            snippet = c.text[:70].replace("\n", " ")
            print(f"  [sim={c.score:.3f}] {c.intent} :: {snippet}...")
        answer = generator.generate(q, [c.text for c in chunks])
        print("ANSWER:", answer)


if __name__ == "__main__":
    main()
