import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_guard.rag.retriever import Retriever
from rag_guard.rag.generator import Generator
from rag_guard.guardrails.claim_extractor import extract_claims
from rag_guard.guardrails.grounding_check import GroundingChecker


def main() -> None:
    retriever = Retriever()
    generator = Generator()
    checker = GroundingChecker()

    q = "How can I block my lost credit card?"
    chunks = retriever.retrieve(q, top_k=3)
    answer = generator.generate(q, [c.text for c in chunks])

    print("QUESTION:", q)
    print("ANSWER:", answer)

    claims = extract_claims(answer)
    print("\nCLAIMS extracted:")
    for c in claims:
        print("  -", c)

    results = checker.check(claims, [c.text for c in chunks])
    print("\nNLI grounding per claim:")
    for r in results:
        print(
            f"  supported={r['supported']} ({r['verdict']} {r['score']:.2f}) :: "
            f"{r['claim'][:70]}"
        )

    supported = sum(r["supported"] for r in results) / max(1, len(results))
    print(f"\nGrounding score: {supported:.2f}")


if __name__ == "__main__":
    main()
