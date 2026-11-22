"""Synthetic-fault test: how well do the detectors catch answers that are wrong BY CONSTRUCTION?

No human labels are used. The test cases are built from the dataset itself:

  good              the answer body of the top retrieved chunk, copied verbatim. Faithful to the
                    context by construction.
  wrong_topic       a real reply from the dataset for a *different* intent (fluent, but not
                    supported by this question's context).
  fabricated_claim  the good answer plus one invented, number-free policy sentence.
  fabricated_figure the good answer plus one invented sentence containing a figure (fee, days, limit).

A detector should accept `good` and refuse the three fault types. Note the limits: copied text is the
easy case for `good` (false alarms on natural paraphrases are measured by the false-refusal rate in
run_evaluation.py), and injected faults are easier than natural hallucinations.

Run: python scripts/fault_injection_test.py   (about 10-15 minutes)
"""
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_guard.core.config import PROJECT_ROOT, settings
from rag_guard.guardrails.figure_check import unsupported_figures
from rag_guard.guardrails.hallucination_detector import HallucinationDetector
from rag_guard.rag.retriever import Retriever

EVAL_FILE = settings.DATA_EVAL_DIR / "eval_set_v2.jsonl"
OUT_FILE = PROJECT_ROOT / "reports" / "fault_injection_results.json"
SEED = 42
KINDS = ["good", "wrong_topic", "fabricated_claim", "fabricated_figure"]
MODES = ["nli", "llm", "both"]

FABRICATED_CLAIMS = [
    "Customers must also submit a signed request letter at their home branch.",
    "This service is only available to customers with a premium account.",
    "A manager must personally approve the request before it can proceed.",
    "You will need to provide a notarized copy of your passport.",
    "The bank may suspend your other accounts while the request is processed.",
    "This option is not available on weekends or public holidays.",
    "You must first close all your other products with the bank.",
    "A courier will collect the documents from your home address.",
]
FABRICATED_FIGURES = [
    "A processing fee of $25 applies to this service.",
    "The request is completed within 3 business days.",
    "You are limited to 2 requests per year.",
    "Your daily limit will be reduced to $500.",
    "The service charge is 1.5% of the amount.",
    "A refund of $10 is credited to your account.",
]


def answer_body(chunk: str, max_chars: int = 600) -> str:
    body = re.sub(r"^Question:.*?\nAnswer:\s*", "", chunk, flags=re.DOTALL).strip()
    if len(body) > max_chars:
        cut = body[:max_chars]
        # End at a real sentence end, not at a list marker such as "6." (the lookbehind skips digits).
        ends = [m.end() for m in re.finditer(r"(?<!\d)[.!?](?=\s)", cut)]
        body = cut[: ends[-1]] if ends else cut
    return body


def build_cases(evals: list[dict], retriever: Retriever) -> list[dict]:
    golds = [(e["gold_intent"], e["gold_response"]) for e in evals]
    cases = []
    for q in evals:
        rng = random.Random(f"{SEED}-{q['id']}")
        contexts = [c.text for c in retriever.retrieve(q["question"])]
        good = answer_body(contexts[0])
        answers = {
            "good": good,
            "wrong_topic": rng.choice([g for g in golds if g[0] != q["gold_intent"]])[1],
            "fabricated_claim": f"{good} {rng.choice(FABRICATED_CLAIMS)}",
            "fabricated_figure": f"{good} {rng.choice(FABRICATED_FIGURES)}",
        }
        for kind in KINDS:
            cases.append({"qid": q["id"], "kind": kind, "answer": answers[kind], "contexts": contexts})
    return cases


def score_only_status(score: float) -> str:
    """The verdict from NLI/judge scores alone, ignoring the deterministic figure check."""
    if score >= settings.GROUNDING_THRESHOLD:
        return "grounded"
    return "partially_supported" if score >= settings.GROUNDING_THRESHOLD / 2 else "unsupported"


def summarize(results: list[dict]) -> dict:
    out: dict = {}
    for mode in MODES + ["figure_check_only"]:
        out[mode] = {}
        for kind in KINDS:
            rows = [r for r in results if r["kind"] == kind]
            if mode == "figure_check_only":
                refused = sum(bool(r["figures"]) for r in rows)
                flagged, scores_only = refused, None
            else:
                refused = sum(r[mode]["status"] == "unsupported" for r in rows)
                flagged = sum(r[mode]["status"] != "grounded" for r in rows)
                scores_only = sum(score_only_status(r[mode]["score"]) == "unsupported" for r in rows)
            out[mode][kind] = {
                "n": len(rows),
                "refused": refused, "refused_rate": round(refused / len(rows), 3),
                "not_grounded_rate": round(flagged / len(rows), 3),
                "refused_by_scores_only_rate": None if scores_only is None else round(scores_only / len(rows), 3),
            }
    return out


def main() -> None:
    lines = [l for l in EVAL_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    evals = [e for e in map(json.loads, lines) if e["category"] == "in_domain"]
    if len(sys.argv) > 1:  # optional smoke run: python scripts/fault_injection_test.py 5
        evals = evals[: int(sys.argv[1])]
    retriever, detector = Retriever(), HallucinationDetector()
    cases = build_cases(evals, retriever)
    print(f"{len(cases)} cases ({len(evals)} questions x {len(KINDS)} kinds)", flush=True)

    results, t0 = [], time.time()
    for i, c in enumerate(cases, 1):
        dets = detector.detect_modes(c["answer"], c["contexts"])
        results.append({
            "qid": c["qid"], "kind": c["kind"], "answer": c["answer"],
            "figures": unsupported_figures(c["answer"], c["contexts"]),
            **{m: {"status": d.status, "nli": d.nli_grounding, "llm": d.llm_grounding, "score": d.score}
               for m, d in dets.items()},
        })
        if i % 40 == 0:
            print(f"  {i}/{len(cases)}  {time.time() - t0:.0f}s", flush=True)

    summary = summarize(results)
    OUT_FILE.write_text(json.dumps({"seed": SEED, "summary": summary, "cases": results},
                                   ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nShare of answers REFUSED (status unsupported). good = false alarms (want low); faults = detection (want high)")
    print(f"{'detector':<20}" + "".join(f"{k:>19}" for k in KINDS))
    for mode in MODES + ["figure_check_only"]:
        print(f"{mode:<20}" + "".join(f"{summary[mode][k]['refused_rate']:>19.1%}" for k in KINDS))
    print("\nSame, but ignoring the figure check (NLI/judge scores alone):")
    for mode in MODES:
        print(f"{mode:<20}" + "".join(f"{summary[mode][k]['refused_by_scores_only_rate']:>19.1%}" for k in KINDS))
    print("\nShare NOT graded 'grounded' (stricter view):")
    for mode in MODES:
        print(f"{mode:<20}" + "".join(f"{summary[mode][k]['not_grounded_rate']:>19.1%}" for k in KINDS))
    print(f"\nWritten to {OUT_FILE}")


if __name__ == "__main__":
    main()
