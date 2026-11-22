import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rag_guard.core.config import PROJECT_ROOT, settings
from rag_guard.rag.embeddings import Embedder
from rag_guard.rag.retriever import Retriever
from rag_guard.rag.generator import Generator
from rag_guard.guardrails.hallucination_detector import HallucinationDetector
from rag_guard.rag.prompts import SYSTEM_PROMPT
from rag_guard.services.chat_service import is_self_refusal

MODES = ["none", "nli", "llm", "both"]
EVAL_FILE = settings.DATA_EVAL_DIR / "eval_set_v2.jsonl"
REPORTS_DIR = PROJECT_ROOT / "reports"
RESULTS_FILE = REPORTS_DIR / "evaluation_results_v2.json"


def load_evals() -> list[dict]:
    with EVAL_FILE.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def run_config() -> dict:
    """Everything that changes the results. A resumed run must match it exactly."""
    return {
        "llm_model": settings.LLM_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
        "nli_model": settings.NLI_MODEL,
        "top_k": settings.TOP_K,
        "relevance_min_score": settings.RELEVANCE_MIN_SCORE,
        "grounding_threshold": settings.GROUNDING_THRESHOLD,
        "retry_min_relevance": settings.RETRY_MIN_RELEVANCE,
        "retry_top_k": settings.RETRY_TOP_K,
        "generation_seed": settings.GENERATION_SEED,
        "system_prompt_sha": _sha(SYSTEM_PROMPT),
        "eval_file": EVAL_FILE.name,
        "eval_file_sha": _sha(EVAL_FILE.read_text(encoding="utf-8")),
    }


def load_previous() -> dict | None:
    if not RESULTS_FILE.exists():
        return None
    with RESULTS_FILE.open("r", encoding="utf-8") as fh:
        return json.load(fh)


class ReferenceScorer:
    """Compares an answer with the dataset's own reference reply (no human labels needed).

    similarity: cosine between the answer and the reference reply of the same held-out row.
    control:    cosine between the answer and a reply of a *different* intent (fixed per question).
    An answer that is on topic should be closer to its own reference than to the control.
    """

    def __init__(self, evals: list[dict]) -> None:
        self._emb = Embedder()
        self._golds = [(e["gold_intent"], e["gold_response"]) for e in evals if e.get("gold_response")]

    def score(self, q: dict, answer: str) -> tuple[float, float]:
        rng = random.Random(q["id"])
        control = rng.choice([g for g in self._golds if g[0] != q["gold_intent"]])[1]
        a, ref, ctl = self._emb.embed_documents([answer, q["gold_response"], control])
        cos = lambda x, y: sum(i * j for i, j in zip(x, y)) / math.sqrt(sum(i * i for i in x) * sum(j * j for j in y))
        return round(cos(a, ref), 4), round(cos(a, ctl), 4)


def _row(q: dict, max_rel: float, sources: list, self_ref: bool | None, modes: dict,
         answer: str | None = None, contexts: list[str] | None = None,
         reference: tuple[float, float] | None = None) -> dict:
    return {
        "reference_similarity": reference[0] if reference else None,
        "control_similarity": reference[1] if reference else None,
        "answer": answer,       # what the model generated (before any refusal replaces it)
        "contexts": contexts,   # the chunks it was given, so answers can be audited
        "id": q["id"],
        "question": q["question"],
        "category": q["category"],
        "expected_action": q["expected_action"],
        "expected_grounded": q["expected_grounded"],
        "max_relevance": round(max_rel, 3),
        "sources": sources,
        "self_refusal": self_ref,
        "modes": modes,
    }


def evaluate_q(retriever, generator, detector, q: dict, scorer: ReferenceScorer | None = None) -> dict:
    question = q["question"]
    t0 = time.time()
    chunks = retriever.retrieve(question)
    retrieval_s = time.time() - t0
    max_rel = max((c.score for c in chunks), default=0.0)
    contexts = [c.text for c in chunks]
    sources = [
        {"intent": c.intent, "category": c.category, "score": round(c.score, 3)}
        for c in chunks
    ]

    modes: dict[str, dict] = {}

    if max_rel < settings.RELEVANCE_MIN_SCORE:
        base = {
            "refused": True,
            "status": "insufficient_evidence",
            "nli_grounding": None,
            "llm_grounding": None,
            "unsupported_claims": [],
            "latency": round(retrieval_s, 3),
        }
        modes = {m: dict(base) for m in MODES}
        # Same as production: the gate refusal ends the request, the generator never runs.
        return _row(q, max_rel, sources, None, modes, None, contexts)

    t1 = time.time()
    answer = generator.generate(question, contexts)
    if is_self_refusal(answer) and max_rel >= settings.RETRY_MIN_RELEVANCE:
        # Same retry as ChatService, so the evaluation matches production.
        chunks = retriever.retrieve(question, top_k=settings.RETRY_TOP_K)
        contexts = [c.text for c in chunks]
        answer = generator.generate(question, contexts)
    gen_s = time.time() - t1
    self_ref = is_self_refusal(answer)

    if self_ref:
        base = {
            "refused": True,
            "status": "insufficient_evidence",
            "nli_grounding": None,
            "llm_grounding": None,
            "unsupported_claims": [],
            "latency": round(retrieval_s + gen_s, 3),
        }
        modes = {m: dict(base) for m in MODES}
    else:
        dets = detector.detect_modes(answer, contexts)  # one NLI pass and one judge call, shared
        for mode in MODES:
            if mode == "none":
                det = None
                refused, status, nli, llm, unsupported, invented = False, "unguarded", None, None, [], []
                latency = retrieval_s + gen_s
            else:
                det = dets[mode]
                refused, status = det.status == "unsupported", det.status
                nli, llm, unsupported = det.nli_grounding, det.llm_grounding, det.unsupported_claims
                invented = det.invented_claims
                latency = retrieval_s + gen_s + det.seconds
            modes[mode] = {
                "refused": refused,
                "status": status,
                "nli_grounding": nli,
                "llm_grounding": llm,
                "unsupported_claims": unsupported,
                "invented_claims": invented,  # sentences ChatService would cut from the delivered answer
                "latency": round(latency, 3),
            }

    reference = None
    if scorer and q.get("gold_response") and not self_ref:
        reference = scorer.score(q, answer)
    return _row(q, max_rel, sources, self_ref, modes, answer, contexts, reference)


def compute_metrics(rows: list[dict]) -> dict:
    metrics: dict[str, dict] = {}
    for mode in MODES:
        tp = fp = tn = fn = 0
        refused_rates: dict[str, tuple[int, int]] = {}
        status_counts: dict[str, int] = {}
        latencies: list[float] = []
        for row in rows:
            gold_hall = not row["expected_grounded"]
            m = row["modes"][mode]
            pred = m["refused"]
            if gold_hall and pred:
                tp += 1
            elif gold_hall and not pred:
                fn += 1
            elif not gold_hall and pred:
                fp += 1
            else:
                tn += 1
            cat = row["category"]
            refusal_group = refused_rates.setdefault(cat, [0, 0])
            refusal_group[1] += 1
            if m["refused"]:
                refusal_group[0] += 1
            status_counts[m["status"]] = status_counts.get(m["status"], 0) + 1
            latencies.append(m["latency"])

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        accuracy = (tp + tn) / len(rows)

        answerable = [r for r in rows if r["expected_grounded"]]
        unanswerable = [r for r in rows if not r["expected_grounded"]]
        false_refusals = sum(r["modes"][mode]["refused"] for r in answerable)
        leaked: dict[str, list[int]] = {}
        for r in unanswerable:
            tally = leaked.setdefault(r["category"], [0, 0])
            tally[1] += 1
            tally[0] += not r["modes"][mode]["refused"]
        leaks = sum(v[0] for v in leaked.values())

        # Reference-based answer quality (no human labels): is the delivered answer about the same
        # thing as the dataset's own reference reply, and closer to it than to a wrong-intent reply?
        delivered = [r for r in answerable if r.get("reference_similarity") is not None
                     and not r["modes"][mode]["refused"]]
        quality = {
            "delivered": len(delivered),
            "answers_with_sentences_removed": sum(bool(r["modes"][mode].get("invented_claims")) for r in delivered),
            "mean_reference_similarity": round(float(np.mean([r["reference_similarity"] for r in delivered])), 3)
            if delivered else None,
            "on_topic_rate": round(sum(r["reference_similarity"] > r["control_similarity"] for r in delivered)
                                   / len(delivered), 3) if delivered else None,
        }
        metrics[mode] = {
            "false_refusal_rate": round(false_refusals / max(1, len(answerable)), 3),
            "false_refusals": false_refusals, "n_answerable": len(answerable),
            "leak_rate": round(leaks / max(1, len(unanswerable)), 3),
            "leaks": leaks, "n_unanswerable": len(unanswerable),
            "leak_rate_by_category": {c: round(a / b, 3) for c, (a, b) in leaked.items()},
            "answer_quality": quality,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": round(precision, 3), "recall": round(recall, 3),
            "f1": round(f1, 3), "accuracy": round(accuracy, 3),
            "refused_rate_by_category": {
                cat: round(count / total, 3) for cat, (count, total) in refused_rates.items()
            },
            "status_counts": status_counts,
            "mean_latency": round(float(np.mean(latencies)), 2),
            "median_latency": round(float(np.median(latencies)), 2),
        }
    return metrics


def nli_vs_llm_points(rows: list[dict]) -> tuple[list[float], list[float]]:
    xs, ys = [], []
    for row in rows:
        m = row["modes"]["both"]
        if m["nli_grounding"] is not None and m["llm_grounding"] is not None:
            xs.append(m["nli_grounding"])
            ys.append(m["llm_grounding"])
    return xs, ys


def save_charts(rows: list[dict], metrics: dict) -> None:
    modes = MODES
    labels = {"none": "Baseline", "nli": "NLI", "llm": "LLM-judge", "both": "NLI+LLM"}

    precision = [metrics[m]["precision"] for m in modes]
    recall = [metrics[m]["recall"] for m in modes]
    f1 = [metrics[m]["f1"] for m in modes]
    x = np.arange(len(modes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width, precision, width, label="Precision")
    ax.bar(x, recall, width, label="Recall")
    ax.bar(x + width, f1, width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[m] for m in modes])
    ax.set_ylim(0, 1.05)
    ax.set_title("Refusal correctness (refuse = positive): precision / recall / F1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "detection_prf_v2.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, m, met in zip(axes, modes, metrics.values()):
        cm = np.array([[met["tn"], met["fp"]], [met["fn"], met["tp"]]])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["no refusal", "refused"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["answerable", "should refuse"])
        ax.set_title(labels[m])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Refusal correctness (rows=gold, cols=predicted refusal)")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "confusion_matrices_v2.png", dpi=150)
    plt.close(fig)

    xs, ys = nli_vs_llm_points(rows)
    if len(xs) >= 2:
        corr = np.corrcoef(xs, ys)[0, 1]
    else:
        corr = float("nan")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(xs, ys, alpha=0.7)
    ax.plot([0, 1], [0, 1], "r--", linewidth=1)
    ax.set_xlabel("NLI grounding score")
    ax.set_ylabel("LLM-judge grounding score")
    ax.set_title(f"Detector agreement (Pearson r = {corr:.3f})")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "agreement_scatter_v2.png", dpi=150)
    plt.close(fig)

    cats = sorted({r["category"] for r in rows})
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width2 = 0.2
    x2 = np.arange(len(cats))
    for i, m in enumerate(modes):
        rates = [metrics[m]["refused_rate_by_category"].get(c, 0.0) for c in cats]
        ax.bar(x2 + (i - 1.5) * width2, rates, width2, label=labels[m])
    ax.set_xticks(x2)
    ax.set_xticklabels(cats)
    ax.set_ylim(0, 1.05)
    ax.set_title("Refusal rate by category")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "refusal_behavior_v2.png", dpi=150)
    plt.close(fig)


def main() -> None:
    global EVAL_FILE, RESULTS_FILE
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-file", type=Path, default=EVAL_FILE)
    ap.add_argument("--results", type=Path, default=RESULTS_FILE)
    ap.add_argument("--limit", type=int, default=0, help="smoke run on the first N questions (writes <results>.smoke.json)")
    ap.add_argument("--fresh", action="store_true", help="ignore and overwrite previous results")
    args = ap.parse_args()
    EVAL_FILE, RESULTS_FILE = args.eval_file, args.results
    if args.limit:
        RESULTS_FILE = RESULTS_FILE.with_suffix(".smoke.json")

    evals = load_evals()
    if args.limit:
        evals = evals[:: max(1, len(evals) // args.limit)][: args.limit]  # evenly spread over all categories
    cfg = run_config()

    rows: list[dict] = []
    prev = None if args.fresh else load_previous()
    if prev:
        if prev.get("config") != cfg:
            diff = {k: (prev.get("config", {}).get(k), cfg[k]) for k in cfg if prev.get("config", {}).get(k) != cfg[k]}
            sys.exit(f"Results in {RESULTS_FILE} came from a different configuration {diff}.\n"
                     "Rerun with --fresh (overwrites them) or use --results for a new file.")
        rows = prev["rows"]
    scorer = ReferenceScorer(evals)
    by_id = {q["id"]: q for q in evals}
    for row in rows:  # backfill rows saved before this metric existed; no regeneration needed
        q = by_id.get(row["id"])
        if q and q.get("gold_response") and row.get("answer") and not row["self_refusal"] \
                and row.get("reference_similarity") is None:
            row["reference_similarity"], row["control_similarity"] = scorer.score(q, row["answer"])
    done = {row["id"] for row in rows}
    remaining = [q for q in evals if q["id"] not in done]
    print(f"Eval set: {len(evals)} questions | already done: {len(done)} | remaining: {len(remaining)}", flush=True)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if remaining:
        retriever = Retriever()
        generator = Generator()
        detector = HallucinationDetector()
        start = time.time()
        for q in remaining:
            t = time.time()
            row = evaluate_q(retriever, generator, detector, q, scorer)
            rows.append(row)
            with RESULTS_FILE.open("w", encoding="utf-8") as fh:
                json.dump({"config": cfg, "rows": rows}, fh, ensure_ascii=False, indent=2)
            print(
                f"[{row['id']:>3}] {q['category']:<22} {time.time() - t:5.1f}s  rel={row['max_relevance']:.2f}  "
                + ", ".join(f"{m}:{row['modes'][m]['status']}" for m in MODES),
                flush=True,
            )
        total = time.time() - start
    else:
        total = 0.0

    metrics = compute_metrics(rows)
    data = {"config": cfg, "rows": rows, "metrics": metrics, "total_runtime_s": round(total, 1)}
    with RESULTS_FILE.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    if not args.limit:
        save_charts(rows, metrics)

    print(f"\n===== RESULTS ({len(rows)} questions; unanswerable = should refuse) =====")
    print(f"{'mode':>5} {'false-refusal':>14} {'leak':>10} {'P':>6} {'R':>6} {'F1':>6} {'lat(s)':>7}")
    for mode in MODES:
        m = metrics[mode]
        print(f"{mode:>5} {m['false_refusals']:>5}/{m['n_answerable']:<3} {m['false_refusal_rate']:>5.0%} "
              f"{m['leaks']:>4}/{m['n_unanswerable']:<3} {m['leak_rate']:>4.0%} "
              f"{m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f} {m['mean_latency']:>7.1f}")
        q = m["answer_quality"]
        if q["delivered"]:
            print(f"      delivered answerable answers: {q['delivered']}  mean similarity to reference reply "
                  f"{q['mean_reference_similarity']:.3f}  on-topic rate {q['on_topic_rate']:.0%}  "
                  f"answers with invented sentences removed: {q['answers_with_sentences_removed']}")
    agree = nli_vs_llm_points(rows)
    if len(agree[0]) >= 2:
        print(f"NLI vs LLM-judge correlation (n={len(agree[0])}): {np.corrcoef(*agree)[0, 1]:.3f}")
    print(f"Results written to {RESULTS_FILE}" + ("" if args.limit else f"; charts in {REPORTS_DIR}"))


if __name__ == "__main__":
    main()
