import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rag_guard.core.config import PROJECT_ROOT, settings
from rag_guard.rag.retriever import Retriever
from rag_guard.rag.generator import Generator
from rag_guard.guardrails.hallucination_detector import HallucinationDetector
from rag_guard.guardrails.refusal import refusal_message
from rag_guard.services.chat_service import is_self_refusal

MODES = ["none", "nli", "llm", "both"]
EVAL_FILE = settings.DATA_EVAL_DIR / "eval_set.jsonl"
REPORTS_DIR = PROJECT_ROOT / "reports"
RESULTS_FILE = REPORTS_DIR / "evaluation_results.json"


def load_evals() -> list[dict]:
    with EVAL_FILE.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_done_ids() -> set[int]:
    if not RESULTS_FILE.exists():
        return set()
    with RESULTS_FILE.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return {row["id"] for row in data["rows"]}


def evaluate_q(retriever, generator, detector, q: dict) -> dict:
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

    t1 = time.time()
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
        for mode in MODES:
            t = time.time()
            if mode == "none":
                shown = answer
                refused = False
                status = "unguarded"
                nli = None
                llm = None
                unsupported = []
            else:
                det = detector.detect(answer, contexts, mode=mode)
                refused = det.status == "unsupported"
                status = det.status
                nli = det.nli_grounding
                llm = det.llm_grounding
                unsupported = det.unsupported_claims
                shown = (
                    refusal_message(unsupported)
                    if refused
                    else answer
                )
            total = retrieval_s + gen_s + (time.time() - t) if mode != "none" else retrieval_s + gen_s
            modes[mode] = {
                "refused": refused,
                "status": status,
                "nli_grounding": nli,
                "llm_grounding": llm,
                "unsupported_claims": unsupported,
                "latency": round(total, 3),
            }

    return {
        "id": q["id"],
        "question": question,
        "category": q["category"],
        "expected_action": q["expected_action"],
        "expected_grounded": q["expected_grounded"],
        "max_relevance": round(max_rel, 3),
        "sources": sources,
        "self_refusal": self_ref if max_rel >= settings.RELEVANCE_MIN_SCORE else None,
        "modes": modes,
    }


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
        metrics[mode] = {
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
    ax.set_title("Hallucination detection: precision / recall / F1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "detection_prf.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, m, met in zip(axes, modes, metrics.values()):
        cm = np.array([[met["tn"], met["fp"]], [met["fn"], met["tp"]]])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["no refusal", "refused"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["grounded", "hallucinated"])
        ax.set_title(labels[m])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Confusion matrices (rows=gold, cols=predicted refusal)")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "confusion_matrices.png", dpi=150)
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
    fig.savefig(REPORTS_DIR / "agreement_scatter.png", dpi=150)
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
    fig.savefig(REPORTS_DIR / "refusal_behavior.png", dpi=150)
    plt.close(fig)


def main() -> None:
    evals = load_evals()
    done = load_done_ids()
    remaining = [q for q in evals if q["id"] not in done]
    print(f"Eval set: {len(evals)} questions | already done: {len(done)} | remaining: {len(remaining)}", flush=True)
    if not remaining:
        print("Nothing to do.", flush=True)
        return

    retriever = Retriever()
    generator = Generator()
    detector = HallucinationDetector()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    if RESULTS_FILE.exists():
        with RESULTS_FILE.open("r", encoding="utf-8") as fh:
            rows = json.load(fh)["rows"]

    start = time.time()
    for q in remaining:
        t = time.time()
        row = evaluate_q(retriever, generator, detector, q)
        rows.append(row)
        with RESULTS_FILE.open("w", encoding="utf-8") as fh:
            json.dump({"rows": rows}, fh, ensure_ascii=False, indent=2)
        elapsed = time.time() - t
        print(
            f"[{row['id']:>2}] {q['category']:<14} {elapsed:5.1f}s  "
            f"rel={row['max_relevance']:.2f}  statuses="
            + ", ".join(f"{m}:{row['modes'][m]['status']}" for m in MODES),
            flush=True,
        )

    total = time.time() - start
    metrics = compute_metrics(rows)
    data = {"rows": rows, "metrics": metrics, "total_runtime_s": round(total, 1)}
    with RESULTS_FILE.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)

    save_charts(rows, metrics)

    print("\n===== METRICS (hallucination = question gold-unanswerable / should be refused) =====")
    for mode in MODES:
        met = metrics[mode]
        agree = nli_vs_llm_points(rows)
        print(
            f"{mode:>4}  P={met['precision']:.3f} R={met['recall']:.3f} F1={met['f1']:.3f} "
            f"acc={met['accuracy']:.3f}  TP={met['tp']} FP={met['fp']} TN={met['tn']} FN={met['fn']}  "
            f"lat={met['mean_latency']:.1f}s  {met['refused_rate_by_category']}"
        )
    print(f"\nTotal runtime: {total / 60:.1f} min")
    if len(agree[0]) >= 2:
        xs, ys = agree
        print(f"NLI vs LLM-judge correlation (n={len(xs)}): {np.corrcoef(xs, ys)[0, 1]:.3f}")
    print(f"Charts and JSON written to {REPORTS_DIR}")


if __name__ == "__main__":
    main()