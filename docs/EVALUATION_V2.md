# Evaluation v2

Run 2026-11-22, 129 questions, 7 minutes on the local CPU/GPU setup. Raw results: `rag_guard_bank/reports/evaluation_results_v2.json` (includes the full configuration, the generated answers and the retrieved context for every question). Charts: `reports/*_v2.png`. v1 files are kept unchanged for comparison with the earlier numbers.

## What changed from v1

| | v1 | v2 |
|---|---|---|
| Questions | 35 (20 answerable) | 129 (78 answerable, 51 unanswerable) |
| Answerable selection | Only questions that passed the 0.55 relevance gate | Random held-out questions, 3 per intent (26 intents), no filter |
| Unanswerable kinds | out of domain 10, injection 5 | out of domain 20, **tempting unanswerable 20** (exact fees, limits, rates), injection 11 |
| Reproducibility | Temperature 0.2, no seed, resume mixed old and new rows | Fixed seed 42, full configuration stored and checked on resume (`--fresh` to override) |
| Metrics | Precision/recall/F1 called "hallucination detection" | **False-refusal rate** and **leak rate** (unanswerable question answered) are primary. P/R/F1 renamed "refusal correctness". Answer quality is measured against the dataset's reference replies, and detection ability by a synthetic-fault test. No human labels are used anywhere. |
| Detector cost | NLI and judge run three times per answer | One NLI pass and one judge call shared by the nli, llm and both modes |

Reproducibility check: two identical 20-question runs gave the same generated answer in 19 of 20 cases and identical verdicts in all four modes for 20 of 20.

The harness calls the pipeline components directly, so the keyword injection filter in `ChatService` is **off**. These numbers measure the model plus the guards.

## Results

| Mode | False refusals (of 78 answerable) | Unanswerable answered (of 51) | Refusal precision | Recall | F1 | Mean latency |
|---|---|---|---|---|---|---|
| none (no detector) | 12 (15%) | **6 (12%)** | 0.789 | 0.882 | 0.833 | 2.0 s |
| nli only | 23 (30%) | 1 (2%) | 0.685 | 0.980 | 0.806 | 2.2 s |
| llm judge only | 17 (22%) | 0 | 0.750 | 1.000 | 0.857 | 3.0 s |
| **both (default)** | **14 (18%)** | **0** | 0.785 | 1.000 | 0.879 | 3.2 s |

"none" still has the relevance gate and the model's own refusals, so it is a gate-plus-model baseline, not an unguarded LLM.

Leaks without the detector: 4 of 11 injections, 1 of 20 out of domain, 1 of 20 tempting unanswerable. Under `both` there were no leaks. Hence the mitigation layer removed all 6 leaked answers (12% to 0%) at the cost of 2 extra false refusals (12 to 14 of 78).

Where answerable questions end up under `both`: 45 delivered grounded, 19 delivered partially supported, 12 refused by the model itself, 2 refused by the detector, 0 refused by the relevance gate.

## Findings

1. **The detector earns its place on safety.** It stopped all 6 leaked answers; the cost is small (2 questions) and the latency overhead is 1.2 s.
2. **Most false refusals are the model's own** (12 of 14 under `both`), not the guards'. They concentrate where retrieval relevance is 0.70-0.80 (8 of 25 answerable questions refused) and almost vanish at 0.80 or higher (2 of 46). The retry only triggers at 0.75, which fits the earlier slang and typo finding.
3. **The relevance gate never wrongly rejects an answerable question** (0 of 78; lowest in-domain relevance 0.598). This settles the ADR-02 worry that the v1 pre-filter hid gate false-rejects. But the gate also stops only 11 of 51 unanswerable questions; the model refuses 34 more by itself, and 6 reach the detector.
4. **NLI alone is the weakest mode** (30% false refusals), the judge alone is better, and the average is best. NLI and judge scores correlate at only 0.378 on 71 answers, so they measure different things.
5. 19 of 64 delivered answers (30%) are only partially supported by the detectors. Without human labels we cannot say how many of those contain real errors; see the fault test below for what the detectors can and cannot catch.

## Answer quality against the dataset's reference replies (no labels)

Every answerable question comes from a held-out dataset row that has its own reference reply (written by the dataset, not by Ollama). For each delivered answer the harness measures the cosine similarity (nomic embeddings) to that reference and to a fixed reference of a *different* intent.

| Mode | Delivered answerable answers | Mean similarity to own reference | Closer to own reference than to the wrong-intent one |
|---|---|---|---|
| none | 66 | 0.924 | 100% |
| nli | 55 | 0.925 | 100% |
| llm | 61 | 0.929 | 100% |
| both | 64 | 0.926 | 100% |

Reading: when the bot answers, it answers about the right thing, and the detectors do not change that. This metric measures topic correctness only. It cannot tell a faithful answer from one with an invented detail (mean 0.931 for grounded vs 0.916 for partially supported answers in an earlier check), so it does not replace the fault test below.

## Synthetic-fault test (no labels; `scripts/fault_injection_test.py`)

312 cases built from the dataset: for each of the 78 answerable questions, a **good** answer (the top retrieved chunk's answer text, verbatim, so faithful by construction) and three faulty versions:

- **wrong topic:** a real reply from the dataset for a different intent.
- **fabricated claim:** the good answer plus one invented sentence without numbers ("A manager must personally approve the request...").
- **fabricated figure:** the good answer plus one invented sentence with a figure ("A processing fee of $25 applies...", "within 3 business days").

Share of answers **refused** (status `unsupported`; for `good` lower is better, for faults higher is better):

| Detector | good (false alarm) | wrong topic | fabricated claim | fabricated figure |
|---|---|---|---|---|
| nli | 0% | 83% | **0%** | 100% |
| llm judge | 9% | 82% | 32% | 100% |
| **both (default)** | **0%** | **83%** | **0%** | **100%** |
| figure check alone | 0% | 1% | 0% | 100% |

Ignoring the figure check, NLI and the judge together catch **0%** of the fabricated figures (judge alone 6%). Every figure fault is stopped by the deterministic figure check, not by the language models.

Findings:
1. **Wrong-topic answers are caught well** (65 of 78 refused, 12 more flagged partially supported, 1 passed as grounded).
2. **Fabricated figures are caught only because of the figure check.** NLI and the judge do not notice them.
3. **Main weakness: one invented sentence inside an otherwise correct answer passes.** Under `both`, 0 of 78 were refused; 31 were graded partially supported and 47 grounded. NLI scores the answer as the *fraction* of supported claims (mean 0.87 with one bad claim among about 8), which stays above the 0.7 threshold. The judge scores 0.51 on these but is unreliable (it also refuses 9% of verbatim-correct answers).
4. The test found a bug in the figure check: a step number in a document ("2.") counted as support for an invented "2 requests per year", so 18 of 78 invented figures slipped through. Fixed: list markers are now ignored in the documents too (78 of 78 caught, still 0 false alarms on good answers).

Limits: injected faults are easier than natural hallucinations; copied text is the easy case for `good` (natural paraphrase false alarms are the false-refusal rate above); fault templates are few (8 claims, 6 figures) and chosen by me.

## Limits to state in the dissertation

- There are **no human faithfulness labels**, so no hallucination rate on the bot's natural answers is reported. Detector ability is measured by construction (fault test) and by answerability and reference similarity, and the dissertation should say so.
- The 78 answerable questions are paraphrases of the same templated Bitext intents the knowledge base was built from, so retrieval is easy and "answerable" results may be optimistic against real documents.
- I wrote the unanswerable questions and the fault templates. 51 unanswerable questions give wide uncertainty: 0 of 51 leaks still allows a true rate of several percent.
- One run per configuration with a fixed seed; the seed makes the run repeatable, not the result robust across seeds.
- The injection filter is excluded on purpose; its effect is measured in `QA_REPORT.md`.

## How to rerun

```bash
cd rag_guard_bank
python scripts/build_eval_set.py          # rebuilds data/evaluation/eval_set_v2.jsonl (deterministic)
python scripts/run_evaluation.py --fresh  # ~7 min; adds reference similarity
python scripts/fault_injection_test.py    # ~10 min; writes reports/fault_injection_results.json
```
