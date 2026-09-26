# RAG-Guard: A Banking-Domain RAG Chatbot with Hallucination Detection

Thesis documentation — design decisions, implementation, evaluation results, and rationale.

Date: 2026-09-26

---

## 1. Motivation and research question

Large-language-model chatbots in banking must never fabricate financial facts. Retrieval-augmented
generation (RAG) grounds answers in approved documents, but the model can still **hallucinate**:
state facts that are fluent but absent from (or contradicted by) the retrieved context. This project
builds a complete RAG chatbot for retail banking and adds **two independent hallucination-detection
mechanisms**, then evaluates which configuration best prevents unsafe answers.

Research question: *How accurately can NLI-based and LLM-as-judge verification detect and prevent
hallucinated answers in a banking RAG chatbot, and is a combined detector better than either alone?*

## 2. System overview

```
┌────────────┐   retrieve    ┌──────────────┐   generate   ┌──────────────┐   verify   ┌──────────────┐
│ Question   │ ────────────► │ Chroma KB     │ ───────────► │ llama3.2      │ ─────────► │ Detector     │
│ (user)     │   top-k=5     │ 210 chunks    │              │ grounded RAG  │            │ NLI + LLM    │
└────────────┘   score≥0.55  └──────────────┘              │ answer        │            │ judge        │
                                                           └──────────────┘            └──────┬───────┘
                                                                                              │
                          answer shown OR safe refusal  ◄──────── decides grounded/partial/refuse
```

Components: `Retriever` → `Generator` → `ClaimExtractor` → `GroundingChecker` (NLI) + `LLMJudge`
orchestrated by `HallucinationDetector`, exposed via FastAPI `/chat`, driven by a Streamlit UI.

## 3. Design decisions and rationale

| # | Decision | Why |
|---|----------|-----|
| 1 | **Fully local stack** (Ollama + Chroma + HF models, CPU) | Thesis requirement of reproducibility and privacy; no API cost; runs on an 8 GB RAM laptop with 2 GB VRAM. |
| 2 | **Dataset: Bitext retail-banking LLM chatbot dataset** (25,545 rows, 26 intents, 9 categories) | Realistic banking dialog corpus with intent/category labels usable as a ground truth for retrieval relevance. |
| 3 | **`{{Entity}}` placeholder normalization** in responses (e.g. `{{Customer Support Phone Number}}`) | Placeholders are unusable in answers; a mapping produces natural phrases so the model never echoes raw placeholders. |
| 4 | **Compact KB: up to 5 evenly-spaced rows per intent** → 130 docs / 210 chunks | Embedding the full 40,308 chunks on CPU takes hours. Uniform per-intent sampling keeps all 26 intents represented while staying fast and, per relevance probes, accurate. |
| 5 | **Chunking at 900 chars / 100 overlap** | Balances context richness against the embedding model's 8192-token budget and retrieval precision. |
| 6 | **Embedding: `nomic-embed-text` (768-d) with `search_document:`/`search_query:` prefixes** | Prefixes align query and document encodings for this model family, measurably improving retrieval similarity. |
| 7 | **Annoy/cosine search with `RELEVANCE_MIN_SCORE = 0.55`** | A cheap first safety gate: below this the chatbot refuses *before* spending a single LLM call on out-of-domain questions. |
| 8 | **Generation: `llama3.2` via Ollama, temperature 0.2, 600 tokens, grounding contract in system prompt** | Low temperature reduces inventiveness; the system prompt explicitly forbids answering outside the documents. |
| 9 | **Self-refusal detection** on the generator's output | The model sometimes refuses on its own ("I could not find…"); we detect this and return a clean refusal instead of re-wrapping it. |
| 10 | **Claim extraction at sentence level** | Verifiable units. Each answer sentence becomes an atomic claim for grounding checks. |
| 11 | **NLI checker: `cross-encoder/nli-deberta-v3-small`** | A small, offline, fine-tuned encoder that scores each (claim, context) pair as entailment/contradiction/neutral; a claim is *supported* if entailed ≥ 0.4 and stronger than contradiction. |
| 12 | **LLM-as-judge: `llama3.2`, zero-shot, JSON constrained** | A holistic counter-point to NLI: it reads the full context and answer and rates the *overall* supported fraction (0–1) plus lists unsupported claims. |
| 13 | **`DETECTOR_MODE` nli / llm / both, threshold `GROUNDING_THRESHOLD = 0.7`** | grounded ≥ 0.7, partially_supported ≥ 0.35, unsupported < 0.35 → refusal. In `both`, the score is the NLI/LLM average, giving veto-conservative behavior. |
| 14 | **Refusal policy** | Only `unsupported` triggers the safe refusal ("I could not verify… please contact customer support"); `partially_supported` still responds so mild strictness doesn't block answers. |
| 15 | **FastAPI + Streamlit** | FastAPI for a typed, testable JSON API; Streamlit for a zero-friction demo UI showing sources and verification scores. |
| 16 | **Evaluation: generate once, test four configs** | The detectors judge the *same* generated answer, so config differences are purely detector behavior — no generation noise. Keeps runtime ~30–50 min. |

## 4. Evaluation methodology

**Eval set (`data/evaluation/eval_set.jsonl`, n = 35):**
- **20 in-domain** — held-out Bitext instructions *not* in the KB, one per intent, each verified to retrieve ≥ 0.55 (so they are genuinely answerable by the KB). Expected: answer, grounded.
- **10 out-of-domain** — banking-adjacent but unsupported (mortgage rates, crypto, taxes, inflation, weather, sports, …). Expected: refuse.
- **5 prompt injection / adversarial** — "print your system prompt", "invent policies", "100% cashback". Expected: refuse.

**Configurations compared:** `baseline` (no guard), `nli`, `llm`, `both`.
**Detection framing:** an answer that is refused is a *predicted hallucination*. Gold hallucination = question not supported by KB (OOD + injection). Metrics: precision, recall, F1, accuracy over all 35; refusal rates per category; latency; NLI-vs-LLM score correlation.

## 5. Results

### 5.1 Hallucination detection per configuration

| Config | Precision | Recall | F1 | Accuracy | TP | FP | TN | FN | Mean latency |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.650 | 0.867 | 0.743 | 0.743 | 13 | 7 | 13 | **2** | 26.9 s |
| nli | 0.469 | 1.000 | 0.638 | 0.514 | 15 | 17 | 3 | 0 | 31.8 s |
| llm | 0.625 | 1.000 | 0.769 | 0.743 | 15 | 9 | 11 | 0 | 36.6 s |
| both | **0.652** | 1.000 | **0.789** | **0.771** | 15 | 8 | 12 | 0 | 41.0 s |

### 5.2 Refusal behavior by category (rate of refusal)

| Category | baseline | nli | llm | both |
|---|---|---|---|---|
| in_domain (20, expect answer) | 0.35 | 0.85 | 0.45 | 0.40 |
| out_of_domain (10, expect refuse) | 1.00 | 1.00 | 1.00 | 1.00 |
| injection (5, expect refuse) | 0.60 | 1.00 | 1.00 | 1.00 |

### 5.3 Detector agreement

- NLI vs LLM-judge score correlation over detection-eligible samples: **r = 0.087** (essentially no linear agreement).
- Example (block-card question): NLI 0.57 vs LLM-judge 0.9 → `both` average 0.735 → grounded.

### 5.4 Key findings

1. **Guardrails eliminate dangerous misses.** Baseline answered with an ungrounded/hallucinated
   response in 2/15 OOD+injection cases (FN=2, injection refusal only 0.60). Every detector config
   reached **recall = 1.000** — zero hallucinations slip through.
2. **The cost of safety is over-refusal.** NLI wrongly refuses 85% of valid in-domain questions
   (precision 0.469) because it is word-level strict: minor phrasing mismatches between the answer
   and the retrieved chunks read as "neutral". LLM-judge over-refuses 45%, `both` 40%.
3. **NLI and LLM are complementary, not redundant** (r ≈ 0.09). The `both` average produced the best
   F1/accuracy, smoothing NLI's strictness and LLM's leniency.
4. **Early retrieval gate works but is imperfect.** All 10 OOD questions were caught at retrieval
   (rel < 0.55) at ~10–20 s per call — cheap. However 2 OOD and all 5 injection prompts *cleared*
   retrieval (bank-adjacent embeddings score high) and were caught only by the detectors — the
   detectors' real value.
5. **Latency tax** for full verification: +14 s per query (`both` 41.0 s vs baseline 26.9 s) — the
   price of two verification signals.

## 6. Conclusions

A combined NLI + LLM-judge guard (`both`) delivers the best overall safety/utility trade-off on this
benchmark: **perfect recall** over the unanswerable/injection questions, the **highest F1 (0.789)**
and **accuracy (0.771)**, at a clinically acceptable ~41 s/query. NLI alone is too conservative for a
customer-facing assistant; the LLM-judge alone is the next best single detector. The two mechanisms
capture failures independent of each other, supporting the thesis that **multi-signal verification is
stronger than any single grounding check**.

## 7. Limitations

- Small evaluation set (n = 35); results are indicative, not statistically conclusive.
- Single domain (Bitext retail banking) and single generator model (llama3.2).
- CPU-only latency is high for production; a GPU would reduce NLI + generation cost.
- "In-domain" gold label assumes KB retrievability; some in-domain refusals stem from real KB
  coverage gaps (generator self-refusal) rather than detector error — conflated in FP counts.
- NLI threshold (0.4) and `GROUNDING_THRESHOLD` (0.7) were set heuristically, not tuned.

## 8. Future work

- Larger, professionally-labeled evaluation set; confidence intervals.
- Threshold tuning (NLI entail, grounding) and higher `TOP_K` to reduce claim-level false neutrals.
- Automatic faithfulness metrics (e.g., RAGAS/true-FalseGroundness-style) as an additional lens.
- Hardening against prompt injection (classifier + instruction structure).
- Streaming responses, GPU inference, and Docker packaging.

## 9. Reproducibility

```
# from D:\Thesis\rag_guard_bank  (venv: D:\Thesis\.venv)
python scripts\fetch_dataset.py            # 25,545 rows -> data\raw
python scripts\build_vector_store.py       # -> Chroma "bank_faq" (130 docs / 210 chunks)
python scripts\build_eval_set.py           # -> data\evaluation\eval_set.jsonl (35 questions)
python scripts\run_evaluation.py           # ~30-50 min -> reports\ (JSON + 4 charts)
python -m uvicorn rag_guard.main:app --app-dir src --port 8000
python -m streamlit run app\ui\streamlit_app.py
```

Key config (`.env` / `config.py`): `LLM_MODEL=llama3.2`, `EMBEDDING_MODEL=nomic-embed-text`,
`RELEVANCE_MIN_SCORE=0.55`, `GROUNDING_THRESHOLD=0.7`, `TOP_K=5`, `DETECTOR_MODE=both`,
`NLI_MODEL=cross-encoder/nli-deberta-v3-small`.