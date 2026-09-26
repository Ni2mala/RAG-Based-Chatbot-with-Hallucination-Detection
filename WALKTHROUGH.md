# RAG-Guard — Code Walkthrough & Self-Study Guide

Companion to `THESIS_DOCUMENTATION.md` (decisions, design, results) and `PROGRESS.md` (session handoff).
This file explains *how the code works* so you can read, run, and defend every piece of the project.

---

## Part 1 — The system in one breath

A user asks a question → we **embed** it into a 768-d vector → find the 5 most **cosine-similar**
chunks in Chroma (the compact KB, 130 docs / 210 chunks) → if the best match is too weak
(< 0.55) we refuse *before* paying for an LLM call → otherwise `llama3.2` writes an answer from
those chunks → we **split the answer into claims** → every claim is checked by **NLI**
(entailment vs each chunk) and the whole answer by the **LLM-judge** → the two scores are averaged
→ `grounded` (answer shown) / `partially_supported` (shown, flagged) / `unsupported` (**refused**
with a safe message).

## Part 2 — Component map (read in this order)

| Layer | File | What it does — what you say about it |
|---|---|---|
| Config | `src/rag_guard/core/config.py` | Everything tunable lives in `.env` (thresholds, model names, paths). All knobs are explicit. |
| Data | `scripts/fetch_dataset.py` | Downloads 25,545 Bitext rows; a `SPECIAL` regex map rewrites `{{Entity}}` placeholders into natural phrases (e.g. `{{Customer Support Working Hours}}` → "our business hours"). |
| Index | `src/rag_guard/services/ingestion_service.py` | **Compact KB**: `_deduplicate` keeps ≤ 5 evenly-spaced rows per intent (130 rows). Why: 40k raw chunks take hours to embed on CPU; uniform sampling keeps all 26 intents. |
| Chunking | `src/rag_guard/rag/chunking.py` | Sliding window, 900 chars / 100 overlap. |
| Embedding | `src/rag_guard/rag/embeddings.py` | `nomic-embed-text`, 768-d, `search_document:`/`search_query:` prefixes (required for good retrieval), batch 32. |
| Store | `src/rag_guard/rag/vector_store.py` | Chroma persistent DB, cosine space, `reset()`/`upsert()`. |
| Retrieval | `src/rag_guard/rag/retriever.py` | Embed the query (query prefix), `collection.query`, convert Chroma `distance` → `score = 1 - distance`. |
| Prompts | `src/rag_guard/rag/prompts.py` | The **grounding contract**: "answer ONLY from CONTEXT", plus a fixed sentence *"I could not find sufficient information…"* for gaps. |
| Generate | `src/rag_guard/rag/generator.py` | Ollama `llama3.2`, temperature 0.2 (low = less invention), 600 tokens. |
| Refusal | `src/rag_guard/services/chat_service.py` | Orchestration brain: retrieval gate → generate → self-refusal check → detector → refuse if `unsupported`. |
| Claims | `src/rag_guard/guardrails/claim_extractor.py` | Sentence-splits the answer into atomic factual units. |
| NLI | `src/rag_guard/guardrails/grounding_check.py` | DeBERTa `nli-deberta-v3-small`; for each (chunk, claim) pair → entailment/contradiction/neutral; claim supported if entailment ≥ 0.4 and beats contradiction. |
| Judge | `src/rag_guard/guardrails/llm_judge.py` | llama3.2 in a **JSON-constrained** "faithfulness judge" prompt → `score` 0–1 + list of unsupported claims. Holistic, not per-claim. |
| Policy | `src/rag_guard/guardrails/hallucination_detector.py` | Runs NLI and/or judge per `DETECTOR_MODE`; averages scores (`both`); maps to status. `mode="none"` = baseline (no guard). |
| API | `src/rag_guard/api/routes/*`, `src/rag_guard/main.py`, `api/dependencies.py` | FastAPI `/chat`, `/health`, `/admin/ingest`; singletons via `lru_cache`. |
| UI | `app/ui/streamlit_app.py` | Chat app that posts to `/chat`, renders sources + NLI/LLM scores + unsupported-claim count. |
| Eval | `scripts/build_eval_set.py`, `scripts/run_evaluation.py` | 35 labeled questions; generate once → test 4 configs; JSON + 4 charts to `reports/`. |

**Typical flow (block-card question):** Streamlit `POST /chat` → `ChatService.answer` →
`Retriever.retrieve` (top-5, max 0.78) → `Generator.generate` → `ClaimExtractor` →
`GroundingChecker` (NLI 0.57) + `LLMJudge` (0.9) → avg 0.735 → `grounded` → answer returned
with verification payload.

## Part 3 — Five concepts you must be able to explain cold

1. **Embedding + cosine similarity** — text → vector; similar text → nearby vectors; `score = 1 − distance`.
2. **RAG** — retrieve authoritative chunks, force the LLM to answer only from them (contract in the system prompt); the *retrieval gate* refuses cheaply when nothing matches.
3. **NLI entailment** — a dedicated fine-tuned model classifies (premise, hypothesis) as entailment/contradiction/neutral; each retrieved chunk is the premise, each claim the hypothesis.
4. **LLM-as-judge** — an LLM rates grounding of the whole answer; zero-shot, JSON-schema-constrained output.
5. **Precision / Recall / F1 / accuracy** — in our eval, *hallucinated* = "gold-unanswerable (should refuse)", *predicted hallucinated* = "refused". Precision = of the refusals, how many were right; Recall = of the hallucination-gold, how many we refused; F1 = harmonic mean.

## Part 4 — Self-paced catch-up plan (~4–6 hours, in order)

1. **Read the docs (45 min):** `THESIS_DOCUMENTATION.md`, then `PROGRESS.md`. These are your memory of every decision and its "why".
2. **Replay the build yourself (90 min):** run each script in order and watch the logs —
   `fetch_dataset.py` → `build_vector_store.py` → `build_eval_set.py` → `test_nli.py` →
   `test_guardrails.py`. Type the commands yourself so they are muscle memory.
3. **Run the live demo (30 min):** start uvicorn + Streamlit; ask the block-card question and the weather question; *read* the raw JSON from `Invoke-RestMethod` so you see the `verification` fields.
4. **Do an experiment (45 min):** edit `.env` — set `TOP_K=8` and `RELEVANCE_MIN_SCORE=0.45`, re-run `test_nli.py`, observe changes in retrieval similarity and grounding score, then restore.
5. **Dissect a result you can explain (60 min):** open `reports/evaluation_results.json`; pick 3 rows (in-domain grounded, in-domain refused, injection refused) and trace each through the code path by hand. Open the 4 PNGs and explain what each axis means.
6. **Quiz yourself (30 min):** *Why 5 rows per intent? Why 0.55? What does "both" average? Why is NLI 0.57 vs judge 0.9? Why is both's F1 best? What was the NLI label-mapping bug and its fix?*

## Quick-start commands (from D:\Thesis\rag_guard_bank)

- venv: `D:\Thesis\.venv` (Python 3.13.4)
- API: `.venv\Scripts\python -m uvicorn rag_guard.main:app --app-dir src --port 8000`
- UI: `.venv\Scripts\python -m streamlit run app\ui\streamlit_app.py`
- Rebuild index: `.venv\Scripts\python scripts\build_vector_store.py`
- Eval: `.venv\Scripts\python scripts\run_evaluation.py` (~30–50 min)