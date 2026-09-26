# RAG-Guard Thesis — Session Handoff

Updated: 2026-09-26

## What this project is
Banking-domain RAG chatbot ("RAG-Guard") with hallucination detection on the
Hugging Face Bitext retail-banking dataset (~25.5k rows), for a thesis.
Local offline stack: Ollama llama3.2 + Chroma vector store + NLI + LLM-judge.

## Quick start (from D:\Thesis\rag_guard_bank)
- venv: D:\Thesis\.venv (Python 3.13.4)  |  Ollama models: llama3.2, nomic-embed-text
- API:  .venv\Scripts\python -m uvicorn rag_guard.main:app --app-dir src --port 8000
- UI:   .venv\Scripts\python -m streamlit run app\ui\streamlit_app.py
- Rebuild index: python scripts\build_vector_store.py | Re-fetch: python scripts\fetch_dataset.py
- Smoke: scripts\quick_test.py (RAG), scripts\test_guardrails.py (detectors), scripts\test_nli.py (NLI only)

## Stage status
[1] Config/scaffold + dataset        DONE  (25,545 rows, {{placeholder}} normalized)
[2] RAG pipeline                     DONE  (130 docs / 210 chunks, Chroma `bank_faq`)
[3] FastAPI + Streamlit UI           DONE  (/chat /health /admin/ingest)
[4] UI verified + wording polish     DONE
[5] Hallucination detectors          DONE & RUNNING — 1 suspected bug + 1 polish item (below)
[6] Evaluation harness (metrics)     TODO
[7] Tests/Docker + code walkthrough  TODO

## Open items (stage 5)
1. NLI=0.0 vs LLM-judge=0.9 on a grounded question → suspected id2label case mismatch
   in src/rag_guard/guardrails/grounding_check.py (compares name=="entailment").
   NEXT: inspect config.id2label, map labels correctly, re-run test_guardrails.py.
2. Generator self-refusal gets re-wrapped by our refusal_message (transfer-duration
   answer is clunky). Polish wording logic.

## Verified behavior (today)
- Blocked card → grounded-ish answer (partial; blocked by NLI bug above)
- "How long does a bank transfer take?" → refused, NLI=0.0 AND judge=0.0 (no data in KB)
- "Weather in Paris?" → refused at retrieval (relevance 0.475 < 0.55), no LLM cost

## Key design facts
- Modes: DETECTOR_MODE=nli|llm|both (default both). Score = NLI+LLM avg when both.
  grounded >= 0.7, unsupported < 0.35 → raises refusal ("Please contact support").
- Embeddings nomic-embed-text 768d with search_query:/search_document: prefixes, batch 32.
- KB sampling: max 5 evenly-spaced rows per intent (40k row full index too slow on CPU).
- Thresholds/.env: RELEVANCE_MIN_SCORE=0.55, GROUNDING_THRESHOLD=0.7, ADMIN_TOKEN.

## Next session checklist
1. Commit working tree (claim_extractor.py, grounding_check.py fixes + test_nli.py)
2. Fix NLI label mapping; revalidate
3. Clean double-wrapped refusal wording
4. Streamlit: display nli/llm_grounding + unsupported claims
5. Stage 6: eval set (held-out Bitext + out-of-domain + prompt injection) →
   scripts/run_evaluation.py comparing baseline vs NLI vs LLM-judge, metrics to reports/
6. Optional: add HF_TOKEN to silence download warning

## Pitfalls
- HF pipeline pairs must be {"text":..,"text_pair":..} and return a dict (not list)
- NLI DeBERTa: 534MB first download (cached); truncate chunks (512-token window)
- Don't commit storage/ (Chroma) — ignored via .gitignore