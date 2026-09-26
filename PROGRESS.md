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
[5] Hallucination detectors          DONE & VALIDATED (NLI label bug fixed; "none" baseline mode added)
[6] Evaluation harness (metrics)     DONE  (35-q eval -> reports/; see THESIS_DOCUMENTATION.md)
[7] Tests/Docker + code walkthrough  OPTIONAL (not started)

## Open items (stage 5)
1. [RESOLVED] NLI label-mapping bug. Fixed in grounding_check.py (_resolve + top_k + lowercase).
2. [DONE] Self-refusal no longer double-wrapped (is_self_refusal in chat_service).
3. [DONE] Streamlit shows NLI / LLM-judge scores + unsupported-claims count.

## Verified behavior
- Blocked card → NLI 0.57 vs LLM-judge 0.9 → both-mode avg 0.735 = grounded.
  NLI is claim-strict, judge holistic (thesis finding).
- "How long does a bank transfer take?" → refused; NLI=0.0 AND judge=0.0 (no data in KB)
- "Weather in Paris?" → refused at retrieval (relevance 0.475 < 0.55), no LLM cost

## Key design facts
- Modes: DETECTOR_MODE=nli|llm|both (default both). Score = NLI+LLM avg when both.
  grounded >= 0.7, unsupported < 0.35 → raises refusal ("Please contact support").
- Embeddings nomic-embed-text 768d with search_query:/search_document: prefixes, batch 32.
- KB sampling: max 5 evenly-spaced rows per intent (40k row full index too slow on CPU).
- Thresholds/.env: RELEVANCE_MIN_SCORE=0.55, GROUNDING_THRESHOLD=0.7, ADMIN_TOKEN.

## Next session checklist
1. Review/edit THESIS_DOCUMENTATION.md (results now filled from real run)
2. Optional tuning: NLI entail_threshold, GROUNDING_THRESHOLD, TOP_K (8?) to cut
   in-domain over-refusal (nli wrongly refused 85% of in-domain in eval)
3. Optional: tests (pytest), Docker packaging
4. Optional: full code walkthrough with user
5. Optional: add HF_TOKEN to silence download warning
6. Future: bigger labeled eval set for statistical significance

## Pitfalls
- HF pipeline pairs must be {"text":..,"text_pair":..} and return a dict (not list)
- NLI DeBERTa: 534MB first download (cached); truncate chunks (512-token window)
- Don't commit storage/ (Chroma) — ignored via .gitignore