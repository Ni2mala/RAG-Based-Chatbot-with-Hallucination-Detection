# RAG-Guard Thesis: Session Handoff

Updated: 2026-11-25. **Start here**, then `TODO.md`.

## What this project is
MSc AI dissertation: a banking RAG chatbot with hallucination detection on the Hugging Face Bitext retail-banking dataset (synthetic, 25,545 rows). Local stack: Ollama llama3.2 + Chroma + NLI + LLM judge. Deadline **12 Jan 2027, 19:30** (report PDF and code ZIP).

## Where things are
| File | Content |
|---|---|
| `TODO.md` | Full task list with status |
| `docs/university/ASSIGNMENT_BRIEF.md` | Deadline, word limit, marking scheme (proposal and rubric beside it) |
| `docs/EVALUATION_V2.md` | **Current results and limits; quote these** |
| `docs/QA_REPORT.md` | QA runs, root-cause analysis, fixes (addenda 1-4) |
| `docs/analysis/*.pdf` | Earlier code analysis and ADRs (gitignored; many issues since fixed) |
| `THESIS_DOCUMENTATION.md`, `WALKTHROUGH.md` | **Stale**: describe the old 35-question results and old behaviour |

## Run it (from `rag_guard_bank/`, venv in `.venv`)
- `source .venv/bin/activate`; Ollama models `llama3.2`, `nomic-embed-text`
- API: `python -m uvicorn rag_guard.main:app --app-dir src --port 8000` (restart after code changes)
- UI: `python -m streamlit run app/ui/streamlit_app.py`
- Index: `python scripts/build_vector_store.py`
- Tests: `python tests/test_<name>.py` (six files)
- Evaluation: `python scripts/run_evaluation.py --fresh` (~7 min); fault test: `python scripts/fault_injection_test.py` (~10 min)

## Request pipeline
keyword injection filter -> retrieval gate (0.55) -> llama3.2 (one retry with 8 chunks if it refuses at relevance >= 0.75) -> NLI (3 best sentence windows per claim) + LLM judge, averaged; any figure absent from the documents blocks the answer -> invented sentences (NLI-unsupported and >= 50% new words) are cut.

## Latest results (129 questions: 78 answerable, 51 unanswerable)
- Mode `both`: false refusals 14/78 (18%), unanswerable answered 0/51. Without detector: 12/78 and 6/51.
- Fault test (`both`): invented sentence or figure kept from the user 100%; wrong-topic answers refused 80%; good answers refused or altered 0%.
- Delivered answers are on topic against the dataset's reference replies (100%).
- Retrieval is not the problem: right intent in top 5 for 99.5%; hybrid search and metadata filtering did not help.

## Known gaps
- Invented claims that reuse the documents' words are cut only 54% of the time.
- Most remaining false refusals are the model's own, mainly slang and typo questions.
- No human faithfulness labels (owner's decision), so no hallucination rate on natural answers.

## Next steps
1. University items (owner): ethics form, interim review date, AI-use policy, confirm Ollama/Chroma instead of LangChain with the supervisor.
2. Dissertation, literature review first (the interim review needs it).
3. Optional engineering: pin dependency versions, remaining items in `TODO.md`.

## Pitfalls
- HF pipeline pairs must be `{"text":..,"text_pair":..}`.
- `storage/` is gitignored; rebuild the index on a fresh clone.
- Results change with prompt, model or thresholds; the evaluation script refuses to mix configurations (`--fresh`).
