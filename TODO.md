# TODO

Deadline: **12 Jan 2027, 19:30** (report PDF + code ZIP). Sources: `docs/analysis/`, `docs/university/ASSIGNMENT_BRIEF.md`, `PROGRESS.md`.
Mark items `[x]` as they are done.

## 1. Fix the code (blocks the demo, 10 marks)

- [x] Repair `app/ui/streamlit_app.py`: indentation errors (lines 39, 71-72), mixed CRLF/LF line endings, and the `msg` NameError on a fresh session.
- [x] Cache `HallucinationDetector` as a shared dependency (`api/dependencies.py`). The NLI model currently reloads on every `/chat` request.
- [x] Remove or gate `ChatRequest.mode` so public callers can't disable the guardrails. Keep an admin or eval-only path for the ablation.
- [x] Replace `ADMIN_TOKEN=change_me`, refuse to start with the default, use a constant-time compare, and keep `.env` out of the submission ZIP.
- [x] Fix the `run_evaluation.py:54-66` bug: the low-relevance branch has no return, so the generator still runs and overwrites results.
- [x] Refresh the retriever's collection after `/admin/ingest` (the cached handle likely goes stale). Test it, since this is only inferred.
- [x] Add timeouts and one retry around Ollama calls. If the judge fails, fall back to NLI only instead of scoring 0.0.
- [x] `requirements.txt`: added `torch` and `numpy`, removed unused `pandas`/`pyarrow`/`python-dotenv`.
- [ ] Pin versions: run `pip freeze` in the venv that has a known-good run (needs the working environment, not done yet).
- [x] Make the thresholds single-source: share the KB sampling function and the 0.55 and 5 constants between ingestion and `build_eval_set.py`; take the 0.7 in the judge prompt from config.
- [x] Tidy: remove the unused `embed_document`, remove the leaked `D:/Thesis` path in `.vscode/settings.json`, and make `LOG_LEVEL` work.
- [x] Update the Windows paths in `PROGRESS.md` quick start to macOS (venv path, commands).

## 1b. Remaining code issues (from the analysis PDFs)

- [x] Ollama-down returns 503; empty-claim NLI; strict `DETECTOR_MODE`; refusal text without model quotes; re-ingest refresh verified (QA_REPORT Addendum 4).
- [ ] Decide and document: `partially_supported` answers are delivered; the 0.35 cutoff is `GROUNDING_THRESHOLD / 2`.
- [ ] Deployment limitations to state in the thesis: no rate limiting or auth on `/chat`, no CORS or HTTPS, synchronous `/admin/ingest`.
- [ ] Chunking (900 chars): tail chunks lose the "Question:" header; changing it needs re-index plus re-evaluation.
- [ ] Pin the Hugging Face dataset revision; add `pyproject.toml` so scripts need no `sys.path` patching; remove unused `APP_ENV`.

## 2. Improve detection quality (the research core, 40 + 20 marks)

- [x] Invented sentence inside a correct answer: now cut by a sentence filter (unsupported by NLI and at least half new words); 78 of 78 injected sentences kept from the user, no new false refusals. See `docs/EVALUATION_V2.md`.
- [ ] Remaining gap: invented claims that reuse document words are only removed 54% of the time. Idea: also cut claims the judge lists as unsupported, measuring false alterations first.

- [x] NLI now uses sentence-window premises (top 3 per claim): false refusals 30% -> 16%, no extra unanswerable answers (see QA_REPORT Addendum 3).
- [ ] Claim extraction (optional now): filter non-factual sentences ("Please contact support") before NLI (ADR-07). Test it as a hypothesis, since NLI's 85% in-domain refusal may come from this.
- [x] Self-refusal check now length-limited, wording unified, one wider-context retry (still a substring check; structured output is a possible upgrade).
- [x] Judge bug fixed: it mistook stored customer questions for claims (A8 scored 0.0 every time). Still open: scores saturate at 0.8; try a larger judge or per-claim judging (ADR-05).
- [ ] Remaining false refusals: slang/typo questions below the retry trigger (idea: intent-agreement trigger or spelling normalization, leak-test first).
- [ ] Re-tune thresholds on labeled data: NLI entail 0.4, relevance gate (about 0.68 separated better), `GROUNDING_THRESHOLD`, `TOP_K`.
- [x] Hard-block answers that contain numbers not in the documents (ADR-04 follow-up). Still open: other partially_supported answers ship.
- [x] Basic input-side injection screening (`input_filter.py`) and unsupported-figure check (`figure_check.py`).
- [x] Judge prompt hardened against instructions inside the answer (not shown to be exploitable).
- [ ] Optional: replace the keyword injection filter with a classifier.
- [ ] Optional: compare against the verbatim-retrieval baseline (return the matched KB answer, ADR-01).

## 3. Evaluation (20 marks, needs real rigor)

- [x] Rebuild the eval set without the gate pre-filter and with more questions (aim for 100+; the 35-question set is too small).
- [x] Faithfulness without human labels: reference-reply similarity plus a synthetic-fault test (see `docs/EVALUATION_V2.md`). Decision: no human labelling; state this limit in the dissertation. (human or checked), so the metric measures hallucination detection and not just "refuse unanswerable questions".
- [x] Rename the current metrics to "refusal correctness" until that is done (ADR-15).
- [x] Report the baseline (`none`) against each mitigation mode (`nli`, `llm`, `both`), as the proposal promises (hallucination rate, factuality, answer quality before and after).
- [x] Fix reproducibility: temperature 0 or a seed, a config hash in the resume file, and a pinned dataset revision.
- [x] Rerun the full evaluation after the fixes (about 32 min per 35 questions on CPU, so plan the time) and update `reports/`.
- [~] (guard, robustness, NLI and metric tests exist; claim-extractor and verdict-boundary tests still missing) Add pytest tests for claim extraction, the detector's verdict boundaries, `is_self_refusal` and the refusal messages. The rubric asks for a test plan and results.

## 4. Documents required by the university

- [ ] **Ethics form**: complete the online faculty ethics form and submit the emailed PDF to its Canvas assignment. Check whether it is done already.
- [ ] **Interim review**: report (at most 2,500 words) plus a draft literature review chapter. Confirm the due date with the supervisor.
- [ ] Check the University's **Student Use of AI Policy** and decide what to declare.
- [ ] Confirm the proposal's scope changes (Ollama and Chroma instead of LangChain) with the supervisor and record them.

## 5. Dissertation report (target at most 20,000 words, template + single PDF)

- [ ] Download the official template and set up the chapter files.
- [ ] Ch.1 Introduction: problem, aims, objectives, research questions (from the proposal), industrial relevance.
- [ ] Ch.2 Literature review: RAG, hallucination causes, detection (NLI, LLM-as-judge, RAGAS), mitigation. Broad and deep; start from the 3 proposal references.
- [ ] Ch.3 Methodology: alternatives considered and justified (ADRs 01-16 are the raw material), risk analysis, data-loss recovery strategy, system design, architecture diagram.
- [ ] Ch.4 Results: eval tables and plots from `reports/`.
- [ ] Ch.5 Discussion and recommendations: honest limitations (gate separation, judge saturation, small n, label construction), deployment recommendations.
- [ ] Ch.6 Conclusions and future work.
- [ ] Legal, social and ethical section (full review: synthetic data, banking risk of wrong answers, bias, privacy, regulation).
- [ ] Appendices: install and run instructions, prompts, config table.
- [ ] Harvard references, attribution of all third-party code, models and datasets.
- [ ] Proofread; use the student ID only, with no name in the file or document.

## 6. Submission and demo

- [ ] Clean ZIP: source, data, docs, user documentation, no `.env`, `__pycache__`, `storage/`, or `.DS_Store`.
- [ ] Delete orphan Chroma segment directories (with the app stopped) or regenerate `storage/`.
- [ ] Verify a fresh install from the README instructions on a clean venv.
- [ ] Prepare the presentation and a live, unscripted demo (10 marks each). Test with Ollama running.
- [ ] Submit the PDF and the ZIP on separate Canvas assignments before 12 Jan 2027, 19:30, leaving time for upload problems.

## Suggested order

1. Section 1 (a week, unblocks everything).
2. Ethics form and interim review check (Section 4), as these have their own deadlines.
3. Sections 2 and 3 together, in about 2-3 weeks, since evaluation depends on the fixes.
4. Section 5 writing in parallel from the start; the literature review first.
5. Section 6 in the final week.
