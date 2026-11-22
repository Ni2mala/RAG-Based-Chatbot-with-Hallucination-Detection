# Assignment Brief: MSc AI Project (771952_C25_T3A)

Source: Canvas, "Assignment 2: Report and Portfolio (80%)", pasted 2026-11-22. This file keeps the parts that affect the work. Marking detail is in `Grading_Rubric.pdf`.

## Key facts

| Item | Value |
|---|---|
| Weight | 80% of the module (the dissertation is worth one third of the degree) |
| Due | **12 Jan 2027, 19:30** (submission closes 19 Jan 2027, 19:30) |
| Report format | **Single PDF** via Canvas, using the official dissertation template |
| Code submission | Separate Canvas assignment: **one ZIP** with source code, data, program listings and user documentation |
| Word limit | **20,000 words** (brief) excluding appendices, references and bibliography. Guidance text says "about 25,000 words / about 60 pages" as a maximum. **Use 20,000 to be safe.** |
| Identity | Use **only the student ID**. No name in the report or the file name. |
| Plagiarism | All work goes through plagiarism detection |

## Over-length penalties

- Up to 10% over: no penalty.
- 10-20% over: 10% of marks deducted, or reduction to the pass mark, whichever is higher.
- 20-50% over: maximum mark is the pass mark.
- More than 50% over: not marked, zero recorded.

## Marking scheme (100 marks) and programme competencies

| Criterion | Marks | PC |
|---|---|---|
| Project definition, background and context | 10 | PC01 |
| Problem analysis and solution design | 20 | PC03 |
| Technical achievement | 40 | PC05 |
| Critical evaluation and reflection | 20 | PC10, PC11 |
| Legal, social and ethical issues | 5 | PC03 |
| Presentation and referencing | 5 | PC01 |

Marked by two examiners. The presentation and technical demonstration (10 marks each) are scored separately (see the rubric PDF).

## What top marks require (70%+)

- **Definition and background (10):** very clear scope; broad and deep literature review; industrial relevance clearly defined.
- **Problem analysis and solution design (20):** a good range of alternative solutions rigorously evaluated; risks considered with mitigation planned; the chosen solution explained in depth with a comprehensive plan and design. Every decision is justified and recorded in the report.
- **Technical achievement (40):** a complete working solution, demonstrated with substantial results; state-of-the-art techniques; rigorous testing; new techniques may be developed. Significant development, not just reliance on existing tools.
- **Critical evaluation (20):** objective appraisal against the stated goals and relevant standards; strengths and weaknesses; further refinement and future development; **clear recommendations for deployment**.
- **Ethics and legal (5):** a full ethics review; wider ethical and legal ramifications thoroughly considered.
- **Presentation and referencing (5):** consistent, professional, no spelling or grammar errors; a comprehensive and correctly formatted set of references (Harvard) covering different source types.

## Required report structure

1. Title page
2. Abstract
3. Contents, list of tables and figures
4. Chapter 1: Introduction
5. Chapter 2: Literature review
6. Chapter 3: Methodology
7. Chapter 4: Results
8. Chapter 5: Discussion of results and recommendations
9. Chapter 6: Conclusions
10. References

## Other portfolio components

1. **Ethics approval.** Complete the online faculty ethics form. The emailed PDF copy must be submitted to the Canvas ethics assignment. Supervisor and a second staff member review it. A full ethics request is needed only if significant issues arise. The proposal states the data is public or synthetic (Bitext), so no real customer records are used.
2. **Interim review.** Halfway through the project period. Two items:
   - A review report of **no more than 2,500 words**: introduction (note any scope, aim or objective changes since the proposal); progress against the original plan; achievements (milestones, deliverables, prototypes, screenshots, preliminary results); a revised plan and task list, including how results will be evaluated.
   - A draft dissertation chapter (usually the literature review).
   Software is expected to be incomplete at this stage.

## Expectations that apply directly to this project

- Analyse, design and plan **before** coding. Consider several methods and justify the choice (here: why RAG with an NLI and LLM-judge guardrail, why Ollama and Chroma, and why not LangChain, which the proposal named).
- Manage risk, and have a strategy for recovering from data loss (version control, backups of `storage/`, `data/`, `reports/`).
- Attribute all third-party code, libraries, models and datasets in the code listings **and** in the report (Ollama, llama3.2, nomic-embed-text, DeBERTa NLI, Chroma, Bitext, FastAPI, Streamlit).
- Software should be readable and well documented.
- The report must include **full instructions to install and run** the software.
- Follow the University's ethics, data protection and retention guidelines. Consider the long-term implications of the work.
- The University's **Student Use of AI Policy** applies. Check it before using AI assistance in the report or code, and declare any use as it requires.
- Agree deadlines for each phase with the supervisor (Muhammad Khalid) and meet at regular intervals.

## Links to the project

- Proposal: `202514613_Dissertation_Proposal_Form.pdf`
- Rubric: `Grading_Rubric.pdf`
- Status and known issues: `../../PROGRESS.md`, `../analysis/`
