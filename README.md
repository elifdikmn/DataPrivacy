# GPT Plugin Privacy Risk Analysis & RAG Assistant

**Background:** This project was developed during an internship at the University of Bologna.

A data science analysis of what data GPT plugins (GPT Actions) actually collect, paired with a Retrieval-Augmented Generation (RAG) chatbot that lets you ask questions about the findings and get grounded, source-backed answers.

## Overview

Using the `gpt-data-exposure` dataset — 12,811 parameter records collected by 4,592 unique GPT plugins — this project analyzes what kinds of data these plugins request, how much of it falls into privacy-sensitive categories, how reliably that can be predicted or clustered automatically, and whether plugins' privacy policies disclose it. The analysis (four Jupyter notebooks, Parts 1–4) feeds a RAG chatbot that answers questions about the findings with text grounded in the actual results, plus the matching chart from the analysis.

## Research Questions

| # | Question | Finding |
|---|---|---|
| 1 | What is the category distribution, and what share of collected data is sensitive? | Distribution is skewed (top 3 categories = 48.2% of all data); the 4 sensitive categories (Security credentials, Personal information, Health information, Finance information) together make up **7.3%** of all records |
| 2 | Does the rate of writing a parameter description differ between sensitive and non-sensitive categories? | Statistically significant (p ≈ 0.006) but the effect size is negligible (Cramér's V ≈ 0.024) — sensitivity barely predicts whether a description exists |
| 3 | How accurately can a parameter's category be predicted from its name and description? | Baseline TF-IDF + Logistic Regression: 68.9% accuracy, 46.8% macro F1. The model selected in Part 4 (word + character n-grams, balanced class weights) reaches **76.2% accuracy [95% CI 74.6–77.8%] and 0.642 macro F1 [0.608–0.678]**, and raises Finance-information recall from 14.3% to 64.3%. A spaCy word-vector model trails the baseline overall (54.8% accuracy) |
| 4 | Do plugins naturally cluster into risky vs. safe groups? | No clean binary split — K-Means (K = 10, silhouette 0.27, i.e. weak structure) surfaces **functional/thematic groups** instead (personal information & messaging, security credentials, location & weather, e-commerce, metadata-only, ...), with the sensitive-category share spread across them (**0%–35.1%**) rather than jumping between two clusters |
| 5 | Can records mislabeled "Other" be identified automatically? | Partially — only **11.6%** of "Other" records get a prediction score above 0.5; that subset includes plausible mislabeled sensitive data (e.g. parameters literally named `email`, `key`, `token`), but these are unverified review candidates, not confirmed errors |
| 6 | Do plugins disclose what they collect in their privacy policies? | Across 308 parameters with comparable policy text, **90.3% (278) are never disclosed**; the rest are disclosed clearly (5.2%, 16), vaguely (2.3%, 7), incorrectly/misleadingly (1.9%, 6) or ambiguously (0.3%, 1) |
| 7 | Are the sensitive parameters also the undisclosed ones? | 20 of the 308 audited parameters map to a sensitive category, and **18 of those 20 (90.0%) are undisclosed** — 95% Wilson CI [69.9%, 97.2%], wide because the sample is small |

Research Questions 6 and 7 use a separate audit dataset published by the source paper's authors (see **Data Sources & Attribution** below). The audit logic (`item_disclosure_status` in Part 3) supports all 5 of the source paper's disclosure labels (CLEAR/VAGUE/AMBIGUOUS/INCORRECT/OMITTED), prioritized in that order.

## Methodology

- **EDA & statistics** (Part 1, `bolum1_eda.ipynb`): category distribution, missing-value handling, a chi-square test of independence (with Cramér's V for effect size) comparing description-writing rates between sensitive and non-sensitive parameters.
- **Classification** (Part 2, `bolum2_modelleme.ipynb`): two parallel category-prediction models on the same train/test split — TF-IDF + Logistic Regression as the baseline, and spaCy word-embedding vectors + Logistic Regression as a comparison — plus per-class precision/recall/F1, feature importance, and a confusion-matrix analysis.
- **Clustering & application** (Part 3, `bolum3_uygulama.ipynb`): plugins are profiled by the proportion of each category they collect (not raw counts), then K-Means clustered (K chosen by silhouette score) to look for natural risk groupings; the Part 2 model is also re-applied to reclassify ambiguous "Other" records; the privacy-policy audit (RQ6) and its intersection with the sensitive categories (RQ7) close the notebook.
- **Validation & confidence intervals** (Part 4, `bolum4_dogrulama.ipynb`): three candidate models are compared on a validation split carved out of the training set; the selected model and the baseline are then measured on the test set with a paired, class-stratified bootstrap (2,000 resamples). Sensitive-category recall uses Wilson intervals. Implementation: `analysis/model_validation.py`, `analysis/metrics.py`.
- **What counts as "sensitive" data:** the 25 `main_data_type` category labels themselves come directly from the dataset — they are not something this project invented. Which four of those categories count as **"sensitive"**, however, is this project's own methodological choice, based on the "special category data" concept in GDPR (EU) and HIPAA (US) — i.e., authentication/identity credentials, health data, financial information, and personally identifying information. This is an editorial decision, not a label present in the source data. Other categories that also carry some privacy risk — e.g. Location, Identifier, Web/network data — are deliberately left out of this definition because they don't fall under GDPR/HIPAA's narrower "special category data" concept.
- **Comparison with the source paper's own classifier:** the source paper (see Attribution below) classifies parameters with a few-shot GPT-4o classifier, reaching 92.83% category accuracy; this project instead uses classical ML (TF-IDF + Logistic Regression, 68.9% baseline / 76.2% selected model) to keep the pipeline fully local, free, and reproducible without an LLM API dependency.

## Limitations

- **Train/test split unit.** The split is made per parameter record. A GPT Action usually contributes several parameters, so most test-set Actions also have other parameters in the training set (2,664 of 2,905; see `data_quality` in `project_facts.json`). Exact duplicates are rare (34 test inputs) and removing them barely changes the scores. The reported scores therefore describe how well the model labels *new parameters*; performance on *entirely unseen Actions* is expected to be lower. The confidence intervals also assume independent records.
- **Model selection.** The test set had already been inspected before Part 4, so the Part 4 numbers are exploratory estimates rather than a fresh external evaluation.
- **Clustering.** The best K (10) sits at the upper end of the searched range (2–10) and the silhouette score is low (0.27); some clusters are very small. Cluster IDs are arbitrary, and the sensitive-category share is not a validated risk score.
- **Sample sizes.** RQ7 rests on 20 parameters, so its interval is wide.

## RAG Chatbot

The notebooks (Parts 1–4) are the project's analysis engine — all the data science and ML work (data cleaning, statistical testing, classification models, clustering) happens there. The RAG chatbot is a presentation/access layer on top of that: it turns the analysis's results into an interactive interface that a non-technical user can question in plain language and explore through supporting charts. The chatbot doesn't run any new analysis of its own — it makes the existing analysis's findings explorable.

The chatbot lets you ask questions about the analysis in plain language and get an answer grounded in the actual data, with a supporting chart. Alongside free-text questions, the interface also offers a set of suggested-question chips covering the project's research questions, so a visitor can explore the findings without needing to know what to ask first.

- **Retrieval**: three separate FAISS indices — individual parameter records, the notebooks' written findings, and the privacy-policy audit — searched independently and merged, so a small set of high-value findings never gets crowded out by the much larger record index.
- **Embeddings**: `sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`), run locally and free.
- **LLM**: Anthropic Claude (`claude-haiku-4-5`), used only to phrase the answer from retrieved context — never to invent or compute numbers on its own (see Numeric grounding below).
- **Answer style**: the system prompt (`backend/app/llm.py`) targets non-specialist readers — the direct answer first, two to four sentences by default (longer only when the user asks), plain language with technical terms briefly explained, and the one to three most important terms or numbers in **bold**, which the frontend renders.
- **Follow-up questions**: the frontend sends the last few messages with each question, so follow-ups such as "and its confidence interval?" keep their context; the previous question is also added to the retrieval query.
- **Visualizations**: no chart is generated live. Each of the chatbot's suggested questions is mapped ahead of time to a specific pre-built chart exported from the notebooks (`backend/app/chart_mapping.py`); an unmapped free-text question simply gets no chart, rather than a guessed or mismatched one.
- **Numeric grounding**: `backend/app/project_facts.json` holds every verified statistic from the analysis (accuracy/F1 scores, percentages, counts, confidence intervals), generated from the notebooks by `analysis/rebuild.py`. It is sent with every request as part of the system prompt, which requires the model to copy numbers from this table rather than recomputing or recalling them. Because the instructions and the table are identical on every request, they are marked for prompt caching. A lightweight post-hoc diagnostic (`facts.verify_answer_numbers`) logs numbers in an answer that don't match any table value (also accepting percentage and Turkish decimal-comma forms); it is a review aid, not a proof that each number is attached to the right claim.

## Tech Stack

**Analysis:** pandas, NumPy, scikit-learn, spaCy, SciPy, matplotlib, seaborn

**Backend:** FastAPI, Anthropic SDK, sentence-transformers, FAISS

**Frontend:** React, Framer Motion

## Project Structure

```
DataPrivacy/
├── notebooks/
│   ├── bolum1_eda.ipynb          # Part 1: EDA, category distribution, chi-square test
│   ├── bolum2_modelleme.ipynb    # Part 2: TF-IDF vs. embedding classification, feature importance
│   ├── bolum3_uygulama.ipynb     # Part 3: clustering, "Other" reclassification, policy audit (RQ6, RQ7)
│   ├── bolum4_dogrulama.ipynb    # Part 4: model selection, bootstrap confidence intervals
│   └── exported_charts/          # PNGs exported for the chatbot's visualization library
├── analysis/
│   ├── rebuild.py                # Re-runs the notebooks, refreshes facts, reports and charts
│   ├── model_validation.py       # Validation-only model selection (Part 4)
│   ├── metrics.py                # Paired bootstrap and Wilson intervals
│   └── results/                  # Generated reports (RESULTS.md, JSON/CSV intervals)
├── backend/
│   ├── app/
│   │   ├── main.py               # FastAPI app (/ask, /health, /ready)
│   │   ├── rag.py                # Retrieval + context + answer orchestration
│   │   ├── llm.py                # Claude wrapper, system prompt (audience and answer style)
│   │   ├── retrieval.py          # FAISS search across the 3 indices
│   │   ├── indexing.py           # Builds the FAISS indices from source data
│   │   ├── index_state.py        # Detects stale indices after source changes
│   │   ├── facts.py              # Loads project_facts.json, numeric diagnostic
│   │   ├── project_facts.json    # Verified statistics table (the grounding source)
│   │   └── chart_mapping.py      # Fixed question → chart-file mapping
│   ├── data/                     # Main dataset (data_entries_final.json)
│   ├── knowledge/                # Written findings fed into the RAG knowledge base (hand-edited)
│   ├── final_results/            # Privacy-policy audit dataset (see Attribution)
│   └── static/charts/            # Pre-built chart images/HTML served to the frontend
├── frontend/
│   └── src/                      # React chat interface
├── tests/                        # unittest suite (analysis metrics, backend, API)
├── requirements-analysis.txt     # Notebook / analysis environment
├── requirements-test.txt         # Minimal environment for the test suite
└── .github/workflows/tests.yml   # CI: tests + frontend production build
```

## How to Run

**Backend**

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then add your ANTHROPIC_API_KEY
python -m app.indexing      # builds the FAISS indices (first run only, or after data/knowledge changes)
uvicorn app.main:app --reload
```

Backend runs at `http://127.0.0.1:8000`. By default it accepts requests from the React dev server (`http://localhost:3000` and `http://127.0.0.1:3000`); set `CORS_ORIGINS` in `backend/.env` if the frontend runs elsewhere.

**Frontend**

```bash
cd frontend
npm install
npm start
```

Frontend runs at `http://localhost:3000`.

**Tests** (from the repository root)

```bash
pip install -r requirements-test.txt
python -m unittest discover -s tests -v
```

The same tests and a frontend production build run on every push via GitHub Actions.

**Regenerating the analysis** (from the repository root)

```bash
pip install -r requirements-analysis.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m analysis.rebuild
```

This re-executes all four notebooks and refreshes `project_facts.json`, the reports in `analysis/results/` and the charts in `backend/static/charts/` and `notebooks/exported_charts/`. The hand-edited texts in `backend/knowledge/` are **not** overwritten: generated drafts are written to `analysis/results/generated_knowledge/` for review (add `--update-knowledge` to overwrite the curated files instead). Afterwards, rebuild the retrieval index (`python -m app.indexing` in `backend/`) and restart the backend.

## Data Sources & Attribution

- **Source paper**: Wu, Y., Jaff, E., Yang, K., Zhang, N., & Iqbal, U. (2025). *An In-Depth Investigation of Data Collection in LLM App Ecosystems*. In Proceedings of the 2025 ACM Internet Measurement Conference (IMC '25). https://doi.org/10.1145/3730567.3732912 — this paper is the source of both the main dataset (12,811 parameter records, the 24/25-category and 145-data-type taxonomy) and the privacy-policy audit methodology (the CLEAR/VAGUE/AMBIGUOUS/INCORRECT/OMITTED disclosure-labeling scheme).
- **Main dataset**: [`gpt-data-exposure`](https://github.com/llm-platform-security/gpt-data-exposure) — 12,811 parameter records collected by 4,592 GPT plugins, from the paper above, obtained by contacting the project's authors.
- **Privacy-policy audit dataset** (`backend/final_results/`): produced and published by the source paper's own authors, not generated by this project's own analysis pipeline — used as-is for Research Questions 6 and 7.
- **License:** the MIT license in `LICENSE` covers this project's own code and texts. The datasets in `backend/data/` and `backend/final_results/` belong to their original authors and are not covered by it; check their terms before reusing or redistributing them.
