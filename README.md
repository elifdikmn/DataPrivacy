# GPT Plugin Privacy Risk Analysis & RAG Assistant

**Background:** This project was developed during an internship at the University of Bologna.

A data science analysis of what data GPT plugins (GPT Actions) actually collect, paired with a Retrieval-Augmented Generation (RAG) chatbot that lets you ask questions about the findings and get grounded, source-backed answers. All figures below are generated automatically from a single reproducible pipeline (`analysis/rebuild.py`) rather than typed in by hand, and are backed by bootstrap confidence intervals — see **Validation & Reproducibility** below.

## Overview

Using the `gpt-data-exposure` dataset — 12,811 parameter records collected by 4,592 unique GPT plugins — this project analyzes what kinds of data these plugins request, how much of it falls into privacy-sensitive categories, and how reliably that can be predicted or clustered automatically. The analysis (four Jupyter notebooks) feeds a RAG chatbot that answers questions about the findings with text grounded in the actual results, plus the matching chart from the analysis.

## Research Questions

| # | Question | Finding |
|---|---|---|
| 1 | What is the category distribution, and what share of collected data is sensitive? | Distribution is skewed (top 3 categories = 48.2% of all data); the 4 sensitive categories (Security credentials, Personal information, Health information, Finance information) together make up **7.3%** of all records |
| 2 | Does the rate of writing a parameter description differ between sensitive and non-sensitive categories? | Statistically significant (p ≈ 0.006, 95% Wilson CI [78.6%, 83.6%] vs. [84.0%, 85.3%]) but the effect size is negligible (Cramér's V ≈ 0.024) — sensitivity barely predicts whether a description exists |
| 3 | How accurately can a parameter's category be predicted from its name and description? | A validated **word + character TF-IDF, class-balanced Logistic Regression** model reaches **76.2% accuracy, 64.2% macro F1** (95% CI [74.6%, 77.8%] / [0.608, 0.678]) — a statistically confirmed improvement of **+7.3 points accuracy, +0.175 macro F1** over the original word-only baseline (68.9% / 46.8%). Recall on sensitive categories improves sharply (e.g. Finance information: 14%→64%, Health information: 44%→75%) |
| 4 | Do plugins naturally cluster into risky vs. safe groups? | No clean binary split — clustering over **30,478 deduplicated parameter–plugin pairs** (2,968 eligible plugins) surfaces **functional/thematic groups** instead, with sensitive-data share spread across them (0%–35.1%) rather than jumping between two clusters |
| 5 | Can records mislabeled "Other" be identified automatically? | Partially — only **11.6%** of "Other" records get a high-confidence re-classification, but that subset does catch genuinely mislabeled sensitive data (e.g. parameters literally named `email`, `key`, `token`); ambiguous fine-to-broad category mappings are left unresolved rather than silently guessed |

A sixth, supplementary finding — whether plugins disclose what they collect in their actual privacy policies — comes from a separate audit dataset (see **Data Sources & Attribution** below): across 308 comparable parameters, **90.3% (278) are never disclosed** in the plugin's privacy policy at all; the rest are clearly disclosed (5.2%, 16), vaguely disclosed (2.3%, 7), incorrectly/misleadingly disclosed (1.9%, 6), or ambiguously disclosed (0.3%, 1).

A seventh finding combines RQ1 and RQ6: of the 308 audited parameters, only 20 map (via an explicit, justified hand mapping between the two taxonomies) to one of RQ1's sensitive categories — and of those 20, **18 (90.0%) are undisclosed** (95% Wilson CI [69.9%, 97.2%], n=20). See `backend/knowledge/rq7_hassas_ve_ifsa_edilmemis.md` for the full mapping, method, and caveats (including why this small-sample analysis says nothing about Health information specifically).

## Methodology

- **EDA & statistics** (Part 1): category distribution, missing-value handling, a chi-square test of independence (with Cramér's V for effect size and Wilson confidence intervals) comparing description-writing rates between sensitive and non-sensitive parameters.
- **Classification** (Part 2): a word-only TF-IDF + Logistic Regression baseline, a spaCy word-embedding comparison, and a fine-grained (145-class) model — plus per-class precision/recall/F1, feature importance, and a confusion-matrix analysis.
- **Validation** (Part 4): three candidate models (baseline, word-balanced, word+character-balanced) are compared on a held-out **validation split** (not the test set); the winner by validation macro F1 is refit on the full training data and evaluated once on the original test set, with **2,000-resample paired class-stratified bootstrap** confidence intervals for accuracy/F1 and **Wilson intervals** for sensitive-category recall. The comparison also surfaces a known limitation: most test-set plugins (Actions) also appear in the training set, so these intervals describe performance on this fixed split, not on unseen plugins.
- **Clustering & application** (Part 3): plugins are profiled by the proportion of each category they collect (not raw counts, and with duplicate parameter–plugin pairs removed), then K-Means clustered (K chosen by silhouette score) to look for natural risk groupings; the Part 2 model is also re-applied to reclassify ambiguous "Other" records, with unresolved category mappings kept explicit rather than defaulted to a majority guess.
- **What counts as "sensitive" data:** the 25 `main_data_type` category labels themselves come directly from the dataset — they are not something this project invented. Which four of those categories count as **"sensitive"**, however, is this project's own methodological choice, based on the "special category data" concept in GDPR (EU) and HIPAA (US) — i.e., authentication/identity credentials, health data, financial information, and personally identifying information. Other categories that also carry some privacy risk — e.g. Location, Identifier, Web/network data — are deliberately left out of this definition because they don't fall under GDPR/HIPAA's narrower "special category data" concept. This is an editorial decision, not a label present in the source data.
- **Comparison with the source paper's own classifier:** the source paper (see Attribution below) classifies parameters with a few-shot GPT-4o classifier, reaching 92.83% category accuracy; this project instead uses a classical ML approach (TF-IDF + Logistic Regression) to keep the pipeline fully local, free, and reproducible without an LLM API dependency.

## Validation & Reproducibility

Earlier drafts of this analysis had a real bug — duplicate plugin IDs inside a single record's plugin list were not deduplicated before building each plugin's data-collection profile, inflating 40,261 parameter–plugin rows to what should have been 30,478 distinct pairs — which skewed the Part 3 clustering results. It also typed model-performance numbers into the chatbot's fact table by hand from notebook output, the same failure mode that produced a few numeric mix-ups the chatbot voiced early on.

Both are now fixed structurally rather than patched one at a time:

- **`analysis/rebuild.py`** executes all four notebooks end-to-end and regenerates every downstream artifact — `backend/app/project_facts.json`, the exported chart PNGs, the knowledge-base markdown files, and `analysis/results/` — directly from the executed code, so numbers can no longer drift out of sync with what the notebooks actually compute.
- **`analysis/metrics.py`** implements the paired bootstrap and Wilson-interval logic, unit-tested against `scipy.stats.binomtest`'s reference Wilson implementation and against `sklearn`'s own accuracy/F1 functions.
- **`backend/app/domain.py`** is now the single source of truth for the privacy-policy disclosure priority rules and the fine-to-broad category mapping, shared by both the notebooks and the backend, so they can't silently drift apart.
- **`backend/app/index_state.py`** fingerprints the source data/knowledge/facts and refuses to serve chatbot answers from a stale FAISS index (`/ready` returns `503` until the index is rebuilt after an analysis change).
- The chatbot accepts natural conversational answers in the user’s language, including numbers, F1 and confidence intervals. Complete project facts provide reference values, units and limitations. Numeric checks are nonblocking log diagnostics; they do not guarantee factual correctness.
- `tests/` (`python -m unittest discover -s tests -v`) covers the bootstrap/Wilson math against reference implementations, the corrected deduplicated counts against the raw dataset, the disclosure-audit label counts, the grounded-answer rendering/rejection behavior, and index-freshness handling.

To reproduce all analysis numbers from scratch:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-analysis.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m analysis.rebuild
```

This re-executes all four notebooks and rewrites `backend/app/project_facts.json`, the PNG charts, and the knowledge-base files from that run — nothing above is hand-typed.

## RAG Chatbot

The notebooks (Parts 1–4) are the project's analysis engine — all the data science and ML work (data cleaning, statistical testing, classification models, clustering, validation) happens there. The RAG chatbot is a presentation/access layer on top of that: it turns the analysis's results into an interactive interface that a non-technical user can question in plain language and explore through supporting charts. The chatbot doesn't run any new analysis of its own — it makes the existing analysis's findings explorable.

The chatbot lets you ask questions about the analysis in plain language and get an answer grounded in the actual data, with a supporting chart. Alongside free-text questions, the interface also offers a set of suggested-question chips covering the project's main research questions, so a visitor can explore the findings without needing to know what to ask first.

- **Retrieval**: three separate FAISS indices — individual parameter records, the notebooks' written findings, and the privacy-policy audit — searched independently and merged, so a small set of high-value findings never gets crowded out by the much larger record index.
- **Embeddings**: `sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`), run locally and free.
- **LLM**: Anthropic Claude phrases answers from retrieved context and project facts in natural language, without a required JSON response schema.
- **Visualizations**: no chart is generated live. Each of the chatbot's suggested questions is mapped ahead of time to a specific pre-built chart exported from the notebooks (`backend/app/chart_mapping.py`), including a dedicated chart for the Part 4 validation confidence intervals; an unmapped free-text question simply gets no chart, rather than a guessed or mismatched one.
- **Numeric grounding**: `backend/app/project_facts.json` supplies analysis results, including RQ7 and confidence intervals. The prompt instructs the model to match values to the correct metric, model and unit. Numeric membership diagnostics never reject an answer and may flag legitimate rounding or percentage conversions.
- **Freshness**: `/ready` checks that the FAISS index's source fingerprint matches the current data/knowledge/facts files, and returns `503` if the index is stale, rather than silently serving answers grounded in outdated numbers.

## Tech Stack

**Analysis:** pandas, NumPy, scikit-learn, spaCy, SciPy, matplotlib, seaborn

**Backend:** FastAPI, Anthropic SDK, sentence-transformers, FAISS

**Frontend:** React, Framer Motion

## Project Structure

```
GPTDataPrivacy/
├── notebooks/
│   ├── bolum1_eda.ipynb          # EDA, category distribution, chi-square test
│   ├── bolum2_modelleme.ipynb    # TF-IDF vs. embedding classification, feature importance
│   ├── bolum3_uygulama.ipynb     # Deduplicated clustering, "Other" reclassification, policy audit
│   ├── bolum4_dogrulama.ipynb    # Validation-based model selection + bootstrap confidence intervals
│   └── exported_charts/          # PNGs exported for the chatbot's visualization library
├── analysis/
│   ├── metrics.py                # Paired bootstrap + Wilson interval implementation (unit-tested)
│   ├── model_validation.py       # Fit/validation/test model selection and evaluation
│   ├── rebuild.py                # Single-command pipeline: notebooks → facts → charts → knowledge base
│   └── results/                  # Generated reports (RESULTS.md, JSON intervals, execution manifest)
├── backend/
│   ├── app/
│   │   ├── main.py               # FastAPI app (/ask, /health, /ready)
│   │   ├── rag.py                # Retrieval + context + answer orchestration
│   │   ├── llm.py                # Claude wrapper, system prompt
│   │   ├── retrieval.py          # FAISS search across the 3 indices
│   │   ├── indexing.py           # Builds the FAISS indices from source data
│   │   ├── domain.py             # Shared disclosure-priority and category-mapping rules
│   │   ├── index_state.py        # Source fingerprinting and index-freshness checks
│   │   ├── facts.py              # Loads project_facts.json, renders/validates grounded answers
│   │   ├── project_facts.json    # Verified statistics table (generated, not hand-typed)
│   │   └── chart_mapping.py      # Fixed question → chart-file mapping
│   ├── data/                     # Main dataset (data_entries_final.json)
│   ├── knowledge/                # Written findings fed into the RAG knowledge base
│   ├── final_results/            # Privacy-policy audit dataset (see Attribution)
│   └── static/charts/            # Pre-built chart images served to the frontend
├── tests/                        # Unit tests for analysis math and backend behavior
└── frontend/
    └── src/                      # React chat interface
```

## How to Run

**Reproduce the analysis (optional, before running the app)**

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-analysis.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m analysis.rebuild
```

**Backend**

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then add your ANTHROPIC_API_KEY
python -m app.indexing      # builds the FAISS indices (first run, or after data/knowledge/facts change)
uvicorn app.main:app --reload
```

Backend runs at `http://127.0.0.1:8000`. `/health` checks liveness; `/ready` checks that the retrieval index is current and credentials are configured.

**Frontend**

```bash
cd frontend
npm install
npm start
```

Frontend runs at `http://localhost:3000`.

**Tests**

```bash
python -m unittest discover -s tests -v
cd frontend && CI=true npm run build
```

## Limitations

- **Confidence intervals are conditional, not unconditional guarantees.** The Part 4 bootstrap and Wilson intervals condition on the fixed fitted models and the observed test split; they exclude retraining/model-selection uncertainty and assume independent parameter records.
- **Plugin overlap between train and test.** Most plugins (Actions) in the test set also appear in the training set (2,664 of 2,905), so the validated performance numbers describe this fixed split rather than generalization to entirely new, unseen plugins. Grouped-by-plugin or external evaluation is a natural next step.
- **Charts** are static exports from the notebooks. Conversational text still needs factual evaluation: the model can misinterpret a number or its context even when a reference value exists.
- The embedding-based classification model uses spaCy word vectors instead of `sentence-transformers`, because Hugging Face access was blocked by network policy in the analysis environment — results should be read with that substitution in mind.
- Clustering results are directionally useful but not sharply separated (silhouette score 0.268 at the chosen K).
- The "Other" reclassification model's predictions were not validated against ground truth (none exists) — they're a review-priority signal, not a verified relabeling. Ambiguous fine-to-broad category mappings are left unresolved rather than guessed.
- The privacy-policy audit (Research Question 6) uses a separate, smaller sample with its own labeling taxonomy — its numbers are not directly comparable to the main dataset's statistics.

## Data Sources & Attribution

- **Source paper**: Wu, Y., Jaff, E., Yang, K., Zhang, N., & Iqbal, U. (2025). *An In-Depth Investigation of Data Collection in LLM App Ecosystems*. In Proceedings of the 2025 ACM Internet Measurement Conference (IMC '25). https://doi.org/10.1145/3730567.3732912 — this paper is the source of both the main dataset (12,811 parameter records, the 24/25-category and 145-data-type taxonomy) and the privacy-policy audit methodology (the CLEAR/VAGUE/AMBIGUOUS/INCORRECT/OMITTED disclosure-labeling scheme).
- **Main dataset**: [`gpt-data-exposure`](<!-- TODO: add exact GitHub URL -->) — 12,811 parameter records collected by 4,592 GPT plugins, from the paper above.
- **Privacy-policy audit dataset** (`backend/final_results/`): produced and published by the source paper's own authors, not generated by this project's own analysis pipeline — used as-is for Research Question 6.
