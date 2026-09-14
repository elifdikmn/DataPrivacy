# GPT Plugin Privacy Risk Analysis & RAG Assistant

**Background:** This project was developed during an internship at the University of Bologna.

A data science analysis of what data GPT plugins (GPT Actions) actually collect, paired with a Retrieval-Augmented Generation (RAG) chatbot that lets you ask questions about the findings and get grounded, source-backed answers.

## Overview

Using the `gpt-data-exposure` dataset — 12,811 parameter records collected by 4,592 unique GPT plugins — this project analyzes what kinds of data these plugins request, how much of it falls into privacy-sensitive categories, and how reliably that can be predicted or clustered automatically. The analysis (three Jupyter notebooks) feeds a RAG chatbot that answers questions about the findings with text grounded in the actual results, plus the matching chart from the analysis.

## Research Questions

| # | Question | Finding |
|---|---|---|
| 1 | What is the category distribution, and what share of collected data is sensitive? | Distribution is skewed (top 3 categories = 48.2% of all data); the 4 sensitive categories (Security credentials, Personal information, Health information, Finance information) together make up **7.3%** of all records |
| 2 | Does the rate of writing a parameter description differ between sensitive and non-sensitive categories? | Statistically significant (p ≈ 0.006) but the effect size is negligible (Cramér's V ≈ 0.024) — sensitivity barely predicts whether a description exists |
| 3 | How accurately can a parameter's category be predicted from its name and description? | TF-IDF + Logistic Regression: **68.9% accuracy, 46.8% macro F1**. A word-embedding model trails overall but wins on very-low-sample classes. The model systematically defaults to "Other" when unsure |
| 4 | Do plugins naturally cluster into risky vs. safe groups? | No clean binary split — clustering surfaces **functional/thematic groups** instead (finance, travel, messaging, general-purpose, ...), with sensitive-data share spread gradually across them (0%–16.1%) rather than jumping between two clusters |
| 5 | Can records mislabeled "Other" be identified automatically? | Partially — only **11.6%** of "Other" records get a high-confidence re-classification, but that subset does catch genuinely mislabeled sensitive data (e.g. parameters literally named `email`, `key`, `token`) |

A sixth, supplementary finding — whether plugins disclose what they collect in their actual privacy policies — comes from a separate audit dataset (see **Data Sources & Attribution** below): across 308 comparable parameters, **90.6% are never disclosed** in the plugin's privacy policy at all. Note that the source paper labels disclosure with 5 categories (CLEAR/VAGUE/AMBIGUOUS/INCORRECT/OMITTED); this project's audit logic (`item_disclosure_status` in Part 3) only distinguishes 4 of them (CLEAR/VAGUE/INCORRECT/UNDISCLOSED) — it has no explicit `AMBIGUOUS` branch, so the rare records where `AMBIGUOUS` is the only informative label (1 of the 308) are folded into `UNDISCLOSED` rather than reported separately.

## Methodology

- **EDA & statistics** (Bölüm 1): category distribution, missing-value handling, a chi-square test of independence (with Cramér's V for effect size) comparing description-writing rates between sensitive and non-sensitive parameters.
- **Classification** (Bölüm 2): two parallel category-prediction models on the same train/test split — TF-IDF + Logistic Regression as the baseline, and spaCy word-embedding vectors + Logistic Regression as a comparison — plus per-class precision/recall/F1, feature importance, and a confusion-matrix analysis.
- **Clustering & application** (Bölüm 3): plugins are profiled by the proportion of each category they collect (not raw counts), then K-Means clustered (K chosen by silhouette score) to look for natural risk groupings; the Bölüm 2 model is also re-applied to reclassify ambiguous "Other" records.
- **What counts as "sensitive" data:** the 25 `main_data_type` category labels themselves come directly from the dataset — they are not something this project invented. Which four of those categories count as **"sensitive"**, however, is this project's own methodological choice, based on the "special category data" concept in GDPR (EU) and HIPAA (US) — i.e., authentication/identity credentials, health data, financial information, and personally identifying information. This is an editorial decision, not a label present in the source data. Other categories that also carry some privacy risk — e.g. Location, Identifier, Web/network data — are deliberately left out of this definition because they don't fall under GDPR/HIPAA's narrower "special category data" concept.
- **Comparison with the source paper's own classifier:** the source paper (see Attribution below) classifies parameters with a few-shot GPT-4o classifier, reaching 92.83% category accuracy; this project instead uses a classical ML approach (TF-IDF + Logistic Regression, 68.9% accuracy) to keep the pipeline fully local, free, and reproducible without an LLM API dependency.

## RAG Chatbot

The notebooks (Bölüm 1, 2, 3) are the project's analysis engine — all the data science and ML work (data cleaning, statistical testing, classification models, clustering) happens there. The RAG chatbot is a presentation/access layer on top of that: it turns the analysis's results into an interactive interface that a non-technical user can question in plain language and explore through supporting charts. The chatbot doesn't run any new analysis of its own — it makes the existing analysis's findings explorable.

The chatbot lets you ask questions about the analysis in plain language and get an answer grounded in the actual data, with a supporting chart. Alongside free-text questions, the interface also offers a set of suggested-question chips covering the project's 5 main research questions, so a visitor can explore the findings without needing to know what to ask first.

- **Retrieval**: three separate FAISS indices — individual parameter records, the notebooks' written findings, and the privacy-policy audit — searched independently and merged, so a small set of high-value findings never gets crowded out by the much larger record index.
- **Embeddings**: `sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`), run locally and free.
- **LLM**: Anthropic Claude (`claude-haiku-4-5`), used only to phrase the answer from retrieved context — never to invent or compute numbers on its own (see Grounding below).
- **Visualizations**: no chart is generated live. Each of the chatbot's suggested questions is mapped ahead of time to a specific pre-built chart exported from the notebooks (`backend/app/chart_mapping.py`); an unmapped free-text question simply gets no chart, rather than a guessed or mismatched one.
- **Numeric grounding**: `backend/app/project_facts.json` holds every verified statistic from the analysis (accuracy/F1 scores, percentages, counts) read directly from the notebooks' actual output. It's appended to every request's context, and the system prompt requires the model to copy numbers from this table rather than recomputing or recalling them — the failure mode this exists to prevent. A lightweight post-hoc check (`facts.verify_answer_numbers`) flags any number in a generated answer that doesn't appear in the table, logged for review.

## Tech Stack

**Analysis:** pandas, NumPy, scikit-learn, spaCy, SciPy, matplotlib, seaborn

**Backend:** FastAPI, Anthropic SDK, sentence-transformers, FAISS

**Frontend:** React, Framer Motion

## Project Structure

```
DataPrivacy/
├── notebooks/
│   ├── bolum1_eda.ipynb          # EDA, category distribution, chi-square test
│   ├── bolum2_modelleme.ipynb    # TF-IDF vs. embedding classification, feature importance
│   ├── bolum3_uygulama.ipynb     # Clustering, "Other" reclassification, policy audit
│   └── exported_charts/          # PNGs exported for the chatbot's visualization library
├── backend/
│   ├── app/
│   │   ├── main.py               # FastAPI app (/ask, /health)
│   │   ├── rag.py                # Retrieval + context + answer orchestration
│   │   ├── llm.py                # Claude wrapper, system prompt
│   │   ├── retrieval.py          # FAISS search across the 3 indices
│   │   ├── indexing.py           # Builds the FAISS indices from source data
│   │   ├── facts.py              # Loads project_facts.json, verifies answer numbers
│   │   ├── project_facts.json    # Verified statistics table (the grounding source)
│   │   └── chart_mapping.py      # Fixed question → chart-file mapping
│   ├── data/                     # Main dataset (data_entries_final.json)
│   ├── knowledge/                # Written findings fed into the RAG knowledge base
│   ├── final_results/            # Privacy-policy audit dataset (see Attribution)
│   └── static/charts/            # Pre-built chart images/HTML served to the frontend
└── frontend/
    └── src/                      # React chat interface
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

Backend runs at `http://127.0.0.1:8000`.

**Frontend**

```bash
cd frontend
npm install
npm start
```

Frontend runs at `http://localhost:3000`.

## Limitations

- The chatbot's charts are reliable — they're static images exported directly from the notebooks' actual output, not generated on the fly. The LLM-generated explanatory *text*, however, has occasionally gotten numeric details wrong (e.g. attributing one category's F1 score to a different, similar category; conflating "the two biggest clusters by plugin count" with "the two highest-risk clusters"). The `project_facts.json` grounding mechanism above was added specifically to reduce this risk, but it's a mitigation, not a guarantee — as a general principle for any RAG-based system, numeric claims it produces should be cross-checked against the source data before being relied on for a decision or publication.
- The embedding-based classification model uses spaCy word vectors instead of `sentence-transformers`, because Hugging Face access was blocked by network policy in the analysis environment — results should be read with that substitution in mind.
- Clustering results are directionally useful but not sharply separated (silhouette scores of 0.17–0.25 across all tested K).
- The "Other" reclassification model's predictions were not validated against ground truth (none exists) — they're a review-priority signal, not a verified relabeling.
- The privacy-policy audit (Research Question 6) uses a separate, smaller sample with its own labeling taxonomy — its numbers are not directly comparable to the main dataset's statistics.

## Data Sources & Attribution

- **Source paper**: Wu, Y., Jaff, E., Yang, K., Zhang, N., & Iqbal, U. (2025). *An In-Depth Investigation of Data Collection in LLM App Ecosystems*. In Proceedings of the 2025 ACM Internet Measurement Conference (IMC '25). https://doi.org/10.1145/3730567.3732912 — this paper is the source of both the main dataset (12,811 parameter records, the 24/25-category and 145-data-type taxonomy) and the privacy-policy audit methodology (the CLEAR/VAGUE/AMBIGUOUS/INCORRECT/OMITTED disclosure-labeling scheme).
- **Main dataset**: [`gpt-data-exposure`](<!-- TODO: add exact GitHub URL -->) — 12,811 parameter records collected by 4,592 GPT plugins, from the paper above.
- **Privacy-policy audit dataset** (`backend/final_results/`): produced and published by the source paper's own authors, not generated by this project's own analysis pipeline — used as-is for Research Question 6.
