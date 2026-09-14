# Predicting Category from Parameter Name (Model Performance)

## Question

Looking only at a parameter's name (and description, if present), how accurately can its category be predicted?

## Setup

- Input (X): `name` + `description` concatenated text
- Target (y): `main_data_type` (25 classes)
- Stratified train/test split: 10,248 train / 2,563 test

## Baseline model: TF-IDF + Logistic Regression

- **Accuracy: 68.9%**
- **Macro F1: 46.8%**
- Weighted F1: 69.2%

Patterns:
- Small classes have high precision, low recall (the model rarely predicts them, but is right when it does).
- The `Other` class is the opposite: precision 0.36, recall 0.69 — whenever the model is unsure, it systematically defaults to "Other," acting like a catch-all.
- Among the sensitive categories, Security credentials is very well distinguished (F1 0.85), and Health information is also decent (F1 0.61).

## Embedding-based model

Since network access to Hugging Face is blocked by organizational policy in this environment, spaCy's `en_core_web_md` model (GloVe-style 300-dim word vectors, averaged) was used instead of `sentence-transformers`.

- Accuracy: 54.9%, Macro F1: 42.6% — **fell behind TF-IDF**.
- Reason: parameter names are short, made of specific technical terms (like password, api_key); TF-IDF captures exact word matches, while averaging word vectors blurs that sharp signal.
- Exception: the embedding model beat TF-IDF specifically on **very-low-sample classes** (<30 records: Travel, Weather, Real estate, Food and nutrition, E-commerce, Finance information) — transfer learning helps when data is scarce.

## Feature importance (TF-IDF + LogReg)

The most important words determining the sensitive categories:
- Security credentials: key, api_key, token, password, apikey, secret
- Personal information: email, gender, age, firstname, lastname, birthday, nickname
- Health information: patient, disease, surgery (but also some vague words: does, 30 days)
- Finance information: currency, price, budget, asset, annual (but generic phrases like "related", "related filter" also showed up, likely from a repeated description pattern)

The model is very reliable for Security credentials and Personal information; Health and Finance show some signs of pattern memorization.

## Confusion matrix finding

In the row-normalized confusion matrix, the "Other" column forms a visible stripe across nearly every row. 91% of Sports information and 85% of E-commerce data misclassifications land in "Other." Among sensitive categories, half of the misclassified Finance information and Health information records also fall into "Other." Categories barely get confused with each other — the only real source of confusion is "Other."

## Experiment on data_type (145 fine-grained categories)

67 of the 145 classes (2.2% of records) had fewer than 10 samples and were grouped into a "Rare type (merged)" bucket, yielding a 79-class problem.

- Accuracy: 65.4% (close to the main_data_type model)
- Macro F1: only 35.9% (was 46.8% on main_data_type)
- F1 distribution across the 79 classes: the lower quartile (25th percentile) is exactly 0 — the model completely fails on about a quarter of the classes.
- `Other` repeats the same pattern: precision 0.49, recall 0.87.

## Overall conclusion

Category prediction from a parameter's name and description works with moderate success: the best model (TF-IDF + Logistic Regression) reaches **68.9% accuracy** but only **46.8% macro F1**, because it has a systematic tendency to default to the "Other" category on anything it's unsure about — meaning its outputs should be treated with caution.
