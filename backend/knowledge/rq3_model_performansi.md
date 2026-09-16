# Can a Parameter's Category Be Predicted From Its Name and Description?

## Question

Looking only at a parameter's name (and description, if present), how accurately can its `main_data_type` category be predicted automatically?

## Baseline model

A TF-IDF + Logistic Regression baseline, trained on the 25-class `main_data_type` target, reaches 68.94% accuracy and 0.4676 macro F1 (95% CI [67.30%, 70.66%] / [0.4278, 0.4963]). Performance depends heavily on class size: strong on big classes, weak on small ones. Many small classes get precision near 1.00 but low recall — the model is right whenever it does predict that class, but it mostly doesn't predict it at all, drifting to a bigger class instead. The `Other` class shows the opposite pattern (precision 0.36, recall 0.69) — the model dumps many uncertain records into "Other," turning it into a catch-all. Among sensitive categories, Security credentials (F1 0.848) is distinguished well (words like "password", "api_key" are clear signals), while Finance information is the weakest (F1 0.25, recall only 14.3%) — the baseline misses most Finance records entirely.

## Embedding comparison

Since Hugging Face access is blocked in the analysis environment, an embedding comparison uses spaCy's `en_core_web_md` GloVe-style word vectors (not a modern sentence-transformer) with the same Logistic Regression classifier. It trails the baseline overall (accuracy 54.78%, macro F1 0.4251, CI [52.95%, 56.57%] / [0.3842, 0.4606]) because parameter names are short, specific technical terms where exact word matching beats averaged meaning — except on very-low-sample classes (under 30 records), where the embedding model's pretrained language knowledge helps it generalize better than TF-IDF can from so few examples.

## Feature importance

Logistic Regression coefficients show the baseline model mostly relies on meaningful, domain-specific words: `key`, `api_key`, `token`, `password` for Security credentials; `email`, `gender`, `firstname`, `birthday` for Personal information. Health and Finance information show some signs of "pattern memorization" (generic phrases like "does patient" or "related indicator" ranking highly) rather than purely domain vocabulary — a reason to interpret those two categories' predictions with more caution.

## Confusion matrix

The "Other" column forms a visible stripe across almost every row of the baseline's confusion matrix — the bulk of records lost from nearly every category land in "Other." Confusion-matrix rows are normalized over all true records in a category, not just its errors: half of all true Finance information records and half of all true Health information records were predicted as "Other" by the baseline. Aside from "Other," there's little confusion between genuine categories — the model doesn't mix up, say, Personal information with Security credentials.

## Validated improvement (Part 4)

A held-out validation split (not the test set) is used to select among three candidates: the word-only baseline, a word-only model with balanced class weights, and a word **+ character** n-gram model with balanced class weights. The winner by validation macro F1 — **word + character, balanced weights** — is refit on the full training data and evaluated once on the original test set with 2,000-resample paired class-stratified bootstrap confidence intervals:

- Accuracy: 76.20%, 95% CI [74.56%, 77.80%] (baseline: 68.94%)
- Macro F1: 0.6423, 95% CI [0.6084, 0.6777] (baseline: 0.4676)
- Paired improvement: +7.26 accuracy points, 95% CI [5.74, 8.82]; +0.1747 macro F1, 95% CI [0.1358, 0.2231] — both intervals clear of zero, so this is not test-set luck.

The improvement is largest exactly where it matters most — sensitive-category recall: Security credentials 76.4%→94.5%, Personal information 52.9%→80.5%, Health information 43.8%→75.0%, and Finance information 14.3%→64.3% (more than quadrupling; the baseline was missing roughly 6 of every 7 true Finance information records, the validated model catches about 2 of 3).

## Fine-grained (data_type) baseline

Applying the same baseline method to the 145-value `data_type` column (rare classes merged into a single bucket, 79 classes total) gives accuracy 65.35% but a much lower macro F1 of 0.3593 (95% CI [63.79%, 66.91%] / [0.3281, 0.3752]) — reliability drops sharply when going fine-grained. This model is the one applied to "Other" records for review-priority suggestions.

## Limitations

These confidence intervals condition on the fixed fitted models and this specific train/test split, and assume independent parameter records; they exclude retraining and model-selection uncertainty. Most test-set plugins (Actions) also appear in the training set, so these numbers describe performance on this fixed split, not necessarily on entirely unseen plugins — a plugin-grouped or external evaluation is a natural next step. A prediction's probability score for an individual record is not itself a confidence interval on correctness. Coefficients alone do not establish that a model generalizes.
