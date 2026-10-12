# Can the Model Suggest Categories for "Other" Records?

## Question

The `data_type` column's largest value is "Other" (3,544 records) — an ambiguous label. Can the classification model from Research Question 3 suggest possible categories for human review?

## Method

The fine-category model can't just be reused as-is, because it was trained with "Other" as a valid class — it would likely keep predicting "Other" and teach us nothing. Instead, a model is retrained using only the 9,267 non-"Other" records, over the real 144 `data_type` values (rare classes grouped, 78 classes total). This forces the model to suggest the most likely real category for every "Other" record rather than being able to say "I don't know" and default back to Other. On its own held-out test set, this retrained model reaches 69.0% accuracy and 0.429 macro F1 (95% CI [67.4%, 70.7%] / [0.393, 0.446]) — close to the original fine-category baseline, as expected since it's the same method.

## Finding

Applied to all 3,544 "Other" records, the model's highest prediction score is generally low: the median is only 20.9%. This may reflect ambiguous or general-purpose records, but the scores are not calibrated probabilities. Only 411 records (11.6%) cross a 0.5 prediction-score threshold. These are review candidates, not 411 confirmed labeling errors.

Of those 411 candidates, 7 were assigned to one of the project's selected sensitive categories (4 to Security credentials, 3 to Personal information). Names such as `email`, `key`, `KEY` and `token` warrant closer inspection, but a name alone does not establish the correct category. All 411 above-threshold predictions' fine-to-broad category mappings resolved (0 left unresolved); this checks the category mapping, not the predictions' correctness.

## Interpretation

The model cannot establish which "Other" records are mislabeled. These are **unverified review candidates, not confirmed hidden sensitive records** — for example, a "skills" field mapped to API key, or an "email_type" field mapped to Email address, could be false positives. A high prediction score does not prove correctness, and a low score does not prove the original "Other" label is correct; the 0.5 threshold is an uncalibrated cutoff, not a confidence interval. Use the suggestions as a human-review priority list, not automatic relabeling. The known-label test-set confidence intervals describe a different, labeled population and do not transfer to the unlabeled "Other" records; measuring precision on them requires human-reviewed ground truth.
