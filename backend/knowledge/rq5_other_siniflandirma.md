# Can Records Mislabeled "Other" Be Identified Automatically?

## Question

The `data_type` column's largest value is "Other" (3,544 records) — an admittedly ambiguous label. Can the classification model from Research Question 3 be used to suggest what these records should actually be?

## Method

The fine-category model can't just be reused as-is, because it was trained with "Other" as a valid class — it would likely keep predicting "Other" and teach us nothing. Instead, a model is retrained using only the 9,267 non-"Other" records, over the real 144 `data_type` values (rare classes grouped, 78 classes total). This forces the model to suggest the most likely real category for every "Other" record rather than being able to say "I don't know" and default back to Other. On its own held-out test set, this retrained model reaches 69.0% accuracy and 0.429 macro F1 (95% CI [67.4%, 70.7%] / [0.393, 0.446]) — close to the original fine-category baseline, as expected since it's the same method.

## Finding

Applied to all 3,544 "Other" records, prediction confidence (the model's highest output probability) is generally low: the median is only 20.9%. This isn't surprising — the "Other" label likely already contains genuinely ambiguous or general-purpose records, and the model reflects that as uncertainty. Only 411 records (11.6%) cross a 0.5 confidence threshold. Most confident predictions are intuitive, general-purpose technical categories.

But 7 records confidently point to a sensitive category (4 to Security credentials, 3 to Personal information). Records with names like `email`, `key`, `KEY`, `token` are suggested as sensitive categories with 50-91% confidence — in the original dataset these are labeled "Other," but their names alone make it obvious they're sensitive data. All 411 above-threshold predictions' fine-to-broad category mappings resolved cleanly this time (0 left unresolved) — the mapping logic deliberately leaves a `data_type` unresolved rather than guessing when it maps to more than one broad category.

## Interpretation

The model can't reliably reclassify all "Other" records — but for a small, high-confidence subset, it can catch genuinely mislabeled sensitive data. These are **unverified review candidates, not confirmed hidden sensitive records** — for example, a "skills" field mapped to API key, or an "email_type" field mapped to Email address, could well be false positives. A high prediction score doesn't prove correctness, and a low score doesn't prove the original "Other" label was wrong; the 0.5 threshold is an uncalibrated cutoff, not a confidence interval. The right use of this model is generating a human-review priority list — not automatic relabeling. The known-label test-set confidence intervals above describe a different, labeled population and don't transfer to correctness on these unlabeled "Other" records; measuring real precision on this population needs human-reviewed ground truth.
