# Research Question 5: Reclassifying "Other" Records

## Question

Can the trained model be used to classify records with an uncertain label (especially the `data_type` "Other" category, 3,544 records)?

## Methodology

The Section 2 `data_type` model can't be used as-is because it had seen "Other" as a valid class during training — that would just lead the model to say "Other" again. Instead, the model was retrained using **only the non-"Other"** 9,267 records, over the real 144 `data_type` values (with rare classes grouped, 78 classes). This way, when the model looks at an "Other" record, it's forced to propose a real category.

The model produces a "confidence score" (the highest predicted probability) for every prediction.

## Results

- The retrained model's performance on its own test set: accuracy 69%, macro F1 42.9% (close to Section 2's data_type model).
- When applied to the 3,544 "Other" records, the confidence-score distribution is low: median 20.9%, mean 28.9%.
- **Number of records clearing a 50% confidence threshold: 411 (11.6%)**. The large remaining majority leaves the model uncertain too — these records are probably genuinely ambiguous/general-purpose.
- Most confident predictions are intuitive, general-purpose categories: Current session setting, Resource IDs, Search query, Query filter.
- **7 records** confidently point to a **sensitive category** (Security credentials or Personal information at the main_data_type level). Examples: `name="email"` (91% confidence → Email address), `name="key"`/`name="KEY"` (82% confidence → API key), `name="token"` (51% confidence → Access tokens). These records were labeled "Other" in the original dataset, but their names alone make it obvious they're sensitive data.

## Conclusion

The model can't reliably reclassify all "Other"-labeled records — most genuinely stay ambiguous. But in a small, high-confidence subset, it can catch genuinely mislabeled sensitive data types. The model should be used not for automatic relabeling, but to **generate a review-priority list for human review**.
