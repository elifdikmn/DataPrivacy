# Research Question 2: Description-Writing Rate in Sensitive Categories

## Question

Is the rate at which the `description` field is filled in for parameters in sensitive categories (Security credentials, Personal information, Health information, Finance information) different from non-sensitive categories?

## Note on missing data

Missing values in the `description` field are stored as an **empty string (`""`)**, not `NaN`. Overall, 1,999 of the 12,811 records (15.6%) have an empty description.

## Finding

- Description-writing rate in sensitive categories: **81.2%**
- Description-writing rate in non-sensitive categories: **84.65%**

So, contrary to expectation, the description-writing rate in sensitive categories is slightly **lower**.

## Statistical test

A chi-square test of independence was applied:
- Chi-square statistic: 7.514
- p-value: ≈0.006 (below the 0.05 threshold, statistically significant)
- Cramér's V (effect size): ≈0.024 (very small, negligible)

## Interpretation

The difference is statistically significant, but the effect size is practically negligible. In a large sample like 12,811 records, even very small differences can come out "significant." Conclusion: whether a parameter is sensitive or not barely predicts whether it has a description in practice — the lack of a description seems to be a general habit, independent of category.
