# Research Question 2: Description-Writing Rate in Sensitive Categories

## Question

Is the rate at which the `description` field is filled in for parameters in sensitive categories (Security credentials, Personal information, Health information, Finance information) different from non-sensitive categories? Do plugin developers write a description less often when a parameter is sensitive?

Note: this is about the `description` field in the main parameter catalog (12,811 records) — whether the developer bothered to explain the parameter at all. It is a completely different analysis from whether a plugin's privacy policy document discloses the data collection (see the separate privacy-policy audit finding).

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

The difference is statistically significant, but the effect size is practically negligible. In a large sample like 12,811 records, even very small differences can come out "significant." Conclusion: whether a parameter is sensitive or not barely predicts whether it has a description in practice — the lack of a description seems to be a general habit, independent of category. So, no: plugins do not meaningfully write fewer descriptions for sensitive parameters; the small 81.2% vs 84.65% gap is statistically real but practically negligible.
