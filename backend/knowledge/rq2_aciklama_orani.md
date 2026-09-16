# Does Description-Writing Rate Differ for Sensitive Parameters?

## Question

Intuitively, one might expect that if a parameter requests sensitive data (e.g. a password or health information), the developer would take more care to describe it — clearly explaining to the user why it's needed. This analysis tests that: is the rate at which parameters in sensitive categories have a description genuinely different from non-sensitive ones, or is the difference we see just random fluctuation? (Note: this is about the schema `description` field being present, not about whether a privacy policy discloses the collection — see the separate policy-disclosure finding for that.)

## Finding

The result is the **opposite** of intuition: the description-writing rate for parameters in sensitive categories (81.20%, 95% Wilson CI [78.57%, 83.58%]) is slightly **lower** than for non-sensitive ones (84.65%, CI [83.99%, 85.28%]) — i.e., developers describe parameters that request sensitive data slightly *less*, not more. The paired bootstrap estimate of the difference is -3.44 points, 95% CI [-6.04, -0.97] — small but the intervals barely overlap.

A chi-square test of independence says this difference isn't random (chi-square statistic 7.514, p = 0.0061, below the 5% significance threshold) — there's a statistically significant association. But Cramér's V is only 0.0242 — meaning the relationship between the two variables is **practically very weak**. In short: thanks to the large sample of 12,811 records, even a tiny difference can come out "statistically significant," but the real-world effect of that difference is negligible.

## Interpretation

Practical takeaway: **whether a parameter is sensitive or not barely predicts whether it has a description** — the lack of a description looks like a general habit across the catalog, largely independent of category. This is a caution against assuming that developer diligence naturally scales with data sensitivity.

## Assumptions and limitations

The chi-square test and these confidence intervals assume each parameter record is an independent observation. In practice, plugins that reuse the same description template across many parameters could violate that assumption — the association is real in this catalog, but "how independently informative" each of the 12,811 rows really is stays an open question, and this is not a causal claim (sensitivity is not shown to *cause* fewer descriptions). A confidence interval measures estimation uncertainty on this fixed sample; it is not a probability that any individual record is correctly labeled.
