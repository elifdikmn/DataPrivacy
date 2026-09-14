# Do Plugins Disclose What They Actually Collect?

## Question

For the data a plugin actually collects, does its privacy policy disclose that collection?

Note: this is about the plugin's **privacy policy document** — separate legal/marketing text outside the parameter catalog. It is a different analysis from whether the parameter catalog itself has a `description` field filled in for that parameter (see the separate description-writing-rate finding, which found sensitive and non-sensitive parameters get a description at almost the same rate, ~81-85%).

## Data source and scope

This uses a separate, smaller dataset (`backend/final_results/`, 381 plugin files) produced and published by the authors of the source paper (Wu et al., 2025, IMC '25 — see the project README's Data Sources & Attribution section), not generated as part of this project's own analysis. In it, each collected data parameter was matched against sentences extracted from that plugin's own privacy policy, and each candidate sentence was labeled with one of five categories from the source paper's own taxonomy: `CLEAR` (clearly discloses it), `VAGUE` (vaguely discloses it), `AMBIGUOUS` (unclear whether it discloses it), `INCORRECT` (disclosure is wrong/misleading), or `OMITTED` (that particular sentence doesn't disclose it). Only 184 of the 381 files had any policy text to compare against. This audit uses its own category labels (e.g. "Email address", "User IDs", "Approximate location"), which are a different, finer, hand-labeled taxonomy from the `main_data_type`/`data_type` categories used in the main 12,811-record dataset — the two are not directly comparable one-to-one.

## Method

Each parameter has many candidate policy sentences checked against it (most are automatically irrelevant, hence labeled OMITTED for that sentence). A parameter's overall disclosure status is determined by its *best* matching sentence, using the source paper's own label priority: CLEAR first, then VAGUE, then AMBIGUOUS, then INCORRECT; only if every checked sentence is OMITTED does the parameter count as genuinely undisclosed. In practice AMBIGUOUS is extremely rare in this dataset — only 1 of the 308 audited parameters resolves to it.

## Finding

Across 308 parameters with comparable policy text:
- **90.3% (278) are UNDISCLOSED** — no sentence anywhere in the plugin's privacy policy discloses this data collection.
- 5.2% (16) are clearly disclosed.
- 2.3% (7) are vaguely disclosed.
- 1.9% (6) are disclosed but incorrectly/misleadingly.
- 0.3% (1) is ambiguously disclosed.

## Interpretation

Even on this smaller, independently-labeled sample, the pattern is stark: the large majority of what plugins actually collect never appears in their privacy policy at all. This is a much larger gap than a "gray area" of vague language — it's mostly plain omission. Combined with the main analysis (sensitive categories make up 7.3% of all collected data), this suggests that the written notice a user gets from a plugin's policy is a poor guide to what the plugin actually does — reinforcing the value of a tool like this one that checks the actual catalog of what's collected rather than relying on the policy text alone.

## Limitation

This audit only covers a subset of plugins (184 of 4,592) and uses its own data-type taxonomy, so it can't be merged numerically with the main dataset's sensitive-category statistics. Treat it as a strong directional finding, not a precise percentage for the whole ecosystem.
