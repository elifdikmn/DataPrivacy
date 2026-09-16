# Are the Sensitive Parameters Also the Undisclosed Ones?

## Question

Among the parameters audited for policy disclosure (RQ6), how many are *both* a sensitive category (RQ1's four-category definition) *and* completely undisclosed?

## Data source and method

This combines two analyses that were previously kept separate: RQ1's definition of "sensitive" (`main_data_type` in Security credentials / Personal information / Health information / Finance information, over the main 12,811-record dataset) and RQ6's policy-disclosure audit (`backend/final_results/`, 308 parameters with comparable policy text, using its own finer 30-label `data_type` taxonomy). The two taxonomies aren't the same, so a hand mapping between them is needed for this appendix specifically.

Only `data_type` labels that unambiguously belong to one RQ1 category are mapped: `Passwords` → Security credentials; `Email address`, `Name`, `Phone number` → Personal information; `Other financial info`, `Purchase history` → Finance information. This mapping is a new interpretive choice made for this analysis, not previously defined anywhere else in the project. It's justified using the source paper's own definitions: Table 4 lists "Purchase history" and "Income information" as Finance-information subtypes, and Figure 10's caption explicitly places "exact address" under Location rather than Personal information — which is why the ambiguous `Address` label (8 occurrences in this audit) is deliberately left unmapped rather than guessed. **No label in this audit's 30-label taxonomy corresponds to Health information at all**, so this analysis cannot say anything about health-data disclosure specifically.

## Finding

Of the 308 audited parameters, only **20** map to one of the three sensitive categories present in this audit's taxonomy: 16 Personal information, 3 Finance information, 1 Security credentials, and 0 Health information (no health-related label exists in this taxonomy). Of those 20, **18 (90.0%) are UNDISCLOSED** — 95% Wilson CI [69.9%, 97.2%], wide because the sample is small (n=20). As a share of all 308 audited parameters, that's 18/308 = **5.8%** (95% Wilson CI [3.7%, 9.0%]).

## Interpretation

The pattern from RQ6 (most collection generally goes undisclosed) also holds, if anything more strongly, on the subset that maps to a sensitive category specifically — 90.0% vs. the overall audit's 90.3% undisclosed rate are close, so sensitive parameters are not disclosed any more carefully than non-sensitive ones in this small sample.

**A coincidence worth flagging, not overclaiming:** 5.8% is numerically identical to the source paper's own separately-computed headline figure ("the data collection of only 5.8% of Actions is consistent with their disclosures," aggregated at the Action level). These are two different statistics — a different aggregation unit (parameter here vs. Action there), a different numerator/denominator definition, and a project-specific sensitive-category mapping the paper itself never defines. Treat the matching digits as a coincidence, not the same finding restated.

## Limitations

n=20 is small — relabeling even a single item would move the estimate several points, hence the wide confidence interval. The `data_type` → `main_data_type` mapping above is this project's own judgment call, made explicit and justified rather than hidden, precisely so it can be checked or challenged. Because no Health-information label exists in this particular audit's taxonomy, this analysis is silent on health-data disclosure specifically — a differently-sourced audit would be needed to check that. Like RQ6, this audit only covers 184 of 4,592 plugins, so treat this as a directional finding on a small, non-random sample, not a precise ecosystem-wide percentage.
