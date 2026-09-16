# Do Plugins Naturally Cluster Into Risky vs. Safe Groups?

## Question

When plugins (GPT Actions) are grouped by the type of data they collect, do natural "risky" and "safe" clusters emerge?

## Method

Each plugin's data-collection profile is the proportion of its parameters falling into each of the 25 `main_data_type` categories, computed on the 30,478 distinct parameter–plugin pairs (after deduplicating repeated plugin IDs within a single record's plugin list — an earlier version of this analysis skipped that step and inflated the pair count to 40,261, skewing the profiles). Only plugins with at least 3 distinct parameters are kept (2,968 of 4,592), since a plugin with 1-2 parameters would have a meaningless, chance-driven profile. Profiles are standardized (mean 0, standard deviation 1 per category) before K-Means clustering with K chosen by silhouette score over K=2 through 10.

## Finding

Silhouette scores are low-to-moderate and non-monotonic (ranging from 0.140 at K=7 to 0.268 at K=10) — plugins do not form clusters with clean, sharply separated data-collection profiles, which is expected since most plugins mix several categories. The best score sits at K=10, the edge of the searched range, so it's the best candidate found in 2-10 rather than a proven global optimum.

Clustering surfaces **functional/thematic groups** rather than a clean sensitive/non-sensitive split:
- A personal-information & messaging cluster (162 plugins, 35.1% sensitive share) and a security-credentials & app-usage cluster (198 plugins, 31.1% sensitive share) stand out as the most sensitive-heavy.
- Several much larger clusters have low sensitive shares: a general app-usage cluster (1,541 plugins, 0.9% sensitive share) and an "Other"/Identifier-heavy cluster (531 plugins, 3.2% sensitive share) are the two biggest.
- Smaller thematic clusters include location/weather (19 plugins), vehicle-focused (6 plugins), app-metadata-only (153 plugins), e-commerce (13 plugins), time/location (342 plugins), and real estate (3 plugins).

**"Largest" and "riskiest" are different clusters**: the two largest clusters together make up about 69.8% of eligible plugins but have low sensitive shares (0.9% and 3.2%). The two clusters with the highest sensitive share make up only about 12.1% of eligible plugins. A low sensitive share doesn't mean a cluster is safe overall — it's a share of only the four selected categories; other risk-carrying categories like Location, Identifier, or Message aren't counted in it.

## Interpretation

The natural clusters don't split cleanly into "risky vs. safe," but ranking clusters by sensitive-data density is itself a useful risk-segmentation tool — plugins in the highest-sensitive-share clusters can be flagged as the group that most deserves review priority. Cluster IDs are specific to this corrected, deduplicated run and shouldn't be compared by number to any earlier analysis.
