# Research Question 4: Clustering Plugins and Risk Segmentation

## Question

When plugins are grouped by the type of data they collect, do natural risky/safe clusters emerge?

## Setup

The unit of analysis was switched from parameter to plugin. The `plugin_id_filenames` list was exploded so each (plugin, parameter) pair became its own row: 12,811 parameter records → 40,261 (plugin, parameter) rows → 4,592 unique plugins.

Plugins with at least 3 parameters were kept (3,041 plugins — fewer would make a "profile" meaningless/random chance). For each plugin, the **proportion** (not raw count) of its parameters in each of the 25 `main_data_type` categories was computed, producing a "data collection profile" matrix (3,041 plugins x 25 categories).

## K-Means clustering

After standardizing the categories, silhouette score was tried for K=2..10. The best score was at K=10 (0.254), but scores were low overall (0.17-0.25) — plugins don't form sharply separated clusters.

The 10 clusters are very uneven in size: 2 large clusters (1,523 and 891 plugins) cover most plugins, the remaining 8 clusters are very small (3-19 plugins).

## Cluster profiles and sensitive-category share

| Cluster | # Plugins | Sensitive share | Dominant categories |
|---|---|---|---|
| 1 | 83 | 16.1% | Market data, Time, Finance information |
| 8 | 891 | 12.5% | Identifier, Other, App usage data |
| 5 | 3 | 10.5% | Real estate data, Location |
| 2 | 338 | 9.4% | Message, Files and documents |
| 6 | 5 | 4.2% | Food and nutrition information |
| 4 | 14 | 3.4% | E-commerce data |
| 7 | 11 | 2.3% | Travel information, Time |
| 9 | 19 | 2.2% | Location, Weather information |
| 0 | 1,523 | 1.5% | App usage data, Query |
| 3 | 154 | 0.0% | App metadata, Query |

## Finding

Clustering does not produce a clean "risky vs. safe" binary split. Instead, **functional/thematic groups** emerge (finance & market, travel, e-commerce, location & weather, messaging & files, general-purpose, metadata-only). The share of sensitive categories is distributed gradually across these clusters (from 0% to 16.1%). The two largest clusters (~80% of all plugins) already have a low-to-moderate sensitive share; high risk is concentrated in small, specific-purpose clusters (like finance, real estate).

Practical takeaway: clusters don't give a binary label, but they do provide a **rankable risk score** — plugins in Cluster 1 and Cluster 8 could be flagged as groups warranting priority review.
