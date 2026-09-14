# Research Question 1: Category Distribution and Share of Sensitive Categories

The dataset consists of 12,811 parameter records, collected by 4,592 unique GPT plugins. Each record is labeled with `main_data_type` (25 coarse categories) and `data_type` (145 fine-grained categories).

## Overall distribution

The `main_data_type` distribution is skewed: the top 3 categories (App usage data: 2,568 records, Identifier: 1,888 records, Other: 1,721 records) alone make up 48.2% of the data. The remaining 22 categories form a long tail, shrinking steadily.

Full ranking (record count, percentage):
- App usage data: 2,568 (20.05%)
- Identifier: 1,888 (14.74%)
- Other: 1,721 (13.43%)
- Query: 1,416 (11.05%)
- Time: 1,131 (8.83%)
- Web and network data: 925 (7.22%)
- Location: 701 (5.47%)
- Personal information: 435 (3.40%)
- Files and documents: 412 (3.22%)
- Market data: 367 (2.86%)
- Security credentials: 276 (2.15%)
- Message: 192 (1.50%)
- App metadata: 138 (1.08%)
- Finance information: 138 (1.08%)
- Health information: 82 (0.64%)
- E-commerce data: 66 (0.52%)
- Travel information: 58 (0.45%)
- Sports information: 57 (0.44%)
- Event information: 51 (0.40%)
- Vehicle information: 42 (0.33%)
- Real estate data: 35 (0.27%)
- Food and nutrition information: 33 (0.26%)
- Gaming data: 27 (0.21%)
- Weather information: 26 (0.20%)
- Legal and law enforcement data: 26 (0.20%)

## Sensitive categories

The 4 categories defined as "sensitive" in this project: Security credentials, Personal information, Health information, Finance information. Together these four make up **931 records (7.3% of all data)** — a small but significant slice of the dataset.

## About data_type (fine-grained category)

The `data_type` column has 145 distinct values, with a much longer tail: the median class size is only 11 records, 6 classes have just a single record, and 40 classes have fewer than 5 records. The largest `data_type` value is "Other" — 3,544 records (note: this is not the same as `main_data_type`'s "Other", which has 1,721 records).
