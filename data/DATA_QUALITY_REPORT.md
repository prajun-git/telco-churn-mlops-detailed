# Data Quality Report — Telco Customer Churn Dataset

## Source
Telco Customer Churn dataset (Kaggle: blastchar/telco-customer-churn)

## Shape
7,043 rows, 21 columns

## Target Variable
`Churn` — binary (Yes/No)
- No: 73.46%
- Yes: 26.54%

**Implication:** Moderate class imbalance. F1-score selected as primary
model KPI (per Phase 1) rather than accuracy, since a naive "always No"
model would score ~73% accuracy while being useless.

## Data Quality Issues Found

### 1. `TotalCharges` — incorrect dtype + hidden missing values
- Loaded as `object` (text) instead of numeric.
- 11 rows contained blank/whitespace strings instead of numbers.
- `df.isnull().sum()` did NOT catch these — they were empty strings,
  not `NaN`.

**Root cause:** All 11 affected rows have `tenure = 0` — brand-new
customers who have not yet been billed a full cycle. This is expected
behavior, not corrupted data.

**Resolution:** Converted `TotalCharges` to numeric
(`pd.to_numeric(..., errors='coerce')`), then imputed the 11 missing
values as `TotalCharges = MonthlyCharges` (a day-one customer's total
charges approximate one month's rate).

### 2. No other missing values
All other 20 columns had 0 missing values after the above fix.

### 3. No duplicate rows
Confirmed 0 fully duplicate rows and 0 duplicate `customerID` values.

## Train/Validation/Test Split
- Method: stratified split on `Churn`, random_state=42
- Train: 4,930 rows (70%) — 26.53% churn
- Validation: 1,056 rows (15%) — 26.52% churn
- Test: 1,057 rows (15%) — 26.58% churn

Stratification confirmed: churn ratio consistent across all splits
(~26.5%), preventing skewed evaluation from an unlucky random split.

## Versioning
All raw and processed datasets (including train/val/test splits) are
tracked with DVC. Git tracks the `.dvc` pointer files; DVC tracks the
actual data content, keeping the Git repo lightweight.