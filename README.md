# Telco Customer Churn — MLOps Project

## Phase 1: Problem Definition and Feasibility

### Business Goal
Reduce customer churn by identifying at-risk customers early enough for
retention teams to intervene (e.g., targeted offers, proactive outreach)
before they cancel their service.

### ML Task
Binary classification — predict `Churn` (Yes / No) for each customer based
on account, service, and billing attributes.

### Feasibility
- **Data availability:** Telco Customer Churn dataset (Kaggle), ~7,000 rows
  with demographics, account info, and service usage — sufficient for a
  baseline classifier.
- **Latency requirements:** Not real-time-critical. Batch scoring (e.g.,
  weekly) is a reasonable scope for the MVP.
- **Interpretability:** Important — retention teams need to know *why* a
  customer is flagged. Favors logistic regression or tree-based models
  (with SHAP) over black-box deep learning.

### KPIs
- **Model:**
  - F1-score (primary — churn is typically imbalanced, so accuracy alone
    is misleading)
  - AUC-ROC (secondary)
- **System:**
  - Inference time per batch
  - Pipeline run success rate
  - Data/concept drift thresholds (defined in Phase 7)

### Baseline
A simple heuristic or logistic regression model, trained on raw features
with minimal engineering. This is the bar every later "champion model"
(Phase 4) must beat.

---

## Project Structure

telco-churn-mlops/
├── configs/          # config files (paths, hyperparameters)
├── data/
│   ├── raw/          # original, untouched data (gitignored)
│   └── processed/    # cleaned/feature-engineered data (gitignored)
├── models/           # trained model artifacts (gitignored)
├── notebooks/        # exploration only — no reusable logic here
├── src/              # reusable Python modules (data loading, training, etc.)
├── tests/            # unit tests
├── requirements.txt
└── README.md

## Status
- [x] Phase 1: Problem Definition and Feasibility
- [ ] Phase 2: Data Engineering and Versioning
- [ ] Phase 4: Model Development and Experiment Tracking
- [ ] Phase 5: Pipeline Automation
- [ ] Phase 6: Model Deployment
- [ ] Phase 7: Monitoring and Observability