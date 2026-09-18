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

----------------------------------------------------------------------------

## Champion Model Selection (Phase 4.6)

Three models trained and compared on the validation set:

| Model | F1-score | AUC-ROC | Recall (Churn) | False Negatives |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.613 | 0.845 | 0.60 | 112 |
| Random Forest (default) | 0.566 | 0.838 | 0.52 | 135 |
| Random Forest (class_weight='balanced') | **0.628** | 0.836 | **0.72** | **77** |

**Champion: Random Forest (class_weight='balanced')**

Selected over the baseline despite a marginally lower AUC-ROC (0.836 vs
0.845 — not meaningful) because:
- Higher F1-score (0.628 vs 0.613)
- Substantially better recall on the churn class (0.72 vs 0.60) —
  catches 35 more actual churners
- In a churn-prediction context, a missed churner (false negative) is
  typically costlier to the business than an unnecessary retention
  offer (false positive), since customer acquisition cost generally
  exceeds retention-offer cost.

Trade-off: 64 additional false positives (164 vs 100) — more customers
flagged who would not have churned. Accepted given the above reasoning.

MLflow Run ID: *(paste the random_forest_balanced run ID from the UI here)*

----------------------------------------------------------------------

## Phase 6: Model Deployment

### Deployment Strategy
Online (REST API) — chosen per Phase 1 feasibility: not latency-critical,
but interactive scoring is valuable for demos and future integration.

### Architecture
- `src/app.py` — Flask inference API (`/health`, `/predict`)
- Preprocessing (label encoding) uses the exact encoders saved during
  training (`models/encoders.pkl`) to prevent train/serve skew
- Served via `waitress` (production WSGI server), not Flask's dev server
- Containerized with Docker (`Dockerfile`, `.dockerignore`)

### Endpoints
- `GET /health` — liveness check
- `POST /predict` — accepts a JSON customer record, returns
  `churn_prediction` (Yes/No) and `churn_probability`

### Not implemented (would apply at cloud/enterprise scale)
- **Canary / blue-green deployment (6.5):** at cloud scale, a new model
  version would be routed a small percentage of traffic first, or run
  alongside the current version with instant rollback capability. For
  a single-container laptop deployment, this isn't applicable —
  rollback here means keeping the previous `champion_model.pkl` and
  redeploying it (see Phase 8 for versioning plan).
- **Load balancer / autoscaling (6.6):** relevant when running multiple
  container replicas behind traffic (e.g., on Kubernetes or a managed
  platform like SageMaker/Vertex AI). A single local container has no
  need for this, but the containerized design means it *could* be
  deployed behind a load balancer without any code changes if traffic
  demanded it later.

  ----------------------------------------------------------------------

  ## Phase 7: Monitoring and Observability

### Prediction Logging
Every `/predict` call logs a record (timestamp, input features, prediction,
probability) to `logs/predictions.jsonl` (JSON Lines format, append-only).

### Drift Detection
`src/check_drift.py` compares the mean of key numeric features (`tenure`,
`MonthlyCharges`, `TotalCharges`) between the training set and logged live
predictions, flagging any feature whose mean has shifted more than 20%
(per Step 7.4's example threshold).

**Known limitation:** this check is only statistically meaningful once a
reasonable volume of live predictions has accumulated (dozens+, ideally
hundreds). With only a handful of logged predictions, the reported "drift"
mostly reflects sampling noise from which test cases happened to be sent,
not a genuine shift in customer population. In production, this check
would run on a rolling window (e.g., "last 1,000 predictions" or
"last 7 days") rather than the full log from day one.

### Dashboards / Alerting (Step 7.5) — not implemented
A local script satisfies the monitoring *logic*; a real dashboard
(Grafana/Prometheus) or scheduled alerting job is out of scope for a
laptop deployment but would consume the same `predictions.jsonl` data
or `check_drift.py` output as its source.
---------------------------------------------------------------------------

## Phase 8: Continuous Training and Retraining

### Retraining Triggers (Step 8.1) — documented, not scheduled
In production, retraining would fire on:
- **Scheduled** (e.g., weekly via cron / Windows Task Scheduler / Airflow)
- **Drift threshold crossed** (`src/check_drift.py` reporting DRIFT status)
- **New data volume threshold reached**

For this laptop project, retraining is triggered manually by running
`python src/retrain_gate.py`. The scheduling infrastructure itself
(cron, Task Scheduler, or an orchestrator like Airflow/Prefect) is out
of scope, but the retraining *logic* it would call is fully implemented
and tested.

### Validation Gate (Step 8.3) — implemented
`src/retrain_gate.py` wraps the training pipeline:
1. Evaluates the current champion model on validation data
2. Backs up the current model/encoders to `models/archive/`
3. Trains a new candidate model
4. Compares F1 scores — rejects and auto-rolls-back if the new model's
   F1 drops by more than 2% (`MAX_ACCEPTABLE_F1_DROP = 0.02`)

Tested both paths explicitly: confirmed a normal retrain is accepted,
and confirmed (via a temporarily lowered threshold) that a rejected
retrain correctly restores the previous champion from backup.

### Version Tracking / Rollback (Step 8.4, 8.5) — implemented
Every retrain backs up the current model to
`models/archive/champion_model_<timestamp>.pkl` before overwriting,
enabling manual rollback to any previous version if needed.