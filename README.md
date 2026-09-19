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

### CI/CD (Step 5.3) — implemented
`.github/workflows/ci.yml` runs on every push to `master`:
1. Pulls versioned data from the DVC remote (AWS S3)
2. Runs the full pipeline (`run_pipeline.py`) end-to-end on a fresh
   environment — data cleaning, splitting, training, and evaluation
3. Verifies the resulting model artifacts exist

This validates that the entire pipeline reproduces correctly outside
the local development environment, not just that individual scripts
compile.

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

----------------------------------------------------------------------
## Phase 9: Governance and Compliance

### Model Explainability (Step 9.2) — implemented
`src/explain.py` generates a global SHAP summary plot
(`models/shap_summary.png`) showing which features drive churn
predictions across the model as a whole. Top global drivers: `Contract`,
`tenure`, `OnlineSecurity`, `MonthlyCharges`, `TechSupport` — consistent
with domain intuition (short tenure, flexible contracts, and missing
add-on services correlate with higher churn risk).

The `/predict` API endpoint returns a `top_factors` field with each
response — the top 3 SHAP-driven features for that specific prediction,
with direction (increases/decreases risk). This directly satisfies the
interpretability requirement from Phase 1: retention teams can see not
just *that* a customer is flagged, but *why*, and act on it.

### Model Lineage (Step 9.1) — documented
End-to-end lineage for the current champion model:
`data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` (DVC-tracked)
→ `data_prep.py` → `data/processed/telco_churn_cleaned.csv` (DVC-tracked)
→ `split_data.py` → train/val/test splits (DVC-tracked)
→ `train.py` → `models/champion_model.pkl` + `encoders.pkl`,
  logged to MLflow (experiment: `telco-churn-prediction`)
→ `src/app.py` — deployed via Docker container

### Approval Workflow / Audit Logs (Step 9.3, 9.4) — not implemented
A formal dev→staging→production approval workflow and dedicated audit
trail are enterprise-scale concerns beyond a solo laptop project.
`src/retrain_gate.py` (Phase 8) provides a lightweight automated
approval gate, and `logs/predictions.jsonl` (Phase 7) serves as a basic
prediction audit trail.

### Data Privacy (Step 9.5) — not applicable
The Telco dataset is public and anonymized (no PII beyond a masked
`customerID`); encryption and RBAC are not applicable at this scale.
-----------------------------------------------------------------------------------
## Phase 10: Maintenance and Iteration

### A/B Testing Framework (Step 10.2) — implemented
`src/ab_test.py` compares the current champion model against any
archived candidate (from `models/archive/`, populated by the retraining
gate) on F1 and AUC-ROC. Validated against a known-identical model:
correctly reported a tie, confirming the comparison logic itself works
correctly before it's needed to catch a real difference.

### Runbook: Common Operations

**Retrain the model:**

python src/retrain_gate.py

Automatically backs up the current champion, trains a new candidate,
and only replaces the champion if F1 doesn't drop by more than 2%.

**Check for data drift:*

python src/check_drift.py

Compares live prediction inputs (from `logs/predictions.jsonl`) against
training data distribution. Note: needs a meaningful sample size
(dozens+) to be statistically trustworthy — see Phase 7 notes.

**Roll back to a previous model version:**
Model backups live in `models/archive/champion_model_<timestamp>.pkl`
and `encoders_<timestamp>.pkl`. To roll back manually, copy the desired
backup over `models/champion_model.pkl` and `models/encoders.pkl`.

**Run the full pipeline from scratch:**

python src/run_pipeline.py

Runs data prep → split → train → evaluate end-to-end.

**Start the API locally:**

python src/app.py

**Start the API in Docker:**

docker build -t telco-churn-api .
docker run -p 5000:5000 telco-churn-api


### Architecture Overview

Raw CSV (DVC-tracked)
|
v
data_prep.py --> cleaned CSV (DVC-tracked)
|
v
split_data.py --> train/val/test (DVC-tracked)
|
v
train.py --> champion_model.pkl + encoders.pkl (MLflow-logged)
|
v
app.py --> Flask API (Dockerized) --> /predict (+ SHAP explanation)
|
v
logs/predictions.jsonl --> check_drift.py (monitoring)

retrain_gate.py wraps train.py with a validation gate + rollback,
using archived models for later A/B comparison (ab_test.py).


### Review Schedule (Step 10.1) — documented policy
For a production version of this project, recommended cadence:
- **Weekly:** review `check_drift.py` output against accumulated
  prediction logs
- **Monthly:** re-run `retrain_gate.py` on fresh data, review MLflow
  run comparison
- **Quarterly:** revisit SHAP feature importance — confirm business
  logic still holds as customer behavior evolves

### Cost Optimization (Step 10.4) — not applicable
Cloud cost optimization (spot instances, autoscaling) doesn't apply to
a local laptop deployment. Noted for future cloud migration.

### User Feedback (Step 10.3) — not implemented
Would require a human-in-the-loop interface for retention teams to
confirm/dispute predictions — out of scope for this project's MVP.

-----------------------------------------------------------------------

## API

Two implementations exist:

### FastAPI (primary — used in Docker deployment)
`src/app_fastapi.py` — served via `uvicorn`, containerized in the
Dockerfile. Provides automatic request validation (Pydantic) and
interactive API documentation at `/docs`.

Run locally:

uvicorn src.app_fastapi:app --host 0.0.0.0 --port 8000

Then visit `http://127.0.0.1:8000/docs` for interactive testing.

### Flask (original implementation)
`src/app.py` — served via `waitress`. Kept for comparison; functionally
identical predictions and SHAP explanations.

Run locally:

python src/app.py





