# AAI-540 ML Design Document
## Retail Store Inventory Demand Forecasting Using AWS SageMaker AI
### Final Project — As-Built Update to Deliverable #1

> This document follows the same section structure as `Final-Project_Group4_Deliverable-1_ML-System-Design.docx`
> (Deliverable #1, published October 14, 2026). Deliverable 1 was written before implementation and
> correctly hedged open questions ("must be confirmed," "target design," "where lab environment permits").
> This update replaces those hedges with confirmed facts from the completed, deployed system, and adds a
> Lessons Learned section documenting the real engineering issues encountered and fixed during the build.

---

## Team Info

| Field | Value |
|---|---|
| Project Team Group # | Group 4 |
| Authors | *[Insert all team member names]* |
| Business Name | Retail Store Inventory Forecasting (academic project) |
| Publication Date | October 14, 2026 (Deliverable 1) — updated per this document |
| GitHub Project Link | https://github.com/vedpd/AAI_540_Group4_Retail_Store_Inventory_Forecasting |
| Asana Board Link | *[Insert if used; otherwise N/A]* |
| Team Tracker Link | *[Insert if used; otherwise N/A]* |

---

## Project Scope

### Project Background

Retail organizations must balance product availability against the cost of carrying excess inventory.
Forecasts that underestimate demand contribute to stockouts and lost sales; overestimates lead to excess
inventory, markdowns, and higher holding costs. This project implements a repeatable retail demand
forecasting workflow that predicts near-term demand per (store, product, day) from historical sales,
calendar effects, and pricing/promotion signals, demonstrating the full MLOps lifecycle on AWS SageMaker AI.

This is a supervised time-series forecasting problem. **Both LSTM and Transformer-encoder architectures
were implemented and compared** against a seasonal-naive baseline and three classical ML models (Ridge,
Random Forest, Gradient Boosting). The final model (LSTM) was selected using measured validation
performance (WAPE) rather than architectural complexity alone — the margin between all four real models
was under 1% WAPE, so the selection criterion mattered more than any single model's sophistication.

### Technical Background

The dataset is the **Kaggle "Retail Store Inventory Forecasting" dataset**
(https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset), confirmed
synthetic. Unlike the original assumption of separate `train.csv`/`test.csv`/`stores.csv`/`items.csv`
files, the actual dataset is a **single CSV** (`data/raw/retail_store_inventory.csv`) with one row per
(store, product, day):

- **73,100 records** = 5 stores × 20 products × 731 days (2022-01-01 to 2024-01-01)
- **No missing values, no duplicate (Date, Store ID, Product ID) keys**
- Columns: `Date, Store ID, Product ID, Category, Region, Inventory Level, Units Sold, Units Ordered,
  Demand Forecast, Price, Discount, Weather Condition, Holiday/Promotion, Competitor Pricing, Seasonality`

The data was explored with summary statistics and time-series visualization in
`notebooks/exploratory_data_analysis.ipynb`, cleaned/transformed in `src/data_preprocessing/`, and split
**chronologically** (not shuffled) into training, validation, and test periods to avoid temporal leakage.
Final features include calendar variables, per-(store, product) historical demand lags and rolling
statistics, and the `Holiday/Promotion` indicator. Models were evaluated with MAE, RMSE, and WAPE, with
sMAPE reported as a supplementary metric. The full workflow is reproducible through S3, SageMaker
Processing, a SageMaker Pipeline (training/evaluation/quality-gate/registration), batch inference, and
CloudWatch monitoring — all implemented, not merely targeted.

**A key EDA finding changes the framing of "Candidate features... holiday indicators if present" from
Deliverable 1:** this synthetic dataset has **no real weekly seasonality** (daily totals vary by
day-of-week within noise) and **no measurable promotion effect** (mean `Units Sold` with vs. without
`Holiday/Promotion` differs by <0.1%). This is not a data-quality gap — it was confirmed and documented
in the EDA notebook — and it directly explains why the LSTM/Transformer could not meaningfully outperform
the classical models: there is little additional temporal structure for a more expressive model to exploit
beyond what lag/rolling features already give Ridge/RF/GB.

### Goals vs Non-Goals

**Goals — status**

| Goal (Deliverable 1) | Status |
|---|---|
| Forecast retail demand, compare LSTM and Transformer | **Done.** Both trained; LSTM selected by validation WAPE |
| Reproducible workflow: data prep → feature engineering → training → evaluation → inference | **Done.** `src/pipelines/training_pipeline.py` end to end, plus the SageMaker Pipeline equivalent |
| Use S3, demonstrate Feature Store / Model Registry / Pipelines | **Done.** All three implemented and exercised (see Solution Overview) |
| Measure predictive performance, include monitoring and automated quality checks | **Done.** WAPE/MAE/RMSE on held-out test, model-quality + data-quality CloudWatch reports, pipeline quality gate |
| Documented, testable repository, demonstrable system | **Done.** pytest suite (unit + AWS integration tests), CI on every push/PR, operations runbook |

**Non-Goals** — unchanged from Deliverable 1; none were violated during implementation:
- No production inventory-optimization/replenishment engine
- No claims of real-world stockout/holding-cost reduction (no live controlled evaluation was run)
- No high-availability multi-region real-time API platform (batch transform is the primary, demonstrated pattern)
- No personal/consumer/payment data used (confirmed: dataset contains only store/product/date/price-level fields)
- No redundant MLOps platforms layered on top of SageMaker-native services

---

## Solution Overview

The implemented workflow stores source data and artifacts in a private S3 bucket
(`sagemaker-us-east-1-<account-id>/aai540/retail-forecasting/`), uses shared preprocessing/feature-engineering
code for training and inference, and trains/evaluates candidate models through both a local pipeline
(`src/pipelines/training_pipeline.py`, MLflow-tracked) and a managed **SageMaker Pipeline**
(`src/aws/sagemaker/pipeline.py`). The following, described as "intended"/"target design" in Deliverable 1,
are all **implemented and verified**:

- **SageMaker Feature Store** (`retail-inventory-features-raw` feature group, Active offline store) — manages
  reusable feature records with sanitized feature names (AWS Feature Store rejects names containing spaces
  or slashes, e.g. `Holiday/Promotion` → `Holiday_Promotion`, `Store ID` → `Store_ID`)
- **SageMaker Model Registry** (`retail-inventory-forecasting-models` model package group) — versions
  evaluated model packages with `PendingManualApproval` → `Approved`/`Rejected` status; exercised in practice
  (an early version was rejected due to a scikit-learn version incompatibility; a corrected retrain was
  approved as version 3)
- **SageMaker Pipelines** (`retail-forecasting-pipeline`) — orchestrates
  `Preprocess → TrainModel → EvaluateModel → CheckQualityGate → RegisterModel | QualityGateFailed`, with
  both a passing (`gate-pass`) and an intentionally failing (`gate-fail`, stricter WAPE threshold)
  demonstration run
- **Batch inference** (`src/aws/sagemaker/batch_inference.py`) — the primary, implemented inference pattern;
  confirmed end to end with real predictions written to S3 (7,500 rows, plausible demand values)
- **Monitoring** — pipeline/job status (CloudWatch + Pipeline console), infrastructure metrics (per-job
  CPU/memory via a CloudWatch dashboard built from `SEARCH()` expressions so new jobs appear automatically),
  input data quality (SageMaker Model Monitor baseline + drift analysis with a PSI table), and forecast
  accuracy (`monitoring_reports.py` joins batch predictions against actuals and publishes MAE/RMSE/WAPE to
  a custom CloudWatch namespace)

Before releasing a new model, the pipeline validates input schema, re-derives features identically to
training, and enforces a quality gate (test WAPE ≤ 0.70 **and** ≥10% MAE improvement over the seasonal-naive
baseline) before allowing registration — exactly as targeted in Deliverable 1, now implemented in
`src/aws/sagemaker/pipeline_scripts/evaluate.py` and mirrored in the local training pipeline.

---

## Data Sources

Confirmed against the downloaded dataset (superseding the "must be confirmed" language in Deliverable 1):

- Single file, `data/raw/retail_store_inventory.csv`, ~6 MB, committed to the repository (synthetic, no
  privacy concern)
- 73,100 rows, 5 stores (`S001`–`S005`), 20 products (`P0001`–`P0020`), 731 days (2022-01-01 to 2024-01-01)
- No missing values, no duplicate (Date, Store ID, Product ID) keys
- **Leakage risk identified and mitigated:** `Demand Forecast` correlates 0.997 with `Units Sold` by
  construction (it is the dataset's own precomputed forecast) and is excluded from features. `Inventory
  Level` and `Units Ordered` are determined partly *after* the fact by `Units Sold` and are also excluded.

The dataset schema was reviewed; only fields needed for forecasting were retained (`Category`, `Region`,
`Price`, `Discount`, `Weather Condition`, `Holiday/Promotion`, `Competitor Pricing`, `Seasonality`, plus
derived calendar/lag/rolling features).

## Data Engineering

Raw and processed data live in the project's S3 bucket under `aai540/retail-forecasting/` with separate
prefixes for raw data, processed Parquet splits, Feature Store offline storage, batch-transform
predictions, and monitoring outputs. S3 Block Public Access remained enabled throughout; the project uses
the AWS Academy Learner Lab's `LabRole`, which has `VocLabPolicy1/2/3` plus standard managed policies
(confirmed via `aws iam list-attached-role-policies`) — no access keys were embedded in code or notebooks.

Preprocessing (`src/data_preprocessing/data_preprocessing.py`) validates schema, converts date fields,
standardizes types, checks missing values/duplicates/negative targets, one-hot-encodes categoricals, and
scales numeric features via a `ColumnTransformer` fit only on the training split. The identical
`engineer_features()` function (in `src/aws/sagemaker/scripts/train.py`) is reused for both the SageMaker
training job and the batch-transform inference entry point, with a static `NUMERIC_COLS` column-order
constant shared between them — this was a real integration point that needed an explicit fix (see Lessons
Learned) once CSV serialization without headers was in play. Both local pipeline execution and a
**SageMaker Processing Job** (`src/aws/sagemaker/pipeline_scripts/preprocess.py`, run inside the managed
Pipeline) are implemented.

## Training Data

Data is sorted chronologically and split by a single global date cutoff: **70% training / 15% validation /
15% test**, applied identically across every store/product series (no shuffling, since shuffling would
leak future rows into training). Confirmed split: test set covers 10,500 rows with cutoffs at 2023-06-05
and 2023-09-18.

The target is `Units Sold`, already present in the source dataset — no manual labeling was performed or
needed. Lag/rolling features are computed on the full chronologically sorted series *before* splitting,
but only look strictly backward within each (store, product) group, so this does not constitute leakage —
it mirrors exactly what would be available in production at forecast time.

## Feature Engineering

Implemented features (`src/feature_engineering/feature_engineering.py`):

- **Calendar:** day-of-week, month, quarter, year, week-of-year, is-weekend
- **Lag:** `Units Sold` lagged 1, 7, 14, 30 days per (store, product) series
- **Rolling:** 7-, 14-, 30-day rolling mean and standard deviation of `Units Sold`, shifted so the current
  row never sees its own target
- **Categorical:** `Store ID`, `Product ID`, `Category`, `Region`, `Weather Condition`, `Seasonality`,
  one-hot encoded; `Holiday/Promotion` used directly as a binary indicator
- **Sequences:** sliding (sequence_length=30, n_features) windows per series for the LSTM/Transformer models

All scaling/encoding is fit only on the training split and applied unchanged to validation, test, and
inference data, satisfying the "fitted only on training data" requirement from Deliverable 1.

## Model Training & Evaluation

All four real model families from Deliverable 1's plan were implemented and compared, plus the
seasonal-naive baseline:

| Model | Val WAPE | Val MAE |
|---|---|---|
| **LSTM (selected)** | 0.647 | 88.7 |
| Classical (tuned Ridge) | 0.651 | 89.2 |
| Transformer | 0.654 | 89.7 |
| Seasonal-naive | 0.879 | 120.5 |

**Test set (10,500 rows, held out, never touched during tuning):**

| Metric | Selected model (LSTM) | Seasonal-naive |
|---|---|---|
| MAE | 87.93 | 118.62 |
| RMSE | 108.33 | — |
| WAPE | 0.644 | 0.868 |
| sMAPE | 0.716 | — |

**MAE improvement over the naive baseline: 25.9%.**

Classical models (Ridge, Random Forest, Gradient Boosting) were tuned with `TimeSeriesSplit`-based
`RandomizedSearchCV` (never a shuffled K-fold, per Deliverable 1's leakage-safety requirement). LSTM and
Transformer were trained as global models across every store/product series with early stopping on
validation loss. Training ran on `ml.m5.xlarge` SageMaker training-job instances (CPU was sufficient at
this dataset's scale; a GPU instance such as `ml.g4dn.xlarge` is noted in code comments as an option for
larger datasets).

**Finding, as predicted by Deliverable 1's own caution against "model complexity alone":** all four real
models land within ~1% WAPE of each other, well ahead of the naive baseline but far from a low-error
forecast. This is directly attributable to the EDA finding of negligible weekly seasonality/promotion
effect in this synthetic dataset — there is little additional structure for the more expressive
LSTM/Transformer to exploit beyond what lag/rolling features already give the classical models. The
meaningful, reproducible result is the ~26% MAE improvement over a trivial baseline, not the sub-1%
difference between model families.

## Model Deployment

**Batch inference is implemented as the primary deployment pattern**, confirmed with Deliverable 1's own
reasoning: retail demand forecasts are generated on a schedule, not per individual request.
`src/aws/sagemaker/batch_inference.py` builds the batch input from the test split, launches a SageMaker
Transform job against the registered model artifact, and writes forecasts to
`s3://<bucket>/aai540/retail-forecasting/predictions/`. End-to-end success was confirmed: a completed
transform job (`retail-forecasting-batch-transform-...`) produced 7,500 predictions with plausible
values (~120–148 range), matching the selected model's expected output scale.

**Instance type used (confirmed from successful runs): `ml.m5.xlarge`**, 1 instance, for both training and
batch transform. A real-time endpoint remains available as an optional extension
(`src/aws/sagemaker/deployment.py`, `scripts/inference.py`) but was not the primary demonstrated pattern,
consistent with Deliverable 1's scope. No endpoints were left running between sessions to avoid ongoing
charges.

## Model Monitoring

All three monitoring layers targeted in Deliverable 1 are implemented:

- **Infrastructure monitoring** (`src/aws/sagemaker/infrastructure_monitoring.py`): a CloudWatch dashboard
  (`ml-system-monitoring`) built from `SEARCH()` metric-math expressions tracks pipeline execution
  success/failure, per-step failures, CPU/memory for training/processing/batch-transform jobs, Feature
  Store call volume, S3 bucket size, and (if deployed) endpoint latency/errors — all scoped to this
  project's actual resources rather than wildcard dimensions.
- **Data monitoring** (`monitoring_reports.py data-quality`): a SageMaker Model Monitor baseline is
  suggested from the engineered training features, then an on-demand analysis job compares the
  batch-inference input against that baseline, producing `constraint_violations.json` and a Population
  Stability Index (PSI) drift table. Expected and confirmed drift: calendar features (`month`,
  `week_of_year`, `quarter`) drift because the test period (Sep–Dec) differs from training's January start;
  lag/rolling demand features remain stable.
- **Model/forecast-accuracy monitoring** (`monitoring_reports.py model-quality`): joins batch-transform
  predictions against actual `Units Sold` for the same rows, reports MAE/RMSE/WAPE overall and segmented by
  store/category/region against the seasonal-naive baseline, and publishes the results to the
  `RetailForecasting/ModelQuality` CloudWatch namespace feeding the infrastructure dashboard.

A detected drift signal is treated as a prompt for investigation, not proof of degraded accuracy — consistent
with Deliverable 1's stated monitoring philosophy.

## Model CI/CD

The CI/CD workflow matches Deliverable 1's target exactly, with a concrete passing/failing demonstration:

- **GitHub commit → automated checks:** `.github/workflows/ci-cd.yml` runs lint + pytest on every push/PR,
  then a Docker build (push disabled on PR/push events after an early iteration tried to push to Docker Hub
  with no configured secrets, which failed every run for infrastructure reasons unrelated to code — fixed).
- **SageMaker Pipeline, triggered manually (`workflow_dispatch`) or via the Studio CLI**, runs
  `Preprocess → TrainModel → EvaluateModel → CheckQualityGate → RegisterModel | QualityGateFailed`.
- **Quality gate:** test WAPE ≤ `MaxTestWAPE` (default 0.70) **and** at least `MinWAPEImprovement` (default
  10%) better than the seasonal-naive baseline. Ridge scores ~0.64 test WAPE vs. ~0.87 for seasonal-naive
  (26% better), so the default configuration passes.
- **Demonstrated failure case:** `python -m src.aws.sagemaker.pipeline run --display-name gate-fail
  --max-wape 0.50` deliberately tightens the threshold below what any candidate can clear, producing a real
  `QualityGateFailed` step in the pipeline execution graph — exactly the "one real failed execution" called
  for in Deliverable 1.
- **Tests:** schema/data-quality checks, unit tests for preprocessing and feature generation (including an
  explicit leakage-safety assertion), inference input/output contract tests (the pickled model round-trips
  through the exact header-less CSV format used in production), and pipeline tests
  (`tests/aws/test_pipeline.py`).

## Security Checklist, Privacy and Other Risks

| Checklist item | Design decision / control (confirmed) |
|---|---|
| Does the system store/process PHI? | No. Confirmed — dataset contains no health records. |
| Does the system store/process PII? | No. Dataset schema reviewed; no personal identifiers present. |
| Is user behavior tracked and stored? | No individual user tracking performed. |
| Does the system store/process credit card information? | No. |
| S3 read/write locations | `s3://sagemaker-us-east-1-<account-id>/aai540/retail-forecasting/{raw,processed,predictions,monitoring}/` (actual bucket/prefix, not a placeholder) |
| IAM | AWS Academy Learner Lab `LabRole` (least-privilege available in the lab); no access keys embedded in code |
| Encryption and access | S3 Block Public Access enabled throughout; TLS in transit via AWS SDK defaults |
| Secrets and source control | No credentials, tokens, or large model artifacts committed; `.gitignore` covers generated files, local env files |

**Bias/ethical risks, as identified in Deliverable 1, remain valid and were further confirmed by EDA:** the
dataset is synthetic and near-perfectly balanced across stores/products/categories/regions, which actually
*reduces* the risk of aggregate metrics masking poor performance on underrepresented segments (compared to
a real retail dataset) — but also means results here should not be read as evidence the approach would
perform identically on a real, imbalanced retail dataset with genuine stockout-driven demand censoring.
No claims of real-world stockout or holding-cost reduction are made.

## Engineering Challenges & Lessons Learned *(new section, not present in Deliverable 1)*

Deliverable 1 was written before implementation and could not anticipate the specific integration issues
encountered. These are documented here as evidence of MLOps maturity and debugging discipline, each
resolved via a tracked, reviewed GitHub PR:

1. **Feature Store name sanitization.** SageMaker Feature Store rejects feature names containing spaces or
   slashes (e.g. `Holiday/Promotion`, `Store ID`). Fixed with a `_sanitize_feature_names()` step before
   feature-group creation; confirmed working in production via the Feature Store console schema.
2. **Serving-container dependency conflicts.** The SKLearn serving container crash-looped
   (`TypeError: Descriptors cannot be created directly`) because installing TensorFlow for the LSTM pulled
   in `protobuf==4.25.9`, breaking the container's own protobuf-dependent serving stack, which required
   `protobuf<=3.20.3`. Separately, a scikit-learn version mismatch between the training environment (1.5.2)
   and the SKLearn 1.2-1 serving container caused a numpy ABI unpickling error. Both fixed by pinning exact
   compatible versions in `scripts/requirements.txt`.
3. **CSV header/schema mismatch.** The batch-input writer emitted header-less CSV, but the inference
   entry point's `pd.read_csv` call implicitly treated the first data row as a header, causing `KeyError`s
   when the fitted `ColumnTransformer` looked up feature columns by name. Fixed by introducing a shared,
   static `NUMERIC_COLS` ordering constant used by both the writer and reader.
4. **SageMaker SDK v2→v3 migration breaking changes, mid-project.** After an AWS Academy Learner Lab
   session restart (4-hour session limit), a fresh Studio kernel surfaced two unrelated hard failures:
   `sagemaker.model_monitor` raising `ModuleNotFoundError` as "removed in v3," and `sagemaker.Session`
   missing from the top-level namespace entirely. Root cause: the restarted kernel's base environment
   provided SageMaker functionality via AWS's newer modularized SDK components (`sagemaker-core`,
   `sagemaker-train`, `sagemaker-serve`, etc.) rather than the project's pinned classical `sagemaker`
   PyPI package (`>=2.190.0,<3.0.0`), which had never been (re-)installed in that kernel. Fixed by
   reinstalling the pinned dependency and, separately, moving a module-level `DefaultModelMonitor` import
   to be lazily scoped to only the functions that use it (matching an existing lazy-TensorFlow-import
   pattern elsewhere in the codebase), so the module still loads cleanly for callers that don't need Model
   Monitor baselining.
5. **CloudWatch dashboard JSON schema.** A `PutDashboard` call failed with 26 validation errors because a
   helper function returned bare metric-math expression objects instead of wrapping each one in its own
   single-element array, which CloudWatch's dashboard schema requires for every entry in a widget's
   `metrics` list. Fixed, and a regression test was added asserting every metrics entry is an array.
6. **Batch-transform output format ambiguity.** The monitoring report's model-quality join assumed batch
   output was plain CSV (one prediction per line), but the SKLearn container defaults to a JSON-wrapped
   `{"predictions": [...]}` response when no `Accept` type is requested. Fixed by making the parser handle
   both formats and explicitly requesting `text/csv` on the Transformer going forward for deterministic
   output.

Each of the above was caught, diagnosed, and fixed via a real PR with CI verification — not assumed away —
which is itself evidence of the "testable repository" and "documented... system" goals from Deliverable 1.

## Future Enhancements

Unchanged from Deliverable 1, still valid priorities for continued work:

- Scheduled retraining and automated promotion gated on the existing WAPE quality check
- Forecast uncertainty (prediction intervals / quantile forecasts) and evaluation of how they'd support
  safety-stock policy
- Verified external demand drivers (supplier lead time, weather, regional events) if a real (non-synthetic)
  dataset is adopted
- Cold-start handling for new products/stores using category-level or hierarchical forecasting
- Expanded alerting on sustained accuracy degradation and inference failures (beyond the current dashboard)
- Infrastructure-as-code (e.g. CDK/Terraform) for the SageMaker resources, time permitting
- Migration path to SageMaker Python SDK v3, given the v2 deprecation warnings encountered throughout this
  project and documented in Lessons Learned above

## References

- Amazon Web Services. (n.d.). *Amazon SageMaker AI documentation*. https://docs.aws.amazon.com/sagemaker/
- Amazon Web Services. (n.d.). *Amazon SageMaker Feature Store*. https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store.html
- Amazon Web Services. (n.d.). *Amazon SageMaker Pipelines*. https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines.html
- Amazon Web Services. (n.d.). *Amazon SageMaker Model Registry*. https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html
- Amazon Web Services. (n.d.). *Amazon CloudWatch documentation*. https://docs.aws.amazon.com/cloudwatch/
- Chauhan, A. (n.d.). *Retail store inventory forecasting dataset* [Data set]. Kaggle. https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset
- Christenson, M. (n.d.). *AAI-540 labs* [Source code]. GitHub. https://github.com/mechristenson/aai-540-labs
- Dwivedi, V. P., & Group 4. (2026). *Retail store inventory forecasting* [Source code]. GitHub. https://github.com/vedpd/AAI_540_Group4_Retail_Store_Inventory_Forecasting
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation, 9*(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735
- Vaswani, A., et al. (2017). Attention is all you need. *Advances in Neural Information Processing Systems, 30*.

---

**Submission note:** This document updates Deliverable 1's section structure with confirmed, as-built
facts from the completed system, and adds a Lessons Learned section. Remaining placeholders (team member
names, Asana/Team Tracker links) still need to be filled in before final submission.
