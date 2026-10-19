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

### Impact Measurement

Impact is measured against the business goal stated above — reducing the forecasting error that
drives both stockouts and excess inventory — using **WAPE (Weighted Absolute Percentage Error)** as
the primary metric, with MAE and RMSE reported alongside and sMAPE as a supplementary check. WAPE was
chosen specifically because it stays well-defined across the many low-volume store/product/day rows
in this dataset, where a metric like MAPE would be dominated by division-by-near-zero noise; this
matters for the business goal because a metric that misbehaves on low-volume combinations would
mask exactly the segments most prone to stockouts. Success is defined relative to a **seasonal-naive
baseline** (predicting the value from 7 days earlier) rather than an absolute error target, because
the business question is "does this model add value over the simplest possible forecasting
approach," not "is the error below some arbitrary number" — the latter would be meaningless without
knowing how hard this specific dataset is to forecast.

Concretely, the project's quality gate operationalizes this measurement: a candidate model must
achieve a test-set WAPE at or below 0.70 **and** at least a 10% relative WAPE improvement over the
seasonal-naive baseline before it is allowed to register in the Model Registry (configured in
`config.yaml: training.quality_gates` and enforced identically in
`src/aws/sagemaker/pipeline_scripts/evaluate.py`). The selected LSTM model achieved a test WAPE of
0.644 versus 0.868 for the baseline — a 25.9% MAE improvement — which both clears the gate and gives
a concrete, falsifiable number tied directly to the stated goal of measurably reducing forecast
error, rather than a vague claim of "the model works."

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

Deliverable 1 flagged that the exact files, row counts, and date coverage of the Kaggle dataset would
need to be confirmed by profiling once downloaded, since the README's original description assumed
separate `train.csv`/`test.csv`/`stores.csv`/`items.csv` files typical of other retail-forecasting
datasets on Kaggle. That assumption turned out to be wrong: the actual dataset is a **single CSV**
(`data/raw/retail_store_inventory.csv`, ~6 MB, committed to the repository since it is synthetic and
carries no privacy concern) with one row per (store, product, day) combination — 73,100 rows covering
5 stores (`S001`–`S005`), 20 products (`P0001`–`P0020`), and 731 days from 2022-01-01 to 2024-01-01.
Profiling confirmed no missing values and no duplicate `(Date, Store ID, Product ID)` keys, so no
deduplication or imputation logic was needed before feature engineering.

The dataset schema review also surfaced a target-leakage risk that drove a concrete design decision:
the `Demand Forecast` column correlates 0.997 with `Units Sold`, because it is the dataset's own
precomputed forecast rather than an independent signal — training on it would let the model simply
copy an answer it shouldn't have access to at prediction time. `Inventory Level` and `Units Ordered`
carry a similar problem in the other direction, since both are partly determined by `Units Sold`
*after* the fact (today's inventory level reflects what already sold). All three columns are excluded
from every model's feature set for this reason. The remaining columns — `Category`, `Region`, `Price`,
`Discount`, `Weather Condition`, `Holiday/Promotion`, `Competitor Pricing`, and `Seasonality` — were
reviewed and retained as the basis for calendar and lag/rolling features described below.

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

`src/feature_engineering/feature_engineering.py` builds two families of features reused identically
by every model and, critically, by the inference code path as well — the same `engineer_features()`
function runs during both training and batch-transform scoring, so there is no risk of training/serving
skew from a feature definition drifting between the two. The first family is calendar-derived:
day-of-week, month, quarter, year, week-of-year, and an is-weekend flag, all deterministic functions of
the `Date` column. The second, more important family is per-(store, product) historical demand
signal: `Units Sold` lagged 1, 7, 14, and 30 days, plus 7-, 14-, and 30-day rolling mean and standard
deviation. These lag/rolling values are computed on the full chronologically sorted series before the
train/validation/test split is applied, but each one only looks strictly backward within its own
(store, product) group — the rolling window is shifted so a row never sees its own target — which
mirrors exactly what would be available at forecast time in production and does not constitute
leakage, satisfying Deliverable 1's "fitted only on training data" and leakage-safety requirements.

Categorical columns (`Store ID`, `Product ID`, `Category`, `Region`, `Weather Condition`,
`Seasonality`) are one-hot encoded, while `Holiday/Promotion` is used directly as a binary indicator
rather than being one-hot encoded, since it is already a two-valued flag. For the sequence models
(LSTM and Transformer), an additional step builds sliding windows of shape
`(sequence_length=30, n_features)` per store/product series, since those architectures consume a
fixed-length history rather than a single row of lag/rolling summaries. All scaling (for numeric
features) and one-hot encoding (for categoricals) is fit only on the training split via a
`ColumnTransformer`, then applied unchanged to validation, test, and inference data — a decision that
directly prevents information from the validation/test periods leaking into how features are scaled.

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

All three monitoring layers targeted in Deliverable 1 — infrastructure, data quality, and forecast
accuracy — are implemented and have each been run against real pipeline output, not just coded and
left untested. Infrastructure monitoring (`src/aws/sagemaker/infrastructure_monitoring.py`) builds a
CloudWatch dashboard, `ml-system-monitoring`, out of `SEARCH()` metric-math expressions rather than
hardcoded resource names, so pipeline executions, per-step failures, CPU/memory for
training/processing/batch-transform jobs, Feature Store call volume, S3 bucket size, and (when
deployed) endpoint latency/errors are all tracked automatically as new jobs run, without needing to
edit the dashboard each time a new job name appears. This design choice directly addresses a failure
mode Deliverable 1 only described abstractly ("a dashboard should summarize the key metrics") by
making the dashboard self-maintaining rather than a static snapshot that goes stale.

Data-quality monitoring (`monitoring_reports.py data-quality`) runs a genuine SageMaker Model Monitor
baseline job against the engineered training features, then an on-demand analysis job that compares
the batch-inference input against that baseline, producing `constraint_violations.json` plus a
Population Stability Index (PSI) drift table computed independently in the report for readability.
Running this for real surfaced exactly the drift Deliverable 1's design anticipated in principle:
calendar features (`month`, `week_of_year`, `quarter`) show large PSI values because the test period
(September–December) covers a different slice of the calendar than training (which starts in
January), while every demand-driving feature — all lag and rolling `Units Sold` statistics, `Price`,
`Discount`, `Holiday/Promotion` — stayed essentially unchanged (PSI under 0.02). This outcome matters
for interpretation: a detected drift signal is treated as a prompt for investigation, not automatic
proof of degraded accuracy, and in this case the investigation itself confirms the drift is benign
and expected rather than a sign the model is failing.

Forecast-accuracy monitoring (`monitoring_reports.py model-quality`) closes the loop by joining
batch-transform predictions against the actual `Units Sold` values for the same rows once they are
known, reporting MAE/RMSE/WAPE both overall and segmented by store, category, and region against the
seasonal-naive baseline, then publishing the results to the `RetailForecasting/ModelQuality`
CloudWatch namespace that feeds the infrastructure dashboard. Segmenting by store/category/region was
a deliberate choice beyond what Deliverable 1 strictly required, so that an aggregate metric looking
healthy cannot silently hide one underperforming segment.

## Model CI/CD

The CI/CD workflow matches Deliverable 1's target closely, but required an iteration to get right:
the first version of `.github/workflows/ci-cd.yml` tried to push a Docker image to Docker Hub and
run a deploy/rollback job against AWS on every push and pull request, with no Docker Hub credentials
and no AWS credentials configured for those trigger types. That meant the pipeline failed on *every*
run for infrastructure reasons that had nothing to do with whether the code itself was correct — a
genuinely misleading signal for a CI system to send. The corrected workflow runs lint and the full
pytest suite plus a Docker build (with `push: false`) on every push/PR, and only triggers the actual
SageMaker Pipeline run through `workflow_dispatch` (manual trigger) or the Studio CLI, which is the
appropriate place for a step that needs real AWS credentials and costs real compute time.

The SageMaker Pipeline itself (`src/aws/sagemaker/pipeline.py`) implements the DAG
`Preprocess → TrainModel → EvaluateModel → CheckQualityGate → RegisterModel | QualityGateFailed`, with
the quality gate requiring a test WAPE at or below `MaxTestWAPE` (default 0.70) *and* at least
`MinWAPEImprovement` (default 10%) better than the seasonal-naive baseline before registration is
allowed. The default configuration passes — Ridge scores roughly 0.64 test WAPE versus 0.87 for
seasonal-naive, a 26% improvement — but to demonstrate the gate is a real control rather than a
decorative step, a second execution was run with `--max-wape 0.50`, a threshold deliberately set
below what any candidate model can clear, producing a genuine `QualityGateFailed` step in the
execution graph. This satisfies Deliverable 1's call for "one real failed execution" with an actual
AWS Step Functions-style graph showing the failure, not a simulated or described one.

Test coverage includes schema/data-quality checks, unit tests for preprocessing and feature
generation with an explicit leakage-safety assertion, inference input/output contract tests (the
pickled model round-trips through the exact header-less CSV format used in production, which is
precisely the integration point that broke once during development — see Lessons Learned), and
dedicated pipeline tests in `tests/aws/test_pipeline.py`. Model approval remains a separate, manual
step in the Model Registry after a successful pipeline run, so a passing quality gate registers a
candidate for review rather than promoting it to production automatically.

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
