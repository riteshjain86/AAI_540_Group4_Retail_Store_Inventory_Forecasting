# Demo Video Script — Retail Demand Forecasting MLOps Pipeline

Target length: ~8-10 minutes. Each segment lists what to say and what to show on screen.
Trim segments 5/7 first if you need to shorten; segments 1-4 and 6 are the core story.

---

## 1. Intro & Problem Framing (45s)

**Show:** Title slide or README top section.

**Say:**
"This is Group 4's retail demand forecasting project for AAI-540. The business problem: retail
stores need to forecast daily product demand to avoid two costly mistakes — stockouts from
underestimating demand, and excess inventory/markdowns from overestimating it. We built an
end-to-end MLOps pipeline on AWS SageMaker that goes from raw historical sales data all the way
to a monitored, automatically-gated production model."

---

## 2. Data & Approach (60s)

**Show:** `notebooks/exploratory_data_analysis.ipynb` — scroll to the seasonality/promotion-effect
findings; or `README.md`'s Dataset Information section.

**Say:**
"We used a Kaggle retail inventory dataset — 73,100 daily records across 5 stores and 20 products,
over two years. One important finding from our EDA: this synthetic dataset has almost no real
weekly seasonality and no measurable promotion effect — mean units sold with or without a
promotion differ by less than 0.1%. That's not a bug, it's a real characteristic of the data, and
it matters later when we explain why our fancier models don't dramatically outperform simpler ones.

We also excluded three columns that leak the target: `Demand Forecast` correlates 0.997 with
`Units Sold` by construction, and `Inventory Level`/`Units Ordered` are partly determined by sales
after the fact."

---

## 3. Feature Engineering & Model Comparison (90s)

**Show:** `src/feature_engineering/feature_engineering.py` briefly, then the results table from
`docs/DESIGN_DOCUMENT.md` or README (Model Training section).

**Say:**
"We engineered calendar features, and per-store-per-product lag and rolling-window features —
1/7/14/30-day lags, 7/14/30-day rolling mean and standard deviation — always computed looking only
backward in time, so there's no leakage into the model's own prediction.

We compared five approaches: a seasonal-naive baseline, tuned Ridge regression, Random Forest,
Gradient Boosting, and two deep learning models — an LSTM and a Transformer encoder. All four real
models land within about 1% WAPE of each other. We selected the LSTM — test WAPE 0.644 versus 0.868
for the naive baseline, which is a 25.9% improvement in mean absolute error. The meaningful result
here isn't the small gap between model families — given what we found in the EDA, that's expected —
it's the sizeable improvement over a trivial baseline."

---

## 4. AWS Architecture Walkthrough (60s)

**Show:** `docs/architecture_diagram.png`, or narrate over the README architecture diagram.

**Say:**
"On AWS, the pipeline runs: raw data to S3, feature engineering into a SageMaker Feature Store,
a managed SageMaker Pipeline that trains, evaluates against a quality gate, and conditionally
registers the model, then batch inference as our primary deployment pattern — because retail
forecasts are generated on a schedule, not per individual request — followed by monitoring that
feeds back into CloudWatch dashboards."

---

## 5. Live AWS Console Tour (3 min) — the core evidence segment

This is the most important segment for the rubric: it's where you **prove** things actually ran on
AWS, not just that code exists in a repo. Think of it as walking a non-technical viewer through
"data in → model built → model checked → model used → model watched," in that order, clicking to
each place as you say it. Have every tab already open in your browser *before* you start recording
(see the checklist at the bottom) so you're just switching tabs, not navigating live.

**The five stops, in order, and why that order:** Feature Store (where the data lives) → Pipeline
(how the model got built and checked) → Model Registry (the checkpoint before anything goes live)
→ Batch output (the actual result the model produced) → Dashboards (how we keep watching it after
the fact). That's the natural "data → build → gate → result → watch" story.

### 5a. Feature Store — "here's where our data lives" (25s)
**Click path:** SageMaker Studio (left sidebar) → **Data** → **Feature Store** →
`retail-inventory-features-raw` → **Schema** tab.

**Point at:** the list of feature names in the schema table.

**Say (plain language):** "This is our Feature Store — think of it as a managed, versioned table of
the data our model learns from. Notice names like `Holiday_Promotion` and `Store_ID` — in the raw
file these had a slash and a space, like `Holiday/Promotion` and `Store ID`. AWS's Feature Store
won't accept names with those characters, so we wrote a small cleanup step that renames them before
they get stored here. Small detail, but it's the kind of real-world fix you only find by actually
running the thing."

### 5b. SageMaker Pipeline — "here's how the model got built and checked" (50s)

**Before you record:** this is a different mechanism than the training/batch-transform/monitoring
commands used elsewhere in this script — those are one-off jobs (`training.py`, `batch_inference.py`,
`monitoring_reports.py`) that show up under **Training jobs** / **Transform jobs** / **Processing
jobs**, not under **Pipelines**. The Pipelines tab only populates once you actually create and run
the managed `Pipeline` resource from `src/aws/sagemaker/pipeline.py`. If you've never done that yet,
run this first (from Studio, well before recording — each run takes several minutes):
```bash
python -m src.aws.sagemaker.pipeline upsert
python -m src.aws.sagemaker.pipeline run --display-name gate-pass --wait
python -m src.aws.sagemaker.pipeline run --display-name gate-fail --max-wape 0.50 --wait
```
`upsert` registers the pipeline definition itself (makes it show up in the Pipelines tab at all);
the two `run` calls create the passing and failing executions you'll show on camera.

**Click path:** Studio → **Pipelines** (left sidebar, or "Pipelines" under Deployments) →
`retail-forecasting-pipeline` → **Executions** tab → open the most recent passing execution, then
go back and open the failing one.

**Point at:** the pipeline graph — a left-to-right (or top-to-bottom) flow of boxes:
`Preprocess → TrainModel → EvaluateModel → CheckQualityGate → RegisterModel`. On the passing run,
every box is green. On the failing run, point at the red `QualityGateFailed` box.

**Say (plain language):** "This diagram is the actual automated pipeline: it preprocesses the data,
trains a model, evaluates it, then hits a checkpoint we call the quality gate. On this first run,
every step is green — the model passed the checkpoint and got registered automatically. To prove
this checkpoint isn't just decorative, we ran it a second time with the bar set deliberately too
high for any model to clear — see this red box here, `QualityGateFailed`? That means a bad model
gets stopped *before* it's ever registered, instead of silently going live. That's the automated
quality check actually working, not just described in a document."

### 5c. Model Registry — "here's the checkpoint before anything goes live" (25s)
**Click path:** Studio → **Models** → **Registered models** → `retail-inventory-forecasting-models`
→ scroll the version list.

**Point at:** the status column — one version marked `Rejected`, another marked `Approved`.

**Say (plain language):** "Every model that passes the pipeline lands here, in the Model Registry,
as 'pending approval' — it doesn't go live automatically. You can see we actually rejected a version
once — that one was trained with a library version that turned out to be incompatible at serving
time — and approved this corrected version instead. This is a real, manual control point, not a
rubber stamp."

### 5d. Batch transform output — "here's the actual result" (25s)
**Click path:** S3 console → our bucket → `aai540/retail-forecasting/predictions/` →
`batch_input.csv.out` → **Open** (or in a terminal: `aws s3 cp s3://.../batch_input.csv.out - | head`).

**Point at:** the actual numbers in the file.

**Say (plain language):** "This is the real output — 7,500 individual demand forecasts, one per
store/product/day combination in our test period, sitting in S3 exactly where a downstream
inventory system would pick them up. These aren't made-up numbers for the demo; this file was
produced by an actual SageMaker batch transform job running the approved model."

### 5e. Monitoring — "here's how we keep watching it" (50s)

This stop has two halves: a real CloudWatch dashboard (for infrastructure/job health and
forecast-accuracy trends), and — important nuance — AWS doesn't build a slick visual dashboard for a
one-off data-quality analysis the way it does for a continuous monitoring schedule on a live
endpoint, so for that half we show the actual Processing Job that ran plus our own generated report.
Don't imply there's a native drift-chart page in SageMaker for this part — there isn't one for this
on-demand analysis type, and claiming there is will look bad if a viewer tries to find it themselves.

**Half 1 — infrastructure & accuracy dashboard (20s)**
**Click path:** CloudWatch console → **Dashboards** (left sidebar) → `ml-system-monitoring` → scroll
through 2-3 widgets.
**Point at:** one widget showing job health/CPU, one widget showing the MAE/WAPE accuracy numbers
(`RetailForecasting/ModelQuality` namespace).
**Say:** "This CloudWatch dashboard automatically tracks every pipeline run and training/batch job's
health, and separately tracks forecast accuracy over time, using metric-math expressions so new jobs
show up without us editing anything."

**Half 2 — data-quality evidence (30s)**
**Click path:**
1. SageMaker Studio → **Jobs** → **Processing jobs** → find `retail-dq-baseline-<timestamp>` and
   `retail-dq-analysis-<timestamp>` → click one open to show status `Completed` and its S3 output
   location.
2. Switch to a text editor tab showing `reports/monitoring/data_quality.md` (and
   `model_quality.md`).

**Point at:** the `Completed` processing-job status, then the PSI table in the markdown report —
specifically the three `ALERT` rows (`week_of_year`, `month`, `quarter`) next to the many `ok` rows.

**Say (plain language):** "There isn't a single slick built-in chart for this kind of one-off
check — that richer visualization only exists for a continuous schedule on a live endpoint, which
we don't run here since batch inference is our pattern. So instead, here's the actual SageMaker
Processing Job that ran the analysis — completed successfully — and here's our own report built
from its output. We also generate a model-quality report comparing predictions against what
actually sold — MAE about 88 versus 119 for a naive guess-the-same-as-last-week baseline. The
data-quality report flagged drift in our calendar features — `month`, `week_of_year`, `quarter` —
while every demand-related feature stayed completely stable. That's expected, not a bug: our test
period is September through December, while training started in January, so the calendar values
themselves are naturally different ranges. That's exactly the kind of signal this monitoring is
supposed to catch, and it's reassuring that it didn't flag anything on the features that actually
drive the forecast."

### 5f. (Optional) LSTM real-time endpoint — "here's the other deployment pattern" (90s)

Batch transform is the primary pattern (shown in 5d), but the project also supports a real-time
endpoint as an optional extension. This segment only works live if you deploy the endpoint
*immediately before* recording this part, since it costs money per hour it's running — see the
Recording checklist. Deployment and invocation are run from the terminal (matching every other AWS
component in this project — this is not done in a notebook; see Group 5's reference repo, which
uses the identical CLI pattern), but the **proof that it's real** is shown live in the console.

**Before recording:** run the deploy step from the terminal in the background while you set up the
camera, since provisioning takes 5-8 minutes. Keep the invoke+delete script ready to paste
immediately after you finish this segment on camera, so the endpoint doesn't sit idle.

**Click path:**
1. SageMaker Studio (or console) → **Inference** → **Endpoints** → find your endpoint
   (`retail-forecasting-endpoint-<timestamp>`) → point at status **InService**.
2. Switch to the terminal tab where you already ran the invoke script → show the printed
   `Model prediction: ...` line next to the `Ground truth: 40` comment.
3. (Optional) Endpoint detail page → **Data capture** tab → show that request/response capture is
   enabled, writing to `s3://.../monitoring/data-capture/`.
4. Switch back to the terminal and run the delete command live, on camera, so the recording itself
   is evidence the endpoint didn't get left running.

**Say (plain language):** "Separately from batch transform, the project also supports deploying the
same LSTM model behind a real-time endpoint for on-demand scoring. Here it is, status InService in
the console. I fed it a real 30-day window of actual historical data for one store and product, and
here's the terminal output — the model's prediction next to the real value that was actually sold
the next day, so you can see the mechanism genuinely works end to end, not just that it was coded.
Real-time endpoints bill per hour regardless of how many requests you send, so the very next thing I
do, right now on camera, is delete it."

---

## 6. CI/CD (45s)

**Show:** GitHub repo → Actions tab, a recent workflow run; optionally scroll through PR history
showing the sequence of merged fixes.

**Say:**
"On the software engineering side, every push and PR runs linting and our full pytest suite,
including leakage-safety tests and inference contract tests, plus a Docker build. The SageMaker
Pipeline itself can be triggered from GitHub Actions on manual dispatch. You can see our PR
history here — eighteen PRs, each one a focused, reviewed fix, with CI passing before merge."

---

## 7. Engineering Challenges — Lessons Learned (60s) — optional but strong for rubric

**Show:** `docs/DESIGN_DOCUMENT.md`'s "Engineering Challenges & Lessons Learned" section.

**Say:**
"I want to highlight a few real problems we hit and fixed, because debugging a managed ML platform
is its own skill. Our SKLearn serving container crash-looped because installing TensorFlow for the
LSTM pulled in an incompatible protobuf version that broke the container's own serving stack.
Separately, a scikit-learn version mismatch between training and serving caused a numpy ABI
unpickling error. We also hit a CSV header mismatch between our training and inference code, a
CloudWatch dashboard schema bug, and — midway through the project — an AWS Academy Lab session
restart silently swapped in a different SageMaker SDK build that broke imports we were relying on.
Every one of these was caught, diagnosed, and fixed through a tracked GitHub PR with CI
verification before merging."

---

## 8. Wrap-up (30s)

**Show:** Back to README or the metrics table.

**Say:**
"To summarize: we built and compared five forecasting approaches, selected an LSTM with a 26%
MAE improvement over baseline, and deployed it through a fully monitored SageMaker pipeline with
an automated quality gate, model registry approval flow, and both infrastructure and
model-accuracy monitoring. Thanks for watching."

---

## Recording checklist (do this well before hitting record — the pipeline step takes several minutes)

- [ ] **Create and run the SageMaker Pipeline first, if you haven't already** — the Pipelines tab is
      empty until you do this (see section 5b for exactly why):
      ```bash
      python -m src.aws.sagemaker.pipeline upsert
      python -m src.aws.sagemaker.pipeline run --display-name gate-pass --wait
      python -m src.aws.sagemaker.pipeline run --display-name gate-fail --max-wape 0.50 --wait
      ```
      Confirm both executions show up under Studio → Pipelines → `retail-forecasting-pipeline` →
      Executions before you start recording.
- [ ] **(If doing the optional 5f segment) Deploy the LSTM endpoint 5-10 minutes before you start
      recording** so it's already `InService` by the time you reach that segment - don't make the
      camera wait on live provisioning. Have the invoke+delete script ready in a terminal tab, not
      yet run, so you can execute it live on camera during 5f and the recording itself proves the
      endpoint got deleted right after.
- [ ] SageMaker Studio open with Pipelines, Models, Feature Store, and Endpoints tabs ready
- [ ] CloudWatch dashboard `ml-system-monitoring` loaded with a reasonable time range set
- [ ] Both pipeline executions (`gate-pass`, `gate-fail`) already completed so you're not waiting live
- [ ] `reports/monitoring/model_quality.md` and `data_quality.md` open in an editor tab
- [ ] GitHub repo Actions tab and PR list open in a browser tab
- [ ] Any standing real-time endpoint deleted *before* recording unless demoing it live, to avoid
      idle billing after you're done
