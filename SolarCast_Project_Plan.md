# SolarCast - Project and Team Execution Plan

## SolarCast

Automated Energy Forecasting | Project and team execution plan

Forecast hourly solar energy. Flag substantial production deviations. Demonstrate a reproducible, deployed and monitored MLOps lifecycle.

### The project to build

Build a one-site system that predicts the next hourly AC energy output in kWh from historical production, calendar features and weather information available at prediction time. Add a separate expected-output estimator using observed weather for delayed production-deviation analysis. Compare simple regressors before considering complex time-series models.

### Course-aligned delivery

The attached 13-page specification requires exactly three students. This plan covers the base requirements, the 100-mark rubric and the Feast/BentoML addendum. Flask is the required cloud API; BentoML is an additional serving demonstration. Hugging Face is conditional and is unnecessary for the selected tabular regression task.

| Member | Primary ownership | Review partner |
| --- | --- | --- |
| Member 1 - name pending | Data, features, Feast, models and MLflow | Member 2 |
| Member 2 - name pending | Git/DagsHub, DVC, Airflow, Kubeflow and CI/CD | Member 3 |
| Member 3 - name pending | Flask/BentoML, containers, Kubernetes, AWS and observability | Member 1 |

### Planning assumptions

- Six weeks is a planning baseline, not a known course deadline. Rebase milestones when the submission date is provided.
- Names and skill preferences are unconfirmed. Member numbers are assignable roles; do not infer that the user is Member 1.
- Dataset/site selection is a first-week feasibility gate. No dataset has yet been downloaded, validated or used to train a model.
- This is an implementation plan. No repository, working service, cloud resources or experimental results are claimed as completed.

### Reading guide

2: requirement coverage | 3: data design | 4: models and evaluation | 5: architecture | 6-8: individual work | 9: shared contracts | 10: automation and deployment | 11: monitoring | 12: milestones | 13: delivery and risks | 14: references.

Prepared 11 November 2026. Primary authority: MLOps_Course_Project_Requirements (1).pdf, pages 1-13. External references [1]-[8] are listed on page 14. All numeric quality and alert thresholds below are proposed project settings, not course-prescribed values.

## 01 / Requirements coverage

A traceable plan for all 100 assessment marks

M1, M2 and M3 refer to Members 1, 2 and 3. A primary owner coordinates delivery; all members contribute tests, integration and defense preparation.

| Requirement / marks | Planned implementation and evidence | Owner |
| --- | --- | --- |
| Problem, data, modular ML / 6 | Site manifest, validated hourly energy, reusable modules, chronological evaluation. Spec pp. 2-3. | M1 |
| Git, DVC, DagsHub / 9 | PR history; DagsHub integration; DVC remote and pipeline; recover old code + data; conflict procedure. pp. 3, 9. | M2 + M1 |
| MLflow, tuning, registry / 9 | Comparable runs, bounded tuning, version lineage, model status and champion selection. p. 3. | M1 |
| Feast feature store / 6 | Entities, views, offline retrieval, online materialization/query, freshness and skew tests. pp. 12-13. | M1 + M3 |
| Airflow + Kubeflow / 12 | One DAG ingests/validates, triggers and waits for containerized KFP ML graph; retries and conditions. pp. 3-4, 6. | M2 |
| Tests + CI/CD + scans / 12 | All six test categories, Ruff, dependency/image scans, failed gate evidence and deployment workflow. p. 4. | M2 + all |
| Docker, Flask, BentoML / 10 | Three required Flask endpoints; Gunicorn; Compose; versioned Bento and Docker smoke test. pp. 4-5, 12-13. | M3 |
| AWS / 12 | S3 artifacts, ECR images, EC2 API, IAM, CloudWatch, approved deployment and cleanup. p. 5. | M3 + M2 |
| Kubernetes / 6 | Deployment/Service; probes; 1 to 3 replicas; pod replacement and traffic evidence. p. 6. | M3 |
| Monitoring + retraining / 6 | Prometheus/Grafana, drift, labels, alert, Airflow retrain, candidate decision and rollback. pp. 6-7. | M3 + M2 + M1 |
| Security and ethics / 2 | Implemented access controls, secrets, scans, dataset terms and project risk register. p. 7. | M3 + all |
| Documentation + defense / 10 | README, report, diagrams, evidence, contribution statements and shared rehearsal. pp. 9-11. | All |

Resolve specification inconsistencies conservatively: some example repository comments mention FastAPI, but the endpoint section, acceptance criteria and addendum explicitly require Flask. Keep Flask. Feast and BentoML remain in scope even though they appear later in the document.

Keep costly orchestration local when practical. AWS hosting of the core prediction API remains mandatory. A local-only demo does not satisfy the specification.

## 02 / Data and forecast design

First resolve the target, site, time axis and information availability

### Data sources and selection gate

Use public PVDAQ/OEDI measured generation as the preferred target-data route [1]. Inspect one system first; accept it only after checking download access, terms, coordinates, capacity, measurement units, timestamp conventions and useful historical coverage. Do not build against the discontinued PVDAQ V3 API [2]. Aim for two overlapping years, with one full seasonal cycle as the minimum working target; shorter coverage must be disclosed.

Join NASA POWER hourly weather for the same physical site and overlapping dates [3]. Candidate fields: ALLSKY_SFC_SW_DWN, T2M, RH2M and WS10M; verify parameter availability and response units before use. Explicitly request UTC. The service default is local solar time, which can differ from civil time.

NASA POWER is an observational archive with delayed updates [4], not an operational next-hour weather forecast feed. A future-weather observation must never be silently used as a forecast input.

| Mode | Inputs permitted at issue time | Claim permitted |
| --- | --- | --- |
| F1: next-hour forecast (core) | Production through issue time; calendar; last published NASA weather only; missingness and age flags. | Predict energy for the next hour. Backtest with an explicit availability policy. |
| E1: expected-output estimate | Weather observed during target hour, once it is published; calendar; approved estimator. | Delayed weather-adjusted production deviation; no real-time forecasting claim. |
| F2: forecast-weather extension | Archived weather forecasts whose issue timestamp precedes prediction time. | Future-weather-assisted forecast only after a matching archive is verified. |

### Canonical hourly data

- Primary key: site_id + interval_start_utc. Store interval_end_utc, target_kwh, capacity_kw, coordinates, source identifiers and quality flags.
- Read power/energy metadata before aggregation. Integrate kW over valid sample durations to get kWh; sum interval energy; difference cumulative meters with reset detection. Missing intervals are not zero production.
- Resolve UTC, civil timezone, daylight-saving ambiguity and interval start/end alignment. Do not pair one site's weather with another site's power.
- Retain immutable raw files and a manifest with URLs, checksums, units, terms, retrieval time and row counts. Version raw/processed outputs with DVC.
- Use feature_event_time and available_at_utc separately. Accept a forecast feature only when available_at_utc <= issue_time_utc. Record provider revisions and snapshot retrieval times.

Historical NASA files do not establish their original publication time. Until a vintage archive is available, simulate a conservative 14-day weather lag, label that assumption and compare 7/14/30-day sensitivity. Validate freshness in actual ingestion. If no suitable measured site is found in week 1, select another public measured-generation source with known coordinates and terms. Synthetic output is a labeled pipeline-demo fallback only.

## 03 / Modeling and evaluation

Forecast accuracy and production deviations are separate questions

### Small, comparable experiment set

- F1 baselines: previous-hour persistence and same-hour-previous-day persistence, with explicit missing-history fallbacks.
- F1 candidates: Ridge regression and HistGradientBoostingRegressor using lagged production, shifted rolling features, calendar cycles and availability-safe weather. Add Random Forest only if time allows.
- E1 estimator: separate weather-to-output regression using target-hour observed irradiance/temperature. Record its own run, inputs, metrics and threshold calibration.
- For each family use a small fixed tuning budget, for example 10 configurations, on identical chronological folds. No LSTM is needed to meet the course objectives.

### Leakage-safe validation

Reserve the latest 20% of valid time as a final untouched test set. Within the earlier 80%, use three expanding-window train/validation folds. Fit imputers, scalers and encoders only on each training fold. Generate lag and rolling features using strictly earlier observations; respect the forecast label horizon at split boundaries. No random row split, centered rolling averages or backward fill from future rows.

| Metric / decision | Proposed definition |
| --- | --- |
| Primary accuracy | Daylight MAE in kWh; daylight defined by site/time solar elevation > 0, independent of target values. Report all-hour MAE and RMSE too. |
| Scale and stability | Capacity-normalized MAE = MAE / (capacity_kw x 1 hour); daily energy error; monthly/daylight slices. Avoid MAPE around zero output. |
| Initial candidate gate | On chronological validation, >=5% daylight MAE improvement over the stronger persistence baseline; capacity-normalized daylight MAE <=0.20; finite, valid outputs. |
| Retraining gate | On the same recent labeled validation window, >=2% MAE improvement over champion and no predefined seasonal slice >5% worse; all mandatory tests pass. |
| Final test governance | Open the held-out test after selection; report it once. A disappointing test score is reported, not used for repeated tuning. Future retraining uses later rolling evaluation windows. |

### Production-deviation rule

For E1, residual = actual_kwh - expected_kwh. Calibrate lower/upper residual quantiles (1st/99th percentiles) on a reserved chronological calibration period, preferably within daylight/irradiance bins with enough samples. Flag a deviation when residual is outside its bin bounds for two consecutive valid daylight hours; suppress nights, missing labels and invalid weather. Tune only on calibration data, then freeze the rule.

Inject clearly labeled 30% production drops, zero-output daytime outages and bad sensor values into replay copies to check alert behavior. Report detection rate, delay and unmodified-hour alert rate. These are controlled tests, not verified real faults. A residual alert signals investigation; it cannot diagnose equipment failure. If no candidate passes the gate, retain the baseline/champion and show the blocked promotion.

## 04 / System architecture

Local orchestration and AWS inference, linked by versioned artifacts

```text
PV generation + NASA weather -> Airflow ingestion/validation -> DVC snapshot
 -> Kubeflow features/Feast -> train/tune -> evaluate/MLflow -> package/register
 -> CI tests/scans/build -> human approval -> ECR -> AWS Flask API
API + delayed labels -> Prometheus/Grafana/CloudWatch -> Airflow retraining
Local Kubernetes: KFP execution + separate API scaling demonstration
BentoML: additional versioned local serving/container demonstration
```

### A single training trigger

Airflow ingests a bounded source interval, validates it, versions the snapshot, submits a Kubeflow run, waits for completion and records its result. KFP owns feature creation, chronological splits, training/tuning, evaluation, model packaging and candidate registration. Airflow must propagate a failed KFP run as a failed DAG; it must not train a second copy of the model.

### Clear environment boundaries

- Developer workstation: Docker Desktop/WSL2; Compose profiles for API, MLflow with durable backend/artifacts, Airflow and monitoring. Start only needed profiles.
- Local kind/Minikube: standalone Kubeflow Pipelines [5] plus a separate namespace for API scaling demonstrations. Verify resource feasibility early; full Kubeflow is unnecessary.
- AWS: private S3 data/model artifacts, ECR images, an EC2-hosted Dockerized Flask API, IAM instance role and CloudWatch. Use a reverse proxy with HTTPS and authenticated access.
- Feast: Parquet offline data and SQLite online store for a one-process demonstration. Snapshot approved online features for the cloud demo; use a shared online backend for replicas or production. Never assume independent SQLite replicas share state.

Network handoff: containerized Airflow must reach the KFP API through a deliberately configured host/cluster route. KFP pods must reach MLflow and artifact storage. Test these connections with a tiny pipeline in week 1; localhost inside one container is not the host or another container. Historical replay, not live power telemetry, is the core demonstration mode.

## 05 / Member 1 work package

Data and ML Engineering | Primary owner of trustworthy predictions

Assign to: Member 1 (name pending). Review partner: Member 2. This member also reviews Member 3's serving and feature-consistency work.

| Work item | Concrete outputs | Completion check |
| --- | --- | --- |
| M1.1 - Data feasibility | docs/data_card.md; site manifest; bounded PVDAQ and NASA ingestion modules; source/terms notes. | One site has correctly aligned measured energy and weather with defensible units and coverage. |
| M1.2 - Data quality | src/data/validation.py; hourly aggregation; quarantine report; schema and quality config. | Duplicate, timestamp, missingness, invalid-unit and capacity violations are tested; invalid snapshots block training. |
| M1.3 - Features + Feast | Shared feature functions; feature_repo definitions/config; offline retrieval; materialization script. | Point-in-time/availability checks and offline-online parity pass; online query is demonstrated. |
| M1.4 - Modeling | Baselines; Ridge/boosting training; bounded tuner; chronological evaluation; serialized preprocessing + models. | Same-fold comparison and final-test report; no target leakage; required metric gate is executable. |
| M1.5 - Experiment lineage | MLflow runs and registry metadata; plots; data/Git/feature versions; model card. | A selected run resolves to exact data, code, feature schema and artifact hashes. |
| M1.6 - Deviations + feedback | E1 residual calibration; production-deviation logic; labeled-replay evaluation; rolling accuracy job. | Controlled cases are detected; no-alert handling covers night, missing data and delayed labels. |

### Required handoffs

- To Member 2 by end of week 1: validated fixture, data manifest schema, module interfaces and deterministic small-data training entrypoint.
- To Member 3 by week 2: approved model bundle format, ordered feature schema, unit/range rules, Feast entity contract and sample predictions.
- By week 3: machine-readable evaluation.json with baseline/champion comparison, eligible flag, thresholds and failure reasons. Do not send only a notebook or screenshot.
- By week 5: calibrated residual rule, rolling-performance outputs and final model-card limitations.

### Member 1 must demonstrate

Recover a versioned dataset; explain kW versus kWh; show why a feature was available at issue time; retrieve historical Feast features; compare MLflow runs; explain a rejected model and a flagged production deviation.

Primary files: src/data/, src/features/, src/models/, feature_store/, configs/data.yaml, configs/evaluation.yaml, tests/test_data.py, tests/test_features.py, tests/test_model.py. Pair with Member 2 on DVC tracking and Member 3 on inference parity.

## 06 / Member 2 work package

MLOps and Automation Engineering | Primary owner of reproducible execution

Assign to: Member 2 (name pending). Review partner: Member 3. This member also reviews Member 1's data/model changes.

| Work item | Concrete outputs | Completion check |
| --- | --- | --- |
| M2.1 - Collaboration | GitHub repo and branch rules; DagsHub connection; issue board; PR/evidence templates. | All three members make meaningful reviewed commits; main has passing checks. |
| M2.2 - Versioning | DVC remote/config; dvc.yaml and lockfile; data recovery and conflict guide. | Checkout old Git commit + dvc pull recreates matching data and config. No large raw data in Git. |
| M2.3 - Airflow | Ingest/validate/version/submit/wait/decision DAG; retry and timeout policy; replay/retrain parameters. | One trigger starts the complete flow; upstream validation failure prevents KFP submission. |
| M2.4 - Kubeflow | Containerized feature/train/tune/evaluate/package/register components; compiled IR YAML. | Executed graph passes artifacts explicitly and refuses packaging/registration eligibility when gates fail. |
| M2.5 - Tests + CI | pytest wiring; Ruff; dependency/image scan jobs; PR image build; test reports. | Push/PR checks work; deliberately broken change is blocked; no AWS credentials reach untrusted PR jobs. |
| M2.6 - Release + retrain | Gated ECR/deployment workflows; promotion manifest; Airflow retraining route. | Approved candidate deploys automatically after approval; failing candidate cannot deploy; previous digest can be restored. |

### Required handoffs

- To all by day 2: project layout, branch/PR workflow, dependency strategy, fixture location and task board.
- To Member 1: agreed data snapshot URI/hash contract, component I/O types and MLflow metadata keys.
- To Member 3 by week 2: deploy manifest contract, CI image naming, registry locations and required environment-variable names.
- To monitoring owner by week 4: authenticated, deduplicated route for a drift/performance event to trigger the retraining DAG.

### Avoid duplicate automation

DVC defines versioned data dependencies and reproducible stage commands. Airflow schedules and coordinates; KFP executes the ML graph. Reuse the same Python functions/stage entrypoints throughout. A KFP packaging step invokes a build workflow or dedicated unprivileged builder; do not mount a host Docker socket into training pods. Await build completion and record the image digest.

### Member 2 must demonstrate

Show a PR failure; trigger Airflow once; inspect KFP component artifacts; reproduce an older code/data pair; explain safe .dvc conflict resolution by choosing the intended artifact lineage and regenerating metadata; trace a successful and a rejected release.

Primary files: pipelines/, dvc.yaml, .github/workflows/, tests/test_pipeline.py, docs/reproducibility.md, docs/collaboration.md. Each component owner supplies tests; Member 2 integrates the suite.

## 07 / Member 3 work package

Deployment and Reliability Engineering | Primary owner of the running system

Assign to: Member 3 (name pending). Review partner: Member 1. This member also reviews Member 2's automation changes.

| Work item | Concrete outputs | Completion check |
| --- | --- | --- |
| M3.1 - Flask API | GET /health, GET /model-info, POST /predict; shared loader; auth; OpenAPI/examples; Gunicorn. | Valid predictions; malformed/stale requests rejected; model metadata and traceable prediction IDs returned. |
| M3.2 - BentoML | Versioned Bento model/service; preprocessing and feature retrieval; Docker image; smoke test. | Same approved model yields matching Flask/Bento predictions within documented tolerance. |
| M3.3 - Containers | API/ML Dockerfiles; Compose profiles; durable volumes; environment templates. | Another member runs the API/supporting services from clean setup; secrets/raw datasets excluded. |
| M3.4 - Kubernetes | Deployment/Service; readiness/liveness; resource limits; replica and restart scripts. | 1 to 3 replicas work, failed pod is replaced, load balancing and state limits explained. |
| M3.5 - AWS + security | S3/ECR/EC2 setup; IAM roles; HTTPS/auth; CloudWatch; budget alert; cleanup runbook. | Authenticated valid prediction works on documented AWS endpoint; unauthorized access blocked. |
| M3.6 - Observability | Prometheus metrics; Grafana JSON dashboard/alerts; drift/performance exporter; runbooks. | Traffic/latency/errors visible; controlled alert fires and resolves; signal reaches retraining flow. |

### Required handoffs

- To Member 1 by week 1: draft request/response schema and exact feature/metadata needs; avoid different training and serving transformations.
- To Member 2 by week 2: image startup/health contract, registry/deploy settings, smoke test and rollback command.
- To all by week 4: deployment guide, metrics names, alert routing and evidence capture checklist.
- To Member 2 by week 5: versioned alert event with site/model/window/reason; to Member 1: joinable prediction and actual-production records.

### Balance the workload

Member 2 owns CI/CD mechanics and alert-to-Airflow wiring; Member 1 owns drift statistics, delayed-label accuracy and anomaly calibration. Member 3 owns collection, dashboards, hosting and operational alerts. All three pair on final AWS integration; Member 3 is not expected to implement all monitoring/model logic alone.

### Member 3 must demonstrate

Serve an authenticated cloud prediction; show invalid-input handling; compare Flask/Bento outputs; scale and replace pods; display an alert; find the model used for a prediction; roll back an image and confirm its model version.

Primary files: src/serving/, src/monitoring/, docker/, docker-compose.yml, k8s/, monitoring/, infra/, tests/test_api.py, tests/test_serving_parity.py, docs/deployment.md.

## 08 / Shared implementation contracts

Freeze these interfaces early so the three work packages can integrate

### Feast and feature availability

Use site_id as the stable entity. Each feature row represents the site state at an issue time: production lags, past-only rolling values, published weather values and feature-age flags. Use the issue-state timestamp as the Feast event timestamp and retain source_event_time/available_at fields for audit. Historical retrieval must be point-in-time correct and must also enforce publication availability; Feast event-time joins alone cannot prove this [6].

Demonstrate Parquet offline retrieval, online materialization and latest-state querying. Feast online storage holds latest values per entity [6]; historical replay must use offline point-in-time retrieval or materialize state sequentially with a replay clock. Never answer an old issue-time request with today's online features. Validate freshness explicitly, not by assuming offline TTL expires online values.

| Contract | Minimum contents |
| --- | --- |
| Feature contract | site_id, issue_time_utc, target interval, ordered feature names/types/units, schema_version, source timestamps, weather age, production freshness, null handling. |
| Model bundle | Serialized preprocessing + estimator; mode F1/E1; schema; training window; Git/DVC/Feast versions; MLflow run/model version; evaluation/calibration artifacts; hashes. |
| Prediction log | prediction_id, issue/target times, site_id, model_version, schema_version, output_kwh, processing latency and availability mode. No tokens or entire raw payloads. |
| Actual/feedback log | site_id, target interval, measured_kwh, observed/available times, source/quality flags; join to all relevant predictions without overwriting model lineage. |
| Promotion manifest | Approved MLflow model version, image digest, Git commit, data hash, feature schema/snapshot, evaluation result, approver and previous release digest. |

### Flask endpoint behavior

- GET /health: readiness reports whether the pinned approved model is loaded and required dependencies are usable; 503 when unready. Use a separate lightweight liveness check for process health.
- GET /model-info: return model version, run ID, mode, schema, training window and units; never credentials or private storage URLs.
- POST /predict: accept site_id, issue_time_utc and horizon_hours=1 for F1. Fetch approved features using the shared provider; reject unknown sites, unsupported horizons, stale features and ambiguous timestamps.
- Return prediction_id, interval_start/end_utc, predicted_energy_kwh, model_version, schema_version and warnings. Use 400 for invalid schema, 401/403 for auth, 503 for unavailable model/features; bound request size and rate.

Forecast example: issue_time 2026-06-01T10:00:00Z predicts [10:00,11:00) after the previous interval closes. Document this convention and expected meter delay. If the latest production interval has not arrived, apply the explicitly tested lag fallback or reject the request. E1 has a separate internal evaluation interface to prevent mode confusion.

## 09 / Automation, release and cloud

Fail closed on quality, then approve and deploy an immutable release

### Airflow to Kubeflow sequence

DAG: resolve configuration -> ingest/replay -> validate -> persist DVC snapshot -> submit KFP(snapshot URI/hash, Git SHA, feature version) -> poll run -> collect evaluation/build result -> mark eligible or rejected -> request release approval. KFP: feature engineering -> Feast offline dataset/splits -> train/tune -> evaluate/log MLflow -> conditional package/build -> register candidate. A failed evaluation records its report but cannot become an approved model.

- Use idempotent run IDs from source snapshot + config + code; prevent duplicate active site runs. Retry transient I/O twice with backoff; do not retry invalid data as though it were a network failure.
- Persist data references and component outputs, not large payloads in Airflow XCom. Configure timeouts and capture KFP run ID, build ID and failure reason.
- Scheduled ingestion/replay can run daily; scheduled retraining weekly. A drift/performance event follows the same validation/training path with a cooldown and minimum new-label count.

### CI/CD responsibilities

| Stage | Checks and outcome |
| --- | --- |
| Every push / PR | Install pinned compatible dependencies; Ruff; unit/data/feature/model/API/integration/regression tests; build image; dependency and image vulnerability scans. Block required checks on failure. |
| Eligible candidate build | KFP packaging submits controlled build using exact model hash. Build and smoke-test Flask image and Bento demo artifact. Publish immutable ECR digest after checks. |
| Promotion approval | Compare candidate to champion; review validation slices, feature schema and security results. Record an explicit human approval; training never directly replaces production. |
| AWS release | GitHub Actions assumes a scoped IAM role using OIDC [8]; deploy pinned ECR digest via SSM; health/API smoke test; record release manifest. Failed smoke test restores previous digest. |
| Rollback | Restore previous image plus compatible feature/config snapshot; verify /model-info and a known prediction. Record reason and resulting version. |

### Cost-conscious deployment

Choose EC2 for the initial AWS path to keep Docker deployment easy to inspect. Right-size after measuring image memory; do not promise a free tier. Keep KFP training/scaling local. Use S3 private artifacts, ECR immutable image references and bounded CloudWatch retention. Run Prometheus/Grafana in the local demo profile; for cloud metrics, scrape the cloud API through a secured route available for the demonstration.

Before provisioning, set an account budget alert and a team-agreed spending cap. Restrict inbound traffic to the authenticated HTTPS endpoint and required admin access; use SSM instead of public SSH where practical. IAM separates artifact-read runtime permissions from release permissions. Never commit .env, keys or Terraform state. Teardown includes EC2, volumes, addresses, registries/artifacts and any other billable resources; a budget alert does not automatically stop spending.

## 10 / Monitoring and retraining

Observe operations immediately; assess accuracy only after labels arrive

| Signal / owner | Proposed alert or gate | Response and evidence |
| --- | --- | --- |
| API operations / M3 | 5xx ratio >5% for 5 min with >=20 requests; or p95 latency >1 s for 5 min with sufficient traffic. | Prometheus counters/histogram; Grafana alert; inspect logs, dependency health and release version. Controlled load triggers and resolves alert. |
| Health + data quality / M3 + M1 | Service unavailable for 2 min; mandatory schema/unit check fails; live production lags >2 h old. | Separate invalid data from missing telemetry. Quarantine snapshot, reject stale prediction where required, show alert and recovery. |
| Weather freshness / M1 | Weather age exceeds declared mode policy (initial NASA forecast policy: 14 days). | Do not substitute future weather. Use tested missing-weather fallback or return unavailable; show mode and freshness in report. |
| Input drift / M1 | PSI >0.20 in >=2 monitored features across two daily windows, with >=200 valid rows per comparison. | Compare similar daylight/hour/season windows. M3 publishes drift result; M2 deduplicates event and triggers Airflow review/retrain path. |
| Model performance / M1 | Seven-day daylight MAE >1.25 x fixed reference MAE for two daily checks; >=30 valid daylight labels; denominator floor. | Join delayed actuals to model-specific predictions. Distinguish data outage from degradation. Trigger candidate training, not automatic promotion. |
| Production deviation / M1 | Frozen E1 residual bounds breached for two valid daylight hours. | Record actual/expected/residual and model; exclude nights/bad sensors. Alert signals investigation; no hardware diagnosis. |

### Dashboard and metrics contract

Expose a protected /metrics endpoint with request totals by route/status, latency histogram and readiness. Add low-cardinality model-version info, data-validation failures, source freshness, drift flags and rolling labeled MAE. Keep prediction IDs in logs, not Prometheus metric labels. Grafana shows API health, resource utilization, model version, actual versus predicted, drift and data freshness. CloudWatch retains AWS/application logs; MLflow retains experiments.

### Controlled feedback-loop demonstration

Replay holdout hours in order and advance a virtual clock. Store predictions before labels become visible; ingest labels later and calculate accuracy. Introduce a labeled weather-distribution shift or model-performance drop in a replay copy. Confirm alert -> Airflow -> revalidation -> KFP retraining -> candidate/champion comparison -> approval decision. Show one rejected candidate, one approved deployment and a rollback. Never mix injected examples into the final accuracy report.

Set a proposed 24-hour cooldown and one active retraining run per site. Missing labels pause accuracy-based promotion; they do not imply zero error. Automatic triggering and evaluation are required; production promotion remains controlled. Save dashboard JSON, alert rules, firing/resolved evidence, DAG/KFP run IDs, MLflow comparison and release/rollback manifests.

## 11 / Six-week execution schedule

Parallel ownership with weekly integration gates

| Week | Member 1 | Member 2 | Member 3 | Shared exit gate |
| --- | --- | --- | --- | --- |
| 1: feasibility | Audit site, energy units/timezone; sample data and baselines. | Repo/DagsHub/DVC; test skeleton; tiny KFP trigger from Airflow. | Flask mock; Compose; container and local cluster feasibility. | Verified data route; agreed contracts; Airflow can reach KFP. |
| 2: vertical slice | Hourly features, first model, MLflow, Feast offline/online demo. | Real ingest -> KFP preprocess/train/evaluate; basic CI and DVC recovery. | Real model in Flask Docker; auth/errors; Bento smoke test. | One trigger produces a versioned model and local API prediction. |
| 3: controls | Tuning, chronological gates, E1 calibration, parity tests. | All test categories; scans; gate conditions; image build handoff. | K8s probes/replicas; Prometheus/Grafana; AWS/IAM preparation. | Bad candidate blocked; good candidate packaged; pod recovery works. |
| 4: cloud | Freeze model/schema; model card; label-join outputs. | OIDC/ECR/release workflow; approval and rollback mechanics. | AWS API with HTTPS/auth and CloudWatch; cloud metrics route. | Authenticated AWS prediction deployed through CI/CD. |
| 5: feedback | Drift/performance logic and controlled anomaly results. | Scheduled/event retraining; candidate/champion comparisons. | Firing/resolved alerts; integrated monitoring and rollback. | Complete alert -> retrain -> decision -> deploy/rollback replay. |
| 6: submission | Results/data/feature documentation; contribution statement. | Clean reproduction; version recovery; failed/passed CI evidence. | Deployment/monitoring guides; cleanup rehearsal; evidence. | All acceptance criteria checked; every member can run the demo. |

### Start here: first 48 hours

- Together: assign the three roles, confirm deadline/hardware/AWS budget, read the acceptance checklist and turn each work item into a small issue.
- Member 1: download metadata plus a small bounded generation sample; prove kWh conversion and time alignment before a full download. Inspect one matching NASA response and document availability assumptions.
- Member 2: initialize the shared repository/DagsHub integration, reviewed branches, DVC remote and a minimal CI job. Establish a fixture-based test path before large dependencies.
- Member 3: run a Flask health/predict mock in Docker, verify Compose and local Kubernetes capacity, draft API schema and cost/access plan.
- Together: execute a tiny Airflow-to-KFP proof before week 1 ends. If cluster resources are inadequate, borrow institutional capacity or use a time-bounded approved host; document the environment.

### Team rhythm

Hold two short integration sessions each week. Use one reviewer per PR; every task includes code/config, meaningful tests and a reproducible evidence step. Rotate the person running the whole system. If the schedule compresses, remove extra sites, complex models and optional UI first; preserve every mandated tool and acceptance demonstration.

## 12 / Delivery, evidence and risks

Completion means another member can reproduce and explain the workflow

### Repository map

data/{raw,processed}/ (DVC); src/{data,features,models,serving,monitoring}/; feature_store/feature_repo/; pipelines/{airflow/dags,kubeflow/components}/; tests/; configs/; docker/; k8s/; monitoring/; infra/; docs/; evidence/. Root: README.md, REPORT.md, dvc.yaml, dvc.lock, docker-compose.yml, dependency lock/config, .env.example and .github/workflows/. Notebooks are exploratory, never the production entrypoint.

### Final demonstration order

- 1. Explain site, target, F1/E1 distinction and architecture. Recover an earlier Git/DVC snapshot and show data lineage.
- 2. Show a failing PR/quality gate and a passing run. Trigger one Airflow DAG, inspect KFP artifacts and compare MLflow candidates.
- 3. Demonstrate Feast offline retrieval, online materialization/query and a stale-feature rejection. Compare Flask/Bento output.
- 4. Approve/deploy an eligible ECR image through CI/CD; call the authenticated AWS endpoint and show model metadata/errors.
- 5. Scale local Kubernetes from 1 to 3 replicas; delete/fail one pod and show replacement plus healthy traffic.
- 6. Trigger/resolve a Grafana alert; run controlled drift or accuracy feedback through retraining; show candidate decision and rollback.
- 7. Another member follows README from a clean setup. Show implemented secrets/access controls and cleanup steps.

### Project-specific risk register

| Risk | Mitigation / accountable owner |
| --- | --- |
| Forecast leakage from delayed or revised weather | Separate F1/E1; audit available_at; label simulated historical lag; run lag sensitivity and do not claim perfect vintage backtesting. M1. |
| Wrong units, time alignment or site pairing | Inspect source metadata; integrate power correctly; test UTC/DST and interval boundaries; lock site coordinates. M1. |
| Limited or inaccessible measured production | Week-1 data gate; alternate measured source; narrow site/window; disclose synthetic demo data. M1 + all. |
| Tool integration exceeds local hardware | Early Airflow/KFP proof; standalone KFP; Compose profiles; bounded tuning; preserve integration time. M2 + M3. |
| Cloud spend or exposed endpoint/secrets | Budget before deployment; scoped IAM/OIDC; HTTPS/auth; secret/image scans; teardown resources. M3 + M2. |
| Misleading fault alerts or site generalization | Daylight gating; weather-adjusted estimator; labeled injection study; report nuisance alerts; one-site limitation and human investigation. M1 + M3. |

Evidence index: store dated run IDs, logs and screenshots under evidence/{reproducibility,mlflow,feast,airflow,kubeflow,ci,serving,aws,kubernetes,monitoring,retraining,rollback}/. REPORT.md links each course acceptance item to evidence and an owner. Each contribution statement lists authored changes, reviewed PRs, integration work and demonstrated components. Fairness discussion should focus on site/season coverage and unequal reliability across operating conditions; this is not a human classification task.

## 13 / References and open decisions

Source-backed design, with planning assumptions kept explicit

### Controlling project specification

MLOps_Course_Project_Requirements (1).pdf, supplied in this workspace; 13 pages. Requirements and team size: pp. 1-7. Roles/structure/deliverables: pp. 7-9. Rubric and acceptance: p. 10. Scope guidance: p. 11. Feast and BentoML addendum: pp. 12-13. The illustrative architecture is guidance; this plan selects a smaller implementation that preserves the stated deliverables.

### Primary technical sources - checked 11 November 2026

[1] OpenEDI - PVDAQ documentation. Measured PV data route, metadata and site-dependent data conventions.
https://github.com/openEDI/documentation/blob/main/pvdaq.md

[2] NLR Developer Network - Solar APIs. PVDAQ V3 is decommissioned; use the published data-access route.
https://developer.nlr.gov/docs/solar/

[3] NASA POWER - Hourly API. Hourly weather/solar access and explicit UTC time-standard selection.
https://power.larc.nasa.gov/docs/services/api/temporal/hourly/

[4] Registry of Open Data on AWS - NASA POWER. Near-real-time source update lag; actual availability must be measured.
https://registry.opendata.aws/nasa-power/

[5] Kubeflow - Pipelines overview. Containerized workflows, compiled pipeline representation and standalone backend option.
https://www.kubeflow.org/docs/components/pipelines/overview/

[6] Feast - online store and historical retrieval API. Latest-value online semantics, materialization and time-aware historical retrieval.
https://github.com/feast-dev/feast/blob/master/docs/getting-started/components/online-store.md
https://rtd.feast.dev/en/master/feast.html

[7] BentoML - Packaging for deployment. Build a versioned serving artifact and containerize it.
https://docs.bentoml.org/en/latest/get-started/packaging-for-deployment.html

[8] GitHub Docs - Configuring OIDC in AWS. Federated deployment access with scoped trust conditions.
https://docs.github.com/en/actions/how-tos/secure-your-work/security-harden-deployments/oidc-in-aws

### Decisions to finalize before implementation

| Decision | Current plan / resolution point |
| --- | --- |
| Members and deadline | Use Member 1/2/3 until names/preferences arrive; rebase the six-week schedule. |
| Exact measured dataset and dates | PVDAQ/OEDI preferred; approve only after metadata/download/coverage checks in week 1. |
| Infrastructure capacity and budget | Local KFP/Kubernetes; AWS EC2 API. Confirm machine resources and spending cap before provisioning. |
| Dependency compatibility | Pin a mutually tested Python, Airflow, KFP, Feast, MLflow and BentoML stack in week 1. Use separate images/environments rather than forcing all tools into one interpreter. |
| Threshold calibration | The proposed values are starting gates. Freeze any justified changes using training/validation evidence before opening final test results. |
