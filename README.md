# SolarCast: Automated Energy Forecasting

A three-member course project for building a reproducible, automated and monitored solar-energy forecasting system.

**Status:** Scope revision and data-feasibility verification. Member 1 has added a Python scaffold, pinned dependencies and setup smoke tests, and has reported production-data findings on `m1/data-feasibility`. The underlying data files have not been independently validated in this workspace. The HRRR weather-forecast source proposed below is not yet approved. There is no working prediction pipeline or deployment yet.

## Project objective

Forecast the next hour of AC solar-energy production in **kWh** for one selected real solar plant, using **weather forecasts (including irradiance and temperature), time-of-day features and historical production**. Compare actual production with the forecast stored before that hour and flag substantial deviations.

The prototype will demonstrate the complete MLOps lifecycle through **chronological historical replay and a cloud-hosted prediction API**. It does not currently have a live connection to the plant. In replay, the target hour must still be in the future relative to the replay clock.

The project separates these uses of weather:

- **Core next-hour forecast:** Use archived weather forecasts whose required fields were available before the energy prediction was issued, alongside available production history and calendar features. NOAA HRRR is the proposed forecast source, pending the verification gate below.
- **Core deviation alert:** After the measured target-hour production becomes available, compare it with the previously stored energy forecast. Calibrate thresholds on earlier validation data. A deviation may reflect weather-forecast error, model error, bad measurements or plant behavior; it does not establish equipment failure.
- **Supporting retrospective analysis:** Use NASA POWER observed-weather estimates for historical weather analysis and a separately evaluated weather-adjusted expected-output benchmark. This analysis is delayed and must not be presented as a forecast made before the target hour.

**Why the source changed:** NASA POWER's historical weather observations arrive after the measured conditions. We cannot use observations from the target hour to predict that hour in advance. An archived weather forecast is different: it preserves a prediction of those conditions made earlier. Adding a forecast archive addresses the timing mismatch only if its actual release timing and target intervals can be verified.

### Prediction example and information-availability rule

At **14:00 UTC**, predict the plant's energy for **[14:00, 15:00) UTC**. Use an earlier weather-model run forecasting that interval, with the necessary fields already released by 14:00, and only plant readings available by 14:00. Store the predicted kWh and its lineage, then compare with actual kWh after the interval closes and measurements arrive.

```text
forecast_available_at_utc <= prediction_issue_time_utc
weather forecast interval matches the target production interval
production_feature_available_at_utc <= prediction_issue_time_utc
```

The weather model's initialization time is not its release time. Do not assume a run initialized at 14:00 was available at 14:00. Use an older run or reject the case when availability cannot be justified. Also account for meter reporting delay before using the latest production interval.

## Data plan

| Data | Proposed source | Purpose and status |
| --- | --- | --- |
| Measured production | PVDAQ/OEDI System 2107 | Candidate training target and actual-production record. Member 1 reports an Arbuckle, California plant, 893 kW DC, 15-minute AC power readings and roughly 2017-November 2023 coverage. These are preliminary findings requiring source-file verification. |
| Forecast irradiance and temperature | NOAA HRRR archived forecasts | Proposed core weather inputs for next-hour energy forecasting. Exact variables, forecast runs, release timing, site extraction and overlapping dates must pass the verification checklist. |
| Observed-weather estimates | NASA POWER | Supporting retrospective weather analysis and expected-output benchmark for the same site/dates. Not a source of future-known observed weather for the core forecast. |
| Calendar and production history | Derived from validated timestamps and meter readings | Hour/season/solar-position features, lagged production and past-only rolling statistics. |

- **Alignment:** Use the actual plant coordinates, UTC timestamps and explicit interval boundaries. Resolve civil-time daylight-saving transitions and distinguish them from NASA local solar time.
- **Target:** Convert validated power measurements into hourly kWh using the channel's documented sampling/averaging semantics. Missing readings are not zero production.
- **Training-serving consistency:** Train and evaluate the core forecasting model with the same forecast-weather inputs and availability policy intended for serving. Do not train on target-hour NASA observations and silently substitute HRRR forecasts at inference.
- **Versioning:** Track large datasets and generated artifacts with DVC, using a configured remote. Retain source metadata, checksums, extraction configuration and model-run identifiers. Log data, feature-definition and code versions with each MLflow experiment.

**Data acceptance is pending:** A public archive's existence does not prove the required fields, dates and original release times are usable for this plant. Start with a bounded sample, not a full archive download.

### Special note for Member 1: verify HRRR before finalizing the dataset

**Owner: Emad / Member 1 (Data and ML Engineering). Next action: investigate and report findings to the team before full training.** The production dataset remains a candidate; HRRR is a proposed addition, not an already verified integration. Preserve the existing research on `m1/data-feasibility`; bring this scope revision into that branch before continuing so its older NASA-only assumptions are reconciled.

Start with **one consecutive week within the reported production period**, including valid daylight hours. Confirm that the following checks pass, then assess whether sufficient seasonal coverage can be retrieved affordably and consistently.

- [ ] **Production channel and site:** Verify the selected channel is the intended plant-level AC power/energy measurement. Record exact coordinates, source identifiers, units, interval duration, timestamp convention and average-versus-instantaneous semantics. Do not treat the reported DC nameplate rating as verified AC capacity.
- [ ] **Time alignment:** Inspect raw rows around daylight-saving transitions, duplicates and missing intervals; normalize to UTC. A local `01:45 -> 03:00` jump can represent one ordinary 15-minute elapsed interval at the spring transition. Establish whether each source labels interval start, center or end.
- [ ] **HRRR archive access and overlap:** Retrieve actual forecast records covering the sample at the plant's location. Record exact bucket/object URLs, forecast-model version, run initialization, lead time and valid time. Archive-wide coverage is not proof that every needed file exists.
- [ ] **Required weather fields:** Extract forecast irradiance/downward shortwave radiation and near-surface temperature. Record exact variable identifiers, levels, units and whether each value is instantaneous, interval-averaged or accumulated. Demonstrate how those values map to the hourly production target; do not assume radiation is an hourly average.
- [ ] **Forecast release timing:** Establish when the required fields would have been available, not just when the model run was initialized. Cite delivery documentation and any usable contemporaneous records. A current archive object's modification date may reflect copying or reprocessing rather than original publication. If only a latency assumption can be supported, label the backtest as simulated availability, test its sensitivity and report the uncertainty; do not mark historical availability as verified.
- [ ] **No hindsight in run selection:** For each prediction issue time, choose the latest eligible earlier run using availability alone. Do not select a later run, an analysis field or whichever forecast happens to match the actual weather best. Where availability is unresolved, exclude/flag that case rather than silently assume it.
- [ ] **Spatial extraction and missingness:** Document nearest-grid-point/interpolation method, grid coordinates and distance from the plant. Check daytime radiation/temperature plausibility, unavailable files and join coverage. Report missingness by month and daylight period, not only a whole-dataset percentage.
- [ ] **NASA's separate role:** Demonstrate matching observed-weather retrieval for the site/sample, with explicit UTC and verified units. Keep target-hour NASA observations outside core forecast features. Document any separate retrospective benchmark and its publication delay.
- [ ] **Terms and reproducibility:** Record attribution/usage terms for the plant data and weather sources, retrieval steps, checksums and extraction dependencies. Store large files in DVC/artifact storage; commit reusable scripts, configuration and metadata rather than raw archives or credentials.
- [ ] **Feasibility and decision:** Report sample download size/time, extraction effort, matched-hour count, usable historical overlap and remaining blockers. Expand checks across seasons before committing to the full training period. Recommend **GO**, **CONDITIONAL** or **NO-GO** with evidence. Do not claim completion of the weather-driven pipeline until this decision is reviewed.

#### What Member 1 should send back

Submit a short `docs/data_feasibility.md` report and a reproducible extraction/validation script through a pull request. Include a small joined preview or DVC-tracked sample with:

```text
site_id, prediction_issue_time_utc, target_start_utc, target_end_utc,
weather_run_init_utc, weather_available_at_utc, availability_evidence_or_assumption,
weather_valid_time_or_interval, forecast_irradiance_wm2, forecast_temperature_c,
production_feature_cutoff_utc, production_feature_available_at_utc,
actual_energy_kwh, actual_available_at_utc, quality_flags
```

`actual_energy_kwh` is a training/evaluation label; it must not enter forecast features before it becomes available. Preserve additional raw timing and units metadata where the simplified preview is insufficient.

The report must identify each finding as **verified**, **assumed** or **unresolved**, and show the forecast-release/target-time alignment for a few real sample rows. If HRRR fails the gate, the team will reconsider the weather source or production dataset before implementation; it will not silently replace timely weather forecasts with two-week-old observations.

## Planned models and evaluation

- Baselines: previous-hour and same-hour-previous-day persistence.
- Initial candidates: Ridge regression and histogram gradient boosting.
- Chronological validation with an untouched final test period.
- MAE and RMSE, daylight evaluation, capacity-normalized error and seasonal slices.
- Core forecast residuals computed from stored predictions and later actuals, with deviation thresholds calibrated on chronological validation data.
- A separately evaluated retrospective weather-adjusted benchmark using NASA POWER, if the supporting analysis is implemented. Never combine its results with advance-forecast accuracy.
- Compare production/time-only models with weather-enhanced models on identical eligible timestamps to measure the added value of forecast weather. Do not assume a particular improvement before testing.
- Automated evaluation gates that prevent an unsuitable candidate from being deployed.

## Planned MLOps stack

| Area | Tools and responsibility |
| --- | --- |
| Collaboration | GitHub branches, issues and reviewed pull requests |
| Data reproducibility | DVC and DagsHub integration |
| Experiments and model lineage | MLflow tracking and model management |
| Feature store | Feast offline retrieval, materialization and online-query demonstration |
| Scheduling | Airflow ingestion, validation and Kubeflow submission |
| ML workflow | Kubeflow Pipelines containerized feature, training and evaluation stages |
| Quality checks | pytest, linting, dependency/image scanning and GitHub Actions |
| Required serving path | Flask REST API with a production server |
| Additional serving path | Versioned BentoML artifact and container demonstration |
| Packaging and orchestration | Docker, Docker Compose and Kubernetes scaling demonstration |
| Cloud | AWS S3, ECR, EC2, IAM and CloudWatch |
| Monitoring | Prometheus, Grafana, drift checks and delayed-label evaluation |
| Improvement | Airflow retraining, candidate comparison, controlled promotion and rollback |

## Team responsibilities

Add remaining names and GitHub usernames once confirmed.

| Member | Primary responsibilities | Integration partners |
| --- | --- | --- |
| Emad / Member 1 - Data and ML Engineering | Data extraction, forecast-data feasibility, validation, feature engineering, Feast, baselines, training, evaluation and MLflow | Member 2 for versioning/pipeline integration; Member 3 for serving consistency |
| Member 2 - MLOps and Automation | GitHub/DagsHub setup, DVC pipeline, Airflow, Kubeflow, test integration and CI/CD | Members 1 and 3 for component contracts and release automation |
| Member 3 - Deployment and Reliability | Flask/BentoML, Docker/Compose, Kubernetes, AWS, operational metrics and dashboards | Member 1 for model/feature contracts; Member 2 for deployment/retraining wiring |

Every member contributes tests and documentation, reviews another member's work, and learns the complete workflow for the oral defense.

## Project documents

- [Detailed project and team plan](SolarCast_Project_Plan.md)
- [PDF project and team plan](output/pdf/SolarCast_Project_and_Team_Plan.pdf)

**Scope precedence:** This README's revised weather-forecast scope supersedes the older forecast-input and primary deviation-check descriptions in the linked plans. Those plans still describe the original NASA-lag approach; their team responsibilities and infrastructure roadmap remain useful, but the forecast-data contract must follow this README pending Member 1's verification. The six-week schedule is provisional until the actual submission deadline is confirmed.

## Planned repository structure

Some Member 1 scaffold directories already exist; remaining directories will be added during implementation. Empty packages and placeholders are not working prediction modules.

```text
SolarCast/
  data/                 # Raw and processed datasets managed with DVC
  src/
    data/               # Extraction, aggregation and validation
    features/           # Shared feature transformations
    models/             # Training, tuning and evaluation
    serving/            # Flask, BentoML and approved model loading
    monitoring/         # Drift, delayed labels and model performance
  feature_store/        # Feast definitions and materialization
  pipelines/
    airflow/            # Scheduling, ingestion and KFP submission
    kubeflow/           # Containerized ML graph and components
  tests/                # Unit, data, model, API, integration and regression tests
  configs/              # Versioned non-secret configuration
  docker/               # Application and ML container definitions
  k8s/                  # Deployment, service, probes and scaling manifests
  monitoring/           # Prometheus and Grafana configuration
  infra/                # AWS configuration and deployment resources
  docs/                 # Data/model cards, runbooks and architecture
  evidence/             # Execution records for the course demonstration
  .github/workflows/    # CI/CD workflows
  README.md
```

## Getting started

1. Accept the repository collaborator invitation.
2. Clone the shared repository using its actual GitHub URL.
3. Read this README's revised scope and Member 1 verification gate, then use the detailed plan for team/infrastructure coordination and agree on the data/model/API contracts.
4. Create a task branch, for example `feature/data-ingestion`, `feature/ml-pipeline` or `feature/flask-api`.
5. Implement a small change with the relevant tests and documentation, then open a pull request into `main`.

Member 1 dependency installation and smoke-test commands are provided below. DVC remote setup, a configured shared MLflow service and prediction-pipeline run commands remain to be completed. There is currently no runnable prediction application or deployed endpoint.

### First handoff for Member 1

Complete the **Special note for Member 1** checklist above first. Report whether the candidate plant data and HRRR forecasts can support the same location, intervals and information-availability policy. Agree on the hourly schema and eligibility rules before large downloads or full model training. Then implement reusable ingestion/validation, baseline models and forecast-weather models with MLflow logging.

GitHub stores the shared code and configuration. An MLflow tracking service and DVC data remote are separate services that the team still needs to configure; creating this repository does not create them.

### Member 1 setup

Member 1 owns data extraction, validation, features, Feast feature store, modeling, and MLflow tracking.

**Environment setup (PowerShell on Windows):**
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Configuration & Credentials:**
- Copy `.env.example` to `.env` and configure tracking credentials:
  `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`.
- Non-secret configurations reside in `configs/`.

**Verification:**
```powershell
python tests/test_smoke.py
# or with pytest and ruff:
pytest tests/
ruff check .
```

## Collaboration rules

- Use one shared repository with task branches and reviewed pull requests into `main`.
- Keep commits focused and meaningful. Pull current changes before starting new work.
- Each owner supplies tests for their component; Member 2 integrates the checks into CI.
- Record source versions and transformations so other members can reproduce results.
- Commit DVC metadata and pipeline definitions; send large data/model files to the configured artifact storage.
- Keep credentials, local environments, generated experiment stores and raw datasets out of Git commits.
- Include an individual contribution statement and reproducible evidence in the final submission.

## Planned API

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Service/model readiness |
| `GET /model-info` | Approved model version and relevant metadata |
| `POST /predict` | Next-hour energy in kWh using available forecast weather, time features and production history; return issue/target timestamps, model version and weather-run lineage |

Request/response examples will be added with the actual API implementation.

## References

- [NASA POWER hourly API](https://power.larc.nasa.gov/docs/services/api/temporal/hourly/)
- [PVDAQ/OEDI data documentation](https://github.com/openEDI/documentation/blob/main/pvdaq.md)
- [NOAA HRRR public archive](https://registry.opendata.aws/noaa-hrrr-pds/)
- [NOAA HRRR model and archive documentation](https://rapidrefresh.noaa.gov/hrrr/)
- [Example NOAA HRRR surface-product inventory](https://www.nco.ncep.noaa.gov/pmb/products/hrrr/hrrr.t00z.wrfsfcf02.grib2.shtml)
- [NOAA HRRR product-delivery notice (historical; verify applicability to selected runs)](https://www.weather.gov/media/notification/tins/tin14-32hrrr_aaa.pdf)

Dataset terms and attribution will be documented when the specific dataset is selected. A project software license has not yet been selected.
