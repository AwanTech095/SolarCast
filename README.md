# SolarCast: Automated Energy Forecasting

A three-member course project for building a reproducible, automated and monitored solar-energy forecasting system.

**Status:** Member 1 built and tested the local historical replay and forecasting pipeline for PVDAQ System 2107. The implementation uses archived HRRR temperature and cloud-cover forecasts with an assumed two-hour availability delay. Cloud deployment remains pending.

## Project objective

Forecast the next hour of AC solar-energy production in **kWh** at PVDAQ System 2107 using **forecast temperature, derived irradiance, time-of-day features and historical production**. Compare actual production with the forecast stored at issue time and flag substantial deviations.

The prototype will demonstrate the complete MLOps lifecycle through **chronological historical replay and a cloud-hosted prediction API**. It does not currently have a live connection to the plant. In replay, the target hour must still be in the future relative to the replay clock.

The project separates these uses of weather:

- **Core next-hour forecast:** Use archived HRRR temperature and cloud-cover forecasts, cloud-adjusted pvlib clear-sky irradiance, production history and calendar features. Weather eligibility follows the replay policy below.
- **Core deviation alert:** After the measured target-hour production becomes available, compare it with the previously stored energy forecast. Calibrate thresholds on earlier validation data. A deviation may reflect weather-forecast error, model error, bad measurements or plant behavior; it does not establish equipment failure.
- **Supporting retrospective analysis:** Use NASA POWER observed-weather estimates for historical weather analysis and a separately evaluated weather-adjusted expected-output benchmark. This analysis is delayed and must not be presented as a forecast made before the target hour.

**Why the source changed:** NASA POWER's historical weather observations arrive after the measured conditions. We cannot use observations from the target hour to predict that hour in advance. An archived weather forecast is different: it preserves a prediction of those conditions made earlier. Adding a forecast archive addresses the timing mismatch only if its actual release timing and target intervals can be verified.

### Prediction example and information-availability rule

At **14:00 UTC**, predict the plant's energy for **[14:00, 15:00) UTC**. Use an earlier weather-model run forecasting that interval, with the necessary fields eligible by 14:00 under the replay policy, and only plant readings available by 14:00. Store the predicted kWh and its lineage, then compare with actual kWh after the interval closes and measurements arrive.

```text
weather_available_at_utc = weather_run_init_utc + 2 hours (replay assumption)
weather_available_at_utc <= prediction_issue_time_utc
weather forecast interval matches the target production interval
production_feature_available_at_utc <= prediction_issue_time_utc
```

For the 14:00 example, the nominal 12:00 UTC run supplies forecast leads +2 and +3, valid at 14:00 and 15:00. The extractor averages the endpoint values for the hour. It can fall back to earlier eligible runs and flags missing weather when none can supply the inputs. The recorded availability time is simulated. Archive coverage establishes retrievability at the time of extraction. Original per-run release times and sensitivity to the assumed delay remain unverified. An operational feed would require recording when the required fields actually arrive and using them only when they arrive by the issue time.

## Data plan

| Data | Proposed source | Purpose and status |
| --- | --- | --- |
| Measured production | PVDAQ/OEDI System 2107 | Measured AC power at the Arbuckle, California site, aggregated from 15-minute kW readings to hourly kWh. 893 kW is the DC nameplate rating. |
| Forecast temperature, cloud cover and derived irradiance | HRRR MesoWest Zarr and pvlib | Forecast TMP and TCDC supply temperature and cloud cover. Cloud cover adjusts pvlib clear-sky GHI to produce an irradiance proxy. Direct forecast DSWRF was unavailable on the selected Zarr route. |
| Observed-weather estimates | NASA POWER | Supporting retrospective weather analysis and expected-output benchmark for the same site/dates. Not a source of future-known observed weather for the core forecast. |
| Calendar and production history | Derived from validated timestamps and meter readings | Hour/season/solar-position features, lagged production and past-only rolling statistics. |

- **Alignment:** Use the actual plant coordinates, UTC timestamps and explicit interval boundaries. Resolve civil-time daylight-saving transitions and distinguish them from NASA local solar time.
- **Target:** Convert validated power measurements into hourly kWh using the channel's documented sampling/averaging semantics. Missing readings are not zero production.
- **Training-serving consistency:** Train and evaluate the core forecasting model with the same forecast-weather inputs and availability policy intended for serving. Do not train on target-hour NASA observations and silently substitute HRRR forecasts at inference.
- **Versioning:** Track large datasets and generated artifacts with DVC, using a configured remote. Retain source metadata, checksums, extraction configuration and model-run identifiers. Log data, feature-definition and code versions with each MLflow experiment.

### HRRR feasibility and open checks

Member 1 checked overlapping archive coverage and site extraction and implemented the TMP/TCDC route for replay. The feasibility report records the field inventory, coverage and route decision. Original field delivery times and replay sensitivity to other delays are still open. [HRRR feasibility report](docs/data_feasibility.md)

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

See the Member 1 handoff for the implementation, validation evidence and integration work. [Member 1 handoff](docs/HANDOFF.md)

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
