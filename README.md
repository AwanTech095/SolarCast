# SolarCast: Automated Energy Forecasting

A three-member course project for building a reproducible, automated and monitored solar-energy forecasting system.

**Status:** Planning and repository setup. Application code, datasets, experiments, infrastructure and deployment are not implemented yet. The components below describe the agreed implementation scope.

## Project objective

Forecast the next hour of AC solar-energy production in **kWh** for one selected solar site, and identify substantial differences between measured production and expected output.

The project separates two tasks:

- **Next-hour forecasting:** Use historical production, calendar features and weather information available when the forecast is issued.
- **Weather-adjusted deviation detection:** Use observed weather after it becomes available to estimate expected production and flag unusual residuals. This is a delayed analysis rather than a live forecast.

NASA POWER provides historical weather with a publication delay. Future weather observations must not enter a next-hour forecast as if they were known in advance.

## Data plan

- **Measured generation:** Select a public PVDAQ/OEDI solar site after verifying access, permitted use, coordinates, capacity, units and historical coverage.
- **Weather:** Retrieve matching hourly solar and meteorological data from NASA POWER for the same site and dates.
- **Alignment:** Use UTC, document interval boundaries, and convert power measurements to interval energy where necessary. Missing readings are not zero production.
- **Versioning:** Track large datasets and generated artifacts with DVC, using a configured remote. Keep source metadata, checksums and processing configuration in version control.

The exact site, date range and usable features remain to be confirmed through a small data-feasibility study.

## Planned models and evaluation

- Baselines: previous-hour and same-hour-previous-day persistence.
- Initial candidates: Ridge regression and histogram gradient boosting.
- Chronological validation with an untouched final test period.
- MAE and RMSE, daylight evaluation, capacity-normalized error and seasonal slices.
- A separate weather-adjusted estimator and calibrated residual thresholds for deviation detection.
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

Replace member labels with names and GitHub usernames once assigned.

| Member | Primary responsibilities | Integration partners |
| --- | --- | --- |
| Member 1 - Data and ML Engineering | Data extraction, validation, feature engineering, Feast, baselines, training, evaluation and MLflow | Member 2 for versioning/pipeline integration; Member 3 for serving consistency |
| Member 2 - MLOps and Automation | GitHub/DagsHub setup, DVC pipeline, Airflow, Kubeflow, test integration and CI/CD | Members 1 and 3 for component contracts and release automation |
| Member 3 - Deployment and Reliability | Flask/BentoML, Docker/Compose, Kubernetes, AWS, operational metrics and dashboards | Member 1 for model/feature contracts; Member 2 for deployment/retraining wiring |

Every member contributes tests and documentation, reviews another member's work, and learns the complete workflow for the oral defense.

## Project documents

- [Detailed project and team plan](SolarCast_Project_Plan.md)
- [PDF project and team plan](output/pdf/SolarCast_Project_and_Team_Plan.pdf)

The six-week schedule in these documents is provisional until the actual submission deadline is confirmed.

## Planned repository structure

The application directories below will be created as implementation begins; they are not working modules yet.

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
3. Read the detailed project plan and agree on the data/model/API contracts.
4. Create a task branch, for example `feature/data-ingestion`, `feature/ml-pipeline` or `feature/flask-api`.
5. Implement a small change with the relevant tests and documentation, then open a pull request into `main`.

Dependency installation, DVC remote configuration, MLflow tracking setup and application run commands will be added when those components exist. There is currently no runnable application or deployed endpoint.

### First handoff for Member 1

Start with a bounded generation sample and matching site metadata. Document units, timestamps, missing values, site coordinates and data terms. Agree on a canonical hourly schema before downloading large datasets. Then add reusable ingestion/validation code and baseline training with MLflow logging.

GitHub stores the shared code and configuration. An MLflow tracking service and DVC data remote are separate services that the team still needs to configure; creating this repository does not create them.

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
| `POST /predict` | Validated next-hour energy prediction |

Request/response examples will be added with the actual API implementation.

## References

- [NASA POWER hourly API](https://power.larc.nasa.gov/docs/services/api/temporal/hourly/)
- [PVDAQ/OEDI data documentation](https://github.com/openEDI/documentation/blob/main/pvdaq.md)

Dataset terms and attribution will be documented when the specific dataset is selected. A project software license has not yet been selected.
