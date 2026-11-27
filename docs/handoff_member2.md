# SolarCast: Member 2 Handoff Specification (Data, Training & CI/CD Orchestration)

**Author:** Member 1 (Data, Features, Feast, Models, MLflow)  
**Target Audience:** Member 2 (DVC, CI/CD, Airflow/KFP Pipelines, Remote Infrastructure)  
**Schema Versions:** Data Manifest `1.0`, Feature Schema `1.1.0`, Evaluation Config `1.0.0`, Alert Event Schema `1.0.0`  
**Repository Branch:** `m1/handoff-docs`

---

## 1. Ownership & Interface Boundaries

| Component | Owner | Deliverables / Contract Files | Member 2 Responsibility |
|---|:---:|---|---|
| **Data Ingestion & Validation** | Member 1 | `src/data/ingestion.py`, `src/data/aggregation.py`, `src/data/validation.py`, `configs/data.yaml`, `configs/data_manifest.json` | Automate execution, configure DVC tracking for `data/raw/` and `data/processed/`. |
| **HRRR Forecast Weather** | Member 1 | `scripts/hrrr_extract.py`, `data/processed/hrrr_forecast_features.parquet` | Distribute pre-extracted weather dataset via DVC/Drive; **do not re-extract in CI**. |
| **Feature Engineering & Feast** | Member 1 | `src/features/preprocessing.py`, `src/features/build_feature_table.py`, `feature_store/`, `src/features/feast_provider.py` | Isolate Feast in `.venv-feast`; materialize SQLite store snapshot for offline/serving. |
| **Model Training & Gating** | Member 1 | `src/models/tune.py`, `src/models/train.py`, `src/models/evaluate.py`, `configs/evaluation.yaml`, `configs/model_params.yaml` | Build Airflow/KFP retraining DAG; wire DagsHub/remote MLflow tracking server. |
| **Deviation Engine** | Member 1 | `src/models/deviation.py`, `configs/deviation.yaml`, `configs/alert_event_schema.json` | Implement alert event consumer webhook/trigger for automated retraining DAG. |
| **CI/CD & Remote Infra** | Member 2 | GitHub Actions workflows, DagsHub DVC remote, Docker images, Airflow/KFP | Configure runners with dual venv (`.venv` and `.venv-feast`), execute `pytest -m "not network"`. |

---

## 2. Execution Runbook: Exact Commands, CLI Flags & Arguments

All commands below are tested and verified on Windows PowerShell. Paths use forward slashes or `pathlib.Path`.

### 2.1 Step 1: Raw Data Ingestion & Manifest Registration
Fetches PVDAQ 15-minute active power meter telemetry for Site 2107 from the NREL OEDI AWS S3 data lake (`s3://oedi-data-lake/pvdaq/2023-solar-data-prize/2107_OEDI/`) and retrieves NASA POWER hourly weather observations (2017–2023) via REST API with exponential backoff.

```powershell
# Ingestion entrypoint (main .venv)
.\.venv\Scripts\python -m src.data.ingestion --config configs/data.yaml
```
- **CLI Flags**:
  - `--config <path>`: Path to data configuration YAML (default: `configs/data.yaml`).
- **Exit Codes & Failure Behavior**:
  - `0`: Ingestion completed, SHA-256 hashes verified, `configs/data_manifest.json` written.
  - `1`: Network timeout or S3 access error (retries up to 3 times before failing).
- **Idempotency**: Fully idempotent. Existing files are checked against expected byte sizes and hashes; redundant downloads are skipped.

---

### 2.2 Step 2: Data Validation, Quarantine & Aggregation
Validates raw 15-minute generation readings using Pandera schemas, isolates invalid rows into a quarantine dataset, resolves Daylight Saving Time (DST) transitions, and aggregates to continuous hourly and quarter-hourly datasets with pvlib astronomical solar position angles.

```powershell
# Aggregation entrypoint (main .venv)
.\.venv\Scripts\python -m src.data.aggregation --config configs/data.yaml
```
- **CLI Flags**:
  - `--config <path>`: Path to data configuration YAML (default: `configs/data.yaml`).
- **Exit Codes & Failure Behavior**:
  - `0`: 240,062 raw 15-minute rows validated. Out-of-bounds readings (>893 kW DC capacity, negatives, non-finite) quarantined to `data/processed/quarantine.parquet`. Outputs `data/processed/hourly.parquet` (60,096 rows) and `data/processed/quarter_hourly.parquet` (240,384 rows).
  - `1` / `SchemaError`: Raw CSV schema mismatch or missing mandatory columns (`Date-Time`, `active_power`).
- **Idempotency**: Deterministic and idempotent. Overwrites processed files identically given the same raw files.

---

### 2.3 Step 3: NOAA HRRR Forecast Weather Extraction (Heavy NWP Pipeline)

The replay field `weather_available_at_utc` equals run initialization plus an assumed two-hour delay. It represents modeled eligibility. Track the packet as historical replay data. An operational pipeline must capture actual required-field arrival times. The feature builder derives irradiance by adjusting pvlib clear-sky GHI with forecast TCDC.
The supplied packet uses 2-meter air temperature (`TMP`) and total cloud cover (`TCDC`) from the MesoWest NOAA HRRR Zarr archive (`s3://hrrrzarr/`) on AWS `us-west-1` for all daylight hours between 2018-07-12 and 2023-11-09.

```powershell
# Merge pre-extracted monthly parts into final dataset (Standard Fast Path for CI/CD)
.\.venv\Scripts\python scripts/hrrr_extract.py --merge --parts-dir data/processed/hrrr_parts --merged-file data/processed/hrrr_forecast_features.parquet

# Full extraction (Heavy; only run when refreshing NWP weather archive)
.\.venv\Scripts\python scripts/hrrr_extract.py --config configs/data.yaml --hourly-file data/processed/hourly.parquet --scan-file data/processed/hrrr_coverage_scan.parquet --parts-dir data/processed/hrrr_parts --merged-file data/processed/hrrr_forecast_features.parquet --variables tmp,tcdc --threads 16 --timeout 30 --max-fallback 6 --merge-after
```
- **CLI Flags**:
  - `--config <path>`: Configuration YAML (default: `configs/data.yaml`).
  - `--hourly-file <path>`: Input hourly target dataset (default: `data/processed/hourly.parquet`).
  - `--scan-file <path>`: Pre-computed coverage scan index (default: `data/processed/hrrr_coverage_scan.parquet`).
  - `--parts-dir <path>`: Directory containing monthly parquet part files (default: `data/processed/hrrr_parts/`).
  - `--merged-file <path>`: Final consolidated parquet output (default: `data/processed/hrrr_forecast_features.parquet`).
  - `--log-file <path>`: Extraction log destination (default: `logs/hrrr_extract.log`).
  - `--start` / `--start-date <str>`: Filter start UTC timestamp (e.g. `2022-10-17`).
  - `--end` / `--end-date <str>`: Filter end UTC timestamp (e.g. `2023-11-09`).
  - `--variables <str>`: Comma-separated variables (default: `tmp,tcdc`; options: `tmp,tcdc,lcdc,mcdc,hcdc`).
  - `--threads <int>`: Worker thread concurrency (default: `16`).
  - `--timeout <int>`: HTTP range-request timeout in seconds (default: `30`).
  - `--max-fallback <int>`: Maximum search lookback in hours for missing runs (default: `6`).
  - `--merge`: Fast merge existing monthly parts and exit.
  - `--merge-after`: Automatically merge monthly parts upon extraction completion.
- **CRITICAL NOTE FOR MEMBER 2**: HRRR extraction requires fetching thousands of spatial chunks across 5.3 years of hourly forecast cycles (~34.8 GB/yr raw GRIB/Zarr chunk volume). **Do NOT execute raw extraction in CI or retraining DAGs.** Member 2 must track and distribute the consolidated file `data/processed/hrrr_forecast_features.parquet` (904 KB) via DVC or Google Drive.
- **Idempotency**: Resumable and idempotent. Monthly parts already present in `--parts-dir` are skipped.

---

### 2.4 Step 4: Feast Feature Table Build
Builds the shared Feature Set B table (`data/processed/feast/features_set_b.parquet`, 23,532 daylight issue hours) containing all 29 model features plus 3 audit non-leakage timestamp columns.

```powershell
# Build feature table (can run in either .venv or .venv-feast)
.\.venv\Scripts\python -m src.features.build_feature_table --hourly data/processed/hourly.parquet --quarter-hourly data/processed/quarter_hourly.parquet --weather data/processed/hrrr_forecast_features.parquet --output data/processed/feast/features_set_b.parquet --start "2018-07-12 00:00:00+00:00" --end "2023-11-09 23:00:00+00:00"
```
- **CLI Flags**:
  - `--hourly <path>`: Path to `hourly.parquet`.
  - `--quarter-hourly <path>`: Path to `quarter_hourly.parquet`.
  - `--weather <path>`: Path to `hrrr_forecast_features.parquet`.
  - `--output <path>`: Output path (default: `data/processed/feast/features_set_b.parquet`).
  - `--start <iso>`: Start UTC timestamp (default: `2018-07-12 00:00:00+00:00`).
  - `--end <iso>`: End UTC timestamp (default: `2023-11-09 23:00:00+00:00`).
- **Exit Codes & Failure Behavior**:
  - `0`: 23,532 rows generated; 100% verified that `production_cutoff_utc <= event_timestamp` and `weather_available_at_utc <= event_timestamp`.
  - `AssertionError`: Non-leakage invariant violated (aborts without saving).
- **Idempotency**: Pure deterministic function of the inputs.

---

### 2.5 Step 5: Feast Registry Apply & Online Store Materialization
Registers entities and feature views in the Feast local registry and materializes feature values into the SQLite online store. Must be executed in `.venv-feast`.

```powershell
# 1. Apply feature view schema to Feast registry
cd feature_store/feature_repo
..\..\.venv-feast\Scripts\feast apply
cd ../..

# 2. Materialize offline features into SQLite online store with MLflow logging
.\.venv-feast\Scripts\python feature_store/materialize.py --start 2023-06-01T00:00:00+00:00 --end 2023-06-15T15:00:00+00:00 --repo-path feature_store/feature_repo --log-mlflow
```
- **CLI Flags for `materialize.py`**:
  - `--start <iso>`: Start UTC timestamp of materialization window.
  - `--end <iso>`: End UTC timestamp of materialization window.
  - `--repo-path <path>`: Path to feature repository directory containing `feature_store.yaml` (default: `feature_store/feature_repo`).
  - `--views [<name> ...]`: Optional list of feature views to materialize (default: all views).
  - `--log-mlflow`: Log materialization duration, view counts, and Git commit to MLflow.
- **Exit Codes**:
  - `0`: Features loaded into `feature_store/feature_repo/data/online_store.db`.
  - `1`: Missing `feature_store.yaml` or unparseable datetime.

---

### 2.6 Step 6: Model Training, Cross-Validation Benchmarking & Bundling
Tunes Ridge and HistGradientBoosting over 3 expanding development folds, selects Set A or B, then fits
a frozen candidate before development Fold 3. Training labels must have arrived before validation starts.
The saved bundle is reloaded and evaluated on that later development fold. Selection used these folds:
this is development validation, not a new independent final test. The supplied historical full-dev champion
is preserved. A new full-dev refit would require its own later validation and cannot reuse this gate.

```powershell
# Model training pipeline (main .venv)
.\.venv\Scripts\python -m src.models.train --eval-config configs/evaluation.yaml --params-config configs/model_params.yaml --hourly data/processed/hourly.parquet --quarter-hourly data/processed/quarter_hourly.parquet --weather data/processed/hrrr_forecast_features.parquet --reports-dir reports --models-dir models
```
- **CLI Flags**:
  - `--eval-config <path>`: Evaluation configuration YAML (default: `configs/evaluation.yaml`).
  - `--params-config <path>`: Hyperparameter ranges YAML (default: `configs/model_params.yaml`).
  - `--hourly <path>`: Hourly generation dataset (default: `data/processed/hourly.parquet`).
  - `--quarter-hourly <path>`: 15-minute generation telemetry (default: `data/processed/quarter_hourly.parquet`).
  - `--weather <path>`: Weather forecast features (default: `data/processed/hrrr_forecast_features.parquet`).
  - `--reports-dir <path>`: Output directory for reports and plots (default: `reports/`).
  - `--models-dir <path>`: Output directory for joblib model bundle (default: `models/`).
- **Exit Codes & Failure Behavior**:
  - `0`: Both cross-validation and saved-candidate gates passed. This does not approve deployment.
  - `2`: Candidate rejected. Archive/log it, stop the release path; do not retry until accepted by chance.
  - Other nonzero: Exception such as missing input, invalid evidence or MLflow registration failure.
- **Outputs**: Each run prints `models/candidates/<candidate-id>/manifest.json`; the directory also contains
  `bundle.joblib`, `validation.parquet` and `evaluation.json`. Plots go to
  `reports/candidates/<candidate-id>/plots/`. Every run has a fresh ID. Root historical evaluation/audit
  files and `models/solarcast_f1_bundle.joblib` are never overwritten. Keep candidate artifacts outside Git.
- **Registry**: Only eligible candidates are registered. No champion alias is moved. The distributable
  serving artifact is the hash-bound bundle (with feature ordering, clipping and nighttime behavior), not
  the bare sklearn estimator. Promotion is not implemented; `deployment_approved` remains false.

---

### 2.7 Step 7: Explicit Development Calibration / Bounded Monitoring

The bare deviation command now fails with usage error; it cannot recalibrate or open a test set by default.
Choose exactly one mode. Neither mode opens the protected historical final-test interval or replaces
`configs/deviation.yaml`. The old combined benchmark and natural-case readers are retired.

```powershell
# Development-only calibration. Supply the actual bundle you intend to assess.
.\.venv\Scripts\python -m src.models.deviation --calibrate --bundle models/solarcast_f1_bundle.joblib --eval-config configs/evaluation.yaml --hourly data/processed/hourly.parquet --features data/processed/feast/features_set_b.parquet --reports-dir reports/deviation_runs --log-mlflow
```

Calibration clones that bundle's estimator recipe across 3 expanding folds in the earlier 80%, purging
training labels unavailable at validation start. These are OOF **recipe** predictions, not the final
bundle's saved-weight predictions. Every daylight source hour must match exactly, with source availability
audits <= issue time. No hard-coded champion parameters are substituted. The output records bundle/data/
feature/config hashes, source MLflow run ID, Git commit, fold windows and protected test boundaries.
It records actual bin counts; unobserved or degenerate bins cannot be treated as calibrated evidence.

Each run prints a fresh directory containing `deviation.yaml`, `validation_residuals.parquet` and
`summary.json`. Keep it outside Git (default directory is ignored). These thresholds are unapproved
experiment artifacts; promotion must bind the model and its matching calibration together.

```powershell
# TEMPLATE: requires joined stored forecasts/actuals from Member 3, not supplied raw meter data.
# Dates illustrate post-holdout monitoring; these observations are not in the historical packet.
.\.venv\Scripts\python -m src.models.deviation --evaluate --bundle models/solarcast_f1_bundle.joblib --deviation-config "reports/deviation_runs/<calibration-id>/deviation.yaml" --records "<joined-records.parquet>" --start-utc "2023-11-10T15:00:00Z" --end-utc "2023-11-10T17:00:00Z" --as-of-utc "2023-11-10T18:00:00Z" --reports-dir reports/deviation_runs --log-mlflow
```

Monitoring reads only the supplied log, never loads raw targets, fits a model or recalibrates thresholds.
Required fields: `site_id`, `timestamp`, `issue_time_utc`, `actual_available_at_utc`,
`model_bundle_sha256`, `actual`, `forecast`, `quality_flag`, `is_daylight`, `solar_elevation_deg`,
`forecast_cloud_bin`. Window semantics: `[start, end)`, `start < end <= as-of`; UTC offsets required.
The window must be after the model's last training label could arrive and must not overlap the historical
final-test interval. Only actuals available by the virtual clock are visible. Missing labels are excluded,
never replaced by zero; zero visible labels yield explicit `no_visible_labels` status, not healthy accuracy.
Both record site/model identity and threshold model hash must agree. Old unbound threshold configs fail
with a clear error; do not append an identity tag to make an unrelated calibration look compatible.

Outputs are a fresh `summary.json` with context, record/threshold/model hashes and confirmation alerts.
This is an independent bounded replay, not a durable monitor or a release-quality score; overlapping
repeated deliveries need deduplication. `--log-mlflow` logs the run/artifacts using `MLFLOW_TRACKING_URI`;
tracking errors propagate. Exit 0 = workflow completed, 2 = usage error, other nonzero = failed evidence/I/O.

For a runnable self-contained demonstration without new plant data or reopening the final test:

```powershell
.\tmp\review_env\Scripts\python -m pytest tests/test_deviation_workflow.py -q
```

---

### 2.8 Step 8: Verify the Saved Candidate (Automation Gate)
Use the manifest printed by training. The checker verifies model/validation/evaluation hashes and chronology,
loads the saved bundle, predicts without refitting, and checks the recomputed result against the saved gate.
Both CV and exact-candidate gates must pass. Changed or missing evidence fails closed. The manifest records
code/config/data/feature metadata and MLflow run identity; it is not a signed provenance certificate.

```powershell
# Replace <candidate-id> with the actual ID from this training run (main .venv).
.\.venv\Scripts\python -m src.models.evaluate --candidate-manifest "models/candidates/<candidate-id>/manifest.json"
```
- **CLI Flags**:
  - `--candidate-manifest <path>`: Required for this saved-artifact gate. No final-test access or refit.
- **Exit Codes**:
  - `0`: Eligible saved candidate. `2`: Rejected candidate. Exceptions fail nonzero.

The legacy `--dry-run` remains a separate research benchmark that refits fixed-parameter models. Its exit
status and metrics cannot approve a saved candidate; do not use it as the retraining/release gate.

---

## 3. Data Snapshot for DVC Tracking

Member 2 must version the following immutable raw and processed files using DVC. Checksums and sizes have been verified directly from `configs/data_manifest.json` and local disk inspection:

### 3.1 Raw Datasets (`data/raw/`)
| Relative File Path | Size (Bytes) | SHA-256 Hash | Provenance / Source |
|---|:---:|---|---|
| `data/raw/2107_meter_15m_data.csv` | 6,319,525 | `7f6c7cf80aae5dd8dd3687a7df95e35d03ffdc5bc0848256a1d76147bce9eb47` | NREL OEDI S3 (DOE Solar Data Prize 2023, DOI 10.25984/1846021) |
| `data/raw/metadata/2107_system_metadata.json` | 64,652 | `17e4f7f448fcb2bec43f7610b40c1f7f4fc794ccc185e41dcde0514afba73ae5` | NREL OEDI S3 Site Metadata |
| `data/raw/weather/nasa_power_2017.json` | 1,032,811 | `3965ae165ce6ef88e4a2922791a748df19aa466e22f6e102ecb94575c66481d0` | NASA POWER Hourly Point API (UTC) |
| `data/raw/weather/nasa_power_2018.json` | 1,032,624 | `782e50f9069e9af0a4598828f91a52565361c08552902a9b23469b28a9912628` | NASA POWER Hourly Point API (UTC) |
| `data/raw/weather/nasa_power_2019.json` | 1,032,660 | `b9d9996a25797f0b37f8055ce352b63397995cf3ed0657611cfdaaf4539e6cc3` | NASA POWER Hourly Point API (UTC) |
| `data/raw/weather/nasa_power_2020.json` | 1,035,552 | `37ff0394b3d8293853a9d2d5d05e6a7b31a1adfb37c28a630bbfe82ed9ab1c46` | NASA POWER Hourly Point API (UTC) |
| `data/raw/weather/nasa_power_2021.json` | 1,032,388 | `9c2a7e6163cbfda0b1b7957badfec3cb7665f2c9bd423f935ab88e085ce46398` | NASA POWER Hourly Point API (UTC) |
| `data/raw/weather/nasa_power_2022.json` | 1,032,509 | `03daedaacd7f09e0cf85379ded642ab92370237c8ea898b0fe363c30e957d50d` | NASA POWER Hourly Point API (UTC) |
| `data/raw/weather/nasa_power_2023.json` | 886,189 | `f3705c931c03b8d1d9a6ff68c7676da5e7f685f96029a41dbcb0608dafaba866` | NASA POWER Hourly Point API (UTC) |

### 3.2 Processed Datasets (`data/processed/`)
| Relative File Path | Size (Bytes) | SHA-256 Hash | Description |
|---|:---:|---|---|
| `data/processed/quarantine.parquet` | 2,577 | `2defa868865bdccae0ecf6fa309475b0f7c977d0b5e9c8d58ffe68124e9dbb51` | Quarantined out-of-range power readings (>893 kW) |
| `data/processed/hourly.parquet` | 1,816,935 | `f9157a5dd5c4378823880c43da6e586c046d09debab99d99be20cf9041105cb6` | 60,096 continuous UTC hours with target kWh, elevation, flags |
| `data/processed/quarter_hourly.parquet` | 6,519,094 | `7c488a111f2eb7b5502f33d3c377bd5f4414b9e06753171efac7771f8c6cb57a` | 240,384 continuous 15-min intervals with power kW |
| `data/processed/hrrr_coverage_scan.parquet` | 434,385 | `a0c10ffc703a94b17bbd1dc8f46844255c223d94ac9d1486359327fb8ccbc68e` | Index of 46,728 HRRR forecast runs availability |
| `data/processed/hrrr_forecast_features.parquet` | 904,412 | `fbc02d8bbbbdb9de818fbdd607cf9e54d7c7177e69d4ebb6d04a91e66360024f` | 23,532 daylight issue hours of HRRR TMP & TCDC |
| `data/processed/feast/features_set_b.parquet` | 2,839,440 | `07fbb11feca94c4b9aba5c5aa21f1f8af922cf3333fbfffb7f9680abf1c8d8f6` | 23,532 daylight issue hours for Feast offline store (29 feats + audit) |
| `data/processed/validation_residuals.parquet` | 1,009,993 | `b5a0d1dab5b4a4c9d88f5fce0695d3695cb2184ad7aa57587c711d3bb02a9b9d` | 27,816 out-of-fold validation residuals for deviation calibration |

---

## 4. MLflow Experiment Architecture & Lineage

The pipeline logs to two dedicated MLflow experiments:

### 4.1 Experiment 1: `solarcast-model-training`
Tracks model cross-validation, candidate benchmarking, acceptance gating, and official final evaluation.

- **Run 1: `HistGradientBoosting_Set_B_15mDelay` (Champion Candidate)**:
  - **Tags**: `git_commit`, `data_hash_hourly`, `data_hash_qh`, `data_hash_hrrr`, `feature_schema_version`, `status` (`candidate`), `is_best_candidate` (`True`).
  - **Params**: `model_type` (`HistGradientBoostingRegressor`), `feature_set` (`B`), `feature_count` (`29`), `random_seed` (`42`), `delay_minutes` (`15`), `l2_regularization` (`1.0`), `learning_rate` (`0.05`), `max_iter` (`200`), `max_leaf_nodes` (`31`), `min_samples_leaf` (`100`).
  - **Metrics**: `daylight_mae_kwh` (`31.4262`), `daylight_rmse_kwh` (`53.3511`), `capacity_norm_daylight_mae` (`0.0352`), `all_hour_mae_kwh` (`15.8267`), `all_hour_rmse_kwh` (`37.7508`), `improvement_over_baseline_pct` (`25.33`), `training_duration_sec`.
  - **Artifacts**: `plots/` (`residual_scatter.png`, `hourly_timeseries_june2022.png`, etc.), `evaluation/evaluation.json`, `bundle/solarcast_f1_bundle.joblib`, registered model `solarcast-f1`.
- **Run 2: `Ridge_Set_B_15mDelay` (Rejected Candidate)**:
  - **Params**: `model_type` (`Ridge`), `alpha` (`10.0`), `fit_intercept` (`True`), `feature_set` (`B`), `feature_count` (`29`).
  - **Tags**: `status` (`rejected`), `is_best_candidate` (`False`).
  - **Metrics**: `daylight_mae_kwh` (`42.62`), `improvement_over_baseline_pct` (`-1.26%`).
- **Run 3: `dry_run_last_fold_evaluation`**:
  - **Tags**: `champion_model` (`HistGradientBoosting_Set_B`), `eval_type` (`dry_run`), `final_test_opened` (`False`), `git_commit`, `schema_version` (`1.1.0`).
  - **Metrics**: `<model_name>_primary_daylight_mae`, `<model_name>_primary_daylight_rmse`, `<model_name>_primary_cap_mae`, `<model_name>_all_daylight_mae`, `<model_name>_all_hour_mae`, `evaluation_duration_sec`.
  - **Params**: `capacity_dc_kw`, `delay_minutes`, `eval_mode`, `excluded_test_hours`, `partition`, `primary_daylight_hours`, `reason`, `total_hours`, `weather_missing_hours`.
- **Run 4: `final_test_evaluation` (One-Time Audited Official Run)**:
  - **Tags**: `champion_model` (`HistGradientBoosting_Set_B`), `eval_type` (`final_test`), `final_test_opened` (`True`), `git_commit`.
  - **Metrics**: `histgradientboosting_set_b_primary_daylight_mae` (`32.6863`), `histgradientboosting_set_b_primary_cap_mae` (`0.0366`), `clear_sky_scaled_15m_persistence_primary_daylight_mae` (`43.4825`).

### 4.2 Experiment 2: `solarcast`
Tracks operational data infrastructure, Feast feature store materialization, and production deviation detection.

- **Run 1: `feature_store_materialization`**:
  - **Tags**: `entity` (`site`), `entity_key` (`site_id`), `feast_repo_git_commit`, `feast_version` (`0.68.0`), `feature_view_name` (`solarcast_f1_set_b`), `offline_format` (`parquet`), `online_store` (`sqlite`), `owner` (`member1`).
  - **Params**: `repo_path`, `feature_views`, `start_date_utc`, `end_date_utc`, `ttl_days` (`7`).
  - **Metrics**: `duration_seconds`, `materialized_views_count`.
- **Run 2: `deviation_detector`**:
  - **Tags**: `data_window` (`2018-07-12 to 2023-11-09`), `model_bundle` (`models/solarcast_f1_bundle.joblib`), `model_name` (`solarcast-f1`), `task` (`M1.6_deviation_detector`), `owner` (`member1`).
  - **Params**: `consecutive_hours_threshold` (`2`), `bins` (`clear,mixed,overcast,no_weather`), `random_seed` (`42`).
  - **Metrics**: `injected_zero_detection_rate_pct` (`95.0`), `injected_badval_detection_rate_pct` (`90.0`), `injected_drop30_detection_rate_pct` (`60.0`), `isolated_single_hour_compliance_pct` (`100.0`), `unmodified_fa_per_1000h` (`0.43`), `natural_sz_detection_rate_pct` (`13.95`), `natural_sz_median_delay_h` (`3.0`).
  - **Artifacts**: `deviation.yaml`, `alert_event_schema.json`, `deviation_summary.json`.

### 4.3 Model Registry, Cryptographic Lineage & Resolver
The MLflow Model Registry (`solarcast-f1`) is configured on the local file store and will migrate cleanly to the shared server:
- **Alias `champion`**: Points to **Version 4** (`f7fef3f7bdf241c5a53dc50d8e4713f9`).
- **Version 4 Tags**: `mode=F1`, `feature_set=B`, `schema_version=1.1.0`, `gate_eligible=true`, `final_test_opened=true`, `bundle_sha256=f72de152727fd406ac35a08cfa83289b6a61b031e0b3f4eb52008288e3980619`, `source_run_id=f7fef3f7bdf241c5a53dc50d8e4713f9`, `status=champion`.
- **Versions 1–3**: Tagged with `status=superseded` with factual run-derived descriptions.
- **Registry Snapshot**: `reports/model_registry_status.json`.
- **Lineage Helper**: `src/models/lineage.py` (`log_lineage(run)`) is stdlib-only and automatically sets full provenance hashes (data files, manifest, feature schema, configs, bundle, git commit/dirty status) across both `.venv` and `.venv-feast`.
- **Lineage Verification CLI**:
  ```powershell
  python scripts/resolve_run.py champion
  python scripts/resolve_run.py <run_id>
  ```
  See `docs/mlflow_lineage.md` for complete audit methodology and shared server migration details.

---

## 5. `evaluation.json` Schema & Acceptance Gate Semantics

For new runs use `models/candidates/<candidate-id>/evaluation.json`, bound by that run's manifest. Root
`evaluation.json` and `reports/evaluation.json` are historical records. The examples below describe that
historical champion, not promised scores for future candidates. New results also include `candidate_id`,
`bundle_sha256`, `validation_sha256`, `validation_scope`, `validation_window` and `cross_validation_gate`.
Unknown metrics/improvements use JSON null and reject eligibility. Both the stored CV gate and the exact
frozen candidate's development metrics must pass the unchanged numerical thresholds.

### 5.1 JSON Field Definitions
- `eligible` (`bool`): `true` if candidate meets all acceptance criteria; `false` otherwise.
- `required_improvement_pct` (`float`): Minimum Daylight MAE reduction required vs baseline champion (`5.0%`).
- `achieved_improvement_pct` (`float`): Percentage improvement achieved by candidate on validation CV (`25.33%`).
- `thresholds` (`object`):
  - `max_daylight_mae_kwh` (`float`): Absolute upper bound on Daylight MAE (`39.98 kWh`).
  - `max_capacity_norm_daylight_mae` (`float`): Maximum capacity-normalized error (`0.20`, or 20.0%).
  - `capacity_limit_kwh` (`float`): DC capacity ceiling (`893.0 kWh`).
- `baseline_metrics` (`object`):
  - `daylight_mae_kwh` (`float`): Baseline champion Daylight MAE (`42.0891 kWh`).
  - `daylight_rmse_kwh` (`float`): Baseline champion RMSE (`67.5237 kWh`).
  - `capacity_norm_daylight_mae` (`float`): Baseline champion normalized error (`0.0471`).
  - `all_hour_mae_kwh` (`float`): All-hour MAE (`21.1500 kWh`).
  - `all_hour_rmse_kwh` (`float`): All-hour RMSE (`47.8266 kWh`).
- `candidate_metrics` (`object`):
  - `daylight_mae_kwh` (`float`): Champion candidate Daylight MAE (`31.4262 kWh`).
  - `daylight_rmse_kwh` (`float`): Champion candidate RMSE (`53.3511 kWh`).
  - `capacity_norm_daylight_mae` (`float`): Champion candidate normalized error (`0.0352`).
  - `all_hour_mae_kwh` (`float`): All-hour MAE (`15.8267 kWh`).
  - `all_hour_rmse_kwh` (`float`): All-hour RMSE (`37.7508 kWh`).
- `failure_reasons` (`list[str]`): List of failure descriptions (empty `[]` if passed).
- `final_test_opened` (`bool`): `true` indicating the 20% test partition has been unlocked.
- `final_test_opened_at_utc` (`str`): ISO timestamp when final test set was evaluated.
- `final_test_reason` (`str`): Reason documented for opening final test set.
- `final_test_metrics` (`object`): Slices for `primary_daylight`, `all_daylight`, `all_hour`, and `generalization_gap`.
- `candidate_model_name` (`str`): `"HistGradientBoosting_Set_B"`.
- `baseline_champion_name` (`str`): `"Clear-Sky Scaled 15-Minute Persistence"`.

### 5.2 Acceptance Gate Rules
1. **Daylight MAE Improvement**: $\ge 5.0\%$ relative improvement over baseline champion on validation CV. Achieved: $+25.33\%$ ($\text{PASSED}$).
2. **Capacity-Normalized Bound**: $\le 0.20$ ($20.0\%$). Achieved: $0.0352$ ($3.52\%$, $\text{PASSED}$).
3. **Physical Constraints**: All predictions strictly finite, non-negative ($\ge 0.0\text{ kWh}$), and $\le 893.0\text{ kWh}$ ($\text{PASSED}$).

---

## 6. Final Test Safeguard: Rules for Automation & Retraining DAGs

> [!CAUTION]
> The final 20% test partition (2022-10-17 to 2023-11-09 UTC, 9,273 valid hours) has **already been opened** on 2026-11-23 under human owner authorization (`reports/final_test_audit.json`).
>
> **Under NO circumstances may Member 2's CI/CD workflows, automated retraining DAGs, or scheduled pipelines execute:**
> ```powershell
> # FORBIDDEN IN AUTOMATION / CI / AIRFLOW:
> python -m src.models.evaluate --final-test --force ...
> ```
> An existing audit blocks all repeated final-test calls, including `--force --reason`. The default
> audit path is anchored to the repository, so changing working directory cannot bypass it. A fresh first
> use requires explicit force/reason outside CI/scheduler contexts and claims the audit exclusively before
> any data I/O. Failed first evaluations retain their claim; never delete/reset it for a retry.
> 
> **Rule for Member 2**: To validate a retrained saved candidate invoke:
> ```powershell
> python -m src.models.evaluate --candidate-manifest "models/candidates/<candidate-id>/manifest.json"
> ```
> This checks the exact frozen candidate on development Fold 3. CI uses deterministic synthetic tests;
> never rerun the research or final-test benchmarks as a substitute for this artifact check.

---

## 7. Dual Virtual Environment Architecture

Feast 0.68.0 strictly requires `numpy>=2.0.0`, whereas our pinned modeling stack requires `numpy==1.26.4` and `scipy==1.13.1` (see `requirements.txt`). To prevent dependency corruption, two separate virtual environments are maintained:

```
SolarCast/
├── .venv/               # Main modeling environment: numpy 1.26.4, scikit-learn 1.5.2, scipy 1.13.1, mlflow 2.17.2
└── .venv-feast/         # Isolated Feast environment: feast 0.68.0, numpy 2.4.6, mlflow 3.18.0
```

### 7.1 CI Test Suite Execution
In CI runners (e.g. GitHub Actions), run tests using the main `.venv`. Feast unit tests (`tests/test_feature_store.py`) contain `pytest.importorskip("feast")` and will **auto-skip cleanly** without failing the build.

```powershell
# Exact CI command to run in GitHub Actions / runner:
pytest -m "not network"
```
- **Pytest Markers**:
  - `@pytest.mark.network`: Skips tests requiring live external HTTP connections (e.g., live NASA POWER API or OEDI S3 downloads).
  - `-m "not network"`: Runs the core offline suite; use the latest HANDOFF entry for observed counts and runtime limitations.

---

## 8. Measured Pipeline Runtimes

Measured on the reference Windows host (Python 3.11):

| Pipeline Stage | Command | Typical Duration | Notes |
|---|---|:---:|---|
| **Data Ingestion** | `python -m src.data.ingestion` | 15–30 s | Cached check is < 2 s; initial download ~25 s |
| **Data Aggregation** | `python -m src.data.aggregation` | 25–35 s | Aggregates 240,062 raw 15-min rows into hourly/quarter-hourly |
| **HRRR Extraction** | `python scripts/hrrr_extract.py --merge` | 1–2 s | Merging pre-extracted parts (full extraction: 2–4 h; use Colab/Drive) |
| **Feast Feature Build** | `python -m src.features.build_feature_table` | 15–20 s | Generates 23,532 daylight issue hours for Set B |
| **Feast Materialization** | `python feature_store/materialize.py` | 2–5 s | Materializes 2-week window into SQLite online store |
| **Model Training & Gating** | `python -m src.models.train` | Not remeasured | 3-fold CV, freeze before Fold 3, save isolated candidate and exact-model gate |
| **Saved Candidate Check** | `python -m src.models.evaluate --candidate-manifest <path>` | Not measured | Hash/chronology check and saved-bundle predictions; no fit |
| **Deviation Calibration** | Explicit `--calibrate --bundle ...` command in section 2.7 | Not remeasured | Dev-only OOF model-recipe calibration into isolated artifacts |
| **Pytest CI Suite** | `pytest -m "not network"` | ~56 s | 61 unit tests pass, 1 skipped cleanly |
| **Feast Test Suite** | `.venv-feast: pytest tests/test_feature_store.py` | ~33 s | 9 unit tests pass in dedicated Feast environment |

---

## 9. Clean-Clone Reproduction Checklist

To reproduce the entire pipeline from scratch on a clean machine:

Use the reviewed integration revision containing the safeguards. `m1/handoff-docs` is historical.
The local repair branch is `m1/evaluation-safeguards`; publish it before expecting another teammate to
clone these fixes. Historical artifact restoration and creating new candidates are separate operations.

- [ ] **1. Clone repository & checkout branch**:
  ```powershell
  git clone <repo-url>
  cd SolarCast
  git checkout <reviewed-integration-commit-or-branch>
  ```
- [ ] **2. Create and initialize main `.venv`**:
  ```powershell
  py -3.11 -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install --upgrade pip
  pip install -r requirements.txt
  ```
- [ ] **3. Pull versioned datasets via DVC**:
  ```powershell
  dvc pull
  ```
  *(Verify that `data/raw/` and `data/processed/` match the hashes in Section 3).*
- [ ] **4. Create and initialize isolated `.venv-feast`**:
  ```powershell
  py -3.11 -m venv .venv-feast
  .\.venv-feast\Scripts\pip install --upgrade pip
  .\.venv-feast\Scripts\pip install -r requirements-feast.txt
  ```
- [ ] **5. Run primary project test suite**:
  ```powershell
  .\.venv\Scripts\pytest -m "not network"
  ```
  *(Expect: 61 passed, 1 skipped).*
- [ ] **6. Run Feast feature store test suite**:
  ```powershell
  .\.venv-feast\Scripts\pytest tests/test_feature_store.py
  ```
  *(Expect: 9 passed).*
- [ ] **7. Verify linter and style**:
  ```powershell
  .\.venv\Scripts\ruff check .
  ```
  *(Expect: All checks passed).*
- [ ] **8. Verify the new candidate's manifest** (use the actual ID printed by training):
  ```powershell
  .\.venv\Scripts\python -m src.models.evaluate --candidate-manifest "models/candidates/<candidate-id>/manifest.json"
  ```
  *(Expect: Fold 3 Daylight MAE 31.76 kWh for HGB Set B, exit code 0).*

---

## 10. Automated Retraining DAG Architecture & Alert Triggering

When production deviation alerts or feature drift are detected, an automated retraining workflow must be executed by Member 2.

```
+-----------------------------------------------------------------------------------------+
|                                    TRIGGER SOURCES                                      |
|                                                                                         |
|  [ Operational Alert Event Stream (configs/alert_event_schema.json v1.1.0) ]            |
|   - residual_deviation: 2 consecutive hours breaching P1/P99 residual bounds             |
|   - input_drift: PSI > 0.20 in >= 2 features on 2 consecutive days (>= 200 rows)        |
|   - performance_degradation: 7-day daylight MAE > 1.25x ref MAE on 2 days (>= 30 labels)|
|   - data_quality: Non-fresh weather share > 15% over trailing 7 days                    |
+-----------------------------------------------------------------------------------------+
                                             │
                                             ▼
+-----------------------------------------------------------------------------------------+
|                                MEMBER 2: AIRFLOW / KFP RETRAINING DAG                   |
|                                                                                         |
|  1. Ingestion Webhook / Event Consumer:                                                 |
|     - Validates payload against configs/alert_event_schema.json                         |
|     - Retraining Trigger Policy (team plan proposal; Member 2 decides):                 |
|       * 24-hour cooldown between retraining runs                                        |
|       * At most one active retraining run per site                                      |
|       * Minimum number of new verified labels before triggering                         |
|                                                                                         |
|  2. Dataset Ingestion & Update:                                                         |
|     - Ingests recent generation data up to latest verified date                        |
|     - Ingests latest HRRR forecast features via DVC / S3                               |
|     - Runs: python -m src.data.aggregation --config configs/data.yaml                   |
|     - Generates new DVC data snapshot and records data version hash                     |
|                                                                                         |
|  3. Model Retraining & Tuning:                                                          |
|     - Runs: python -m src.models.train --eval-config configs/evaluation.yaml            |
|     - Fits candidate on expanding chronological window                                  |
|     - Evaluates acceptance gate in src/models/evaluate.py                               |
|                                                                                         |
|  4. Retraining Acceptance Gate (Airflow Branching):                                     |
|     - Condition: candidate_metrics.daylight_mae <= 0.98 * champion_daylight_mae        |
|       (Candidate must beat champion by >= 2.0% without seasonal slice > 5.0% worse)     |
|     - IF PASSED:                                                                        |
|       * Promotes candidate in MLflow Model Registry (tag: "production")                 |
|       * Calibrates explicit bundle on dev folds; no final-test benchmark (see 2.7) |
|       * Updates Feast feature table and materializes online store                       |
|     - IF REJECTED:                                                                      |
|       * Keeps current champion in production                                            |
|       * Emits notification to operations Slack / PagerDuty with failure reasons         |
+-----------------------------------------------------------------------------------------+
```

### 10.1 Division of Responsibilities for Retraining
- **Member 1 Provides**:
  1. `configs/alert_event_schema.json`: Standardized event contract for alert payloads.
  2. `configs/deviation.yaml`: Calibrated bounds and consecutive-hour alerting state machine.
  3. `src/models/train.py`: Unified retraining entrypoint with expanding-window cross-validation.
  4. `src/models/evaluate.py`: Acceptance gate evaluation logic (`evaluate_candidate_gate`).
  5. `models/solarcast_f1_bundle.joblib`: Bundling logic enforcing physical constraints.
- **Member 2 Builds**:
  1. Airflow DAG or Kubeflow Pipeline orchestrating steps 1 to 4.
  2. Event listener/webhook queue consuming deviation alert events.
  3. Retraining trigger policy check (team plan proposal: 24-hour cooldown, at most one active retraining run per site, and minimum number of new labels; Member 2 decides).
  4. DVC snapshot tagging and S3 remote push.
  5. Notification integration (Slack, email, or monitoring dashboard).

---

## 11. Artifact Packet Specification, Verification & Ingestion

To facilitate DVC initialization, CI/CD runner setup, and pipeline replication without downloading multi-gigabyte raw datasets, Member 1 has packaged an offline artifact packet outside the repository at `D:\SolarCast_packet` and compressed archive `D:\SolarCast_packet.zip`.

### 11.1 Packet Summary & Key Metrics
- **Packet Directory**: `D:\SolarCast_packet` (never committed to Git)
- **Compressed Archive**: `D:\SolarCast_packet.zip` (13,535,820 bytes, ~12.91 MB; SHA-256: `93504e87d870e7881235d3de1fbedbff712492ae567abb05107af5536e797e21`)
- **Total Payload Files**: 31 files (28,116,883 bytes uncompressed, ~26.81 MB)
- **Inclusions**:
  - `data/raw/`: 20 files (raw 15-minute active power meter CSV, site metadata JSONs, raw environment samples, 7 annual NASA POWER hourly weather JSON files).
  - `data/processed/`: 7 files (`hourly.parquet`, `quarter_hourly.parquet`, `quarantine.parquet`, `hrrr_coverage_scan.parquet`, `hrrr_forecast_features.parquet`, `feast/features_set_b.parquet`, `validation_residuals.parquet`).
  - `models/`: Champion bundle (`solarcast_f1_bundle.joblib`).
  - Convenience files: `reports/evaluation.json`, `reports/final_test_audit.json`, `configs/data_manifest.json`.
- **Exclusions**:
  - `data/processed/hrrr_parts/`: Intermediate raw monthly chunk files prior to merge (superseded by `hrrr_forecast_features.parquet`).
  - `data/processed/june_week_extracted.parquet` and `mixed_week_extracted.parquet`: Exploratory samples from early M1.1b feasibility check, not used in documented production pipelines.
  - `mlruns/`: Excluded due to absolute Windows file paths in local store; runs will be re-logged to the shared server by running `python -m src.models.train` once Member 2 configures the central tracking URI.

### 11.2 Verification & Installation Commands
```powershell
# 1. Verify every file against CHECKSUMS.sha256:
python scripts/verify_artifact_packet.py D:\SolarCast_packet

# 2. Verify and install into a clean repository clone:
python scripts/verify_artifact_packet.py D:\SolarCast_packet --install C:\path\to\SolarCast
```

### 11.3 Initializing DVC Tracking for the Packet
```powershell
dvc add data/raw
dvc add data/processed
dvc add models
git add data/.gitignore data/raw.dvc data/processed.dvc models/.gitignore models.dvc
git commit -m "chore(dvc): track raw/processed datasets and model bundle"
dvc push
```

