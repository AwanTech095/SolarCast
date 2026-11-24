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
Extracts 2-meter air temperature (`TMP`) and cloud cover (`TCDC`, `LCDC`, `MCDC`, `HCDC`) from the MesoWest NOAA HRRR Zarr archive (`s3://hrrrzarr/`) on AWS `us-west-1` for all daylight hours between 2018-07-12 and 2023-11-09.

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
Trains baseline models, tunes Ridge Regression and `HistGradientBoostingRegressor` over 3 expanding-window chronological cross-validation folds on the 80% Train/Dev partition (37,090 hours), evaluates the candidate acceptance gate, fits the champion model on all valid daylight training hours, logs lineage to MLflow, and packages `models/solarcast_f1_bundle.joblib`.

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
  - `0`: Training and cross-validation completed. If candidate gate fails, sets `status="rejected"` and `eligible=false` in `evaluation.json`, but exits cleanly without crashing.
  - `1`: Missing input datasets or corrupted parquets.
- **Idempotency**: Deterministic given fixed random seed (`random_seed: 42`). Overwrites `reports/evaluation.json`, `./evaluation.json`, diagnostic plots in `reports/plots/`, and `models/solarcast_f1_bundle.joblib`.

---

### 2.7 Step 7: Deviation Calibration & Alert Engine
Calibrates 1st and 99th percentile residual thresholds across out-of-sample validation folds, updates `configs/deviation.yaml`, benchmarks detection on synthetic fault injections, evaluates historical natural anomalies, and logs run to MLflow.

```powershell
# Deviation detector calibration & benchmarking (main .venv)
.\.venv\Scripts\python -m src.models.deviation --log-mlflow --reports-dir reports --seed 42
```
- **CLI Flags**:
  - `--log-mlflow`: Log metrics, artifacts, and parameters to MLflow experiment `solarcast`.
  - `--eval-config <path>`: Path to `configs/evaluation.yaml`.
  - `--deviation-config <path>`: Path to `configs/deviation.yaml`.
  - `--hourly <path>`: Path to `data/processed/hourly.parquet`.
  - `--features <path>`: Path to `data/processed/feast/features_set_b.parquet`.
  - `--bundle <path>`: Path to `models/solarcast_f1_bundle.joblib`.
  - `--reports-dir <path>`: Destination directory for `deviation_summary.json`.
  - `--seed <int>`: Random seed for fault injection intervals (default: `42`).
- **Exit Codes**:
  - `0`: Thresholds calibrated and persisted to `configs/deviation.yaml`.
  - `1`: Missing bundle or input files.

---

### 2.8 Step 8: Evaluation Dry-Run (Automated CI Benchmark)
Evaluates all baselines and champion models strictly on Fold 3 of Train/Dev. This command must be used by automated CI pipelines to verify model performance without touching the sealed final test partition.

```powershell
# CI benchmark dry run (main .venv)
.\.venv\Scripts\python -m src.models.evaluate --dry-run --reports-dir reports
```
- **CLI Flags**:
  - `--dry-run`: Evaluates Fold 3 within Train/Dev partition (`final_test_opened: false`).
  - `--reports-dir <path>`: Destination for `dry_run_last_fold_summary.json` (default: `reports/`).
- **Exit Codes**:
  - `0`: Evaluation finished, summary written, logged to MLflow as `dry_run_last_fold_evaluation`.

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

---

## 5. `evaluation.json` Schema & Acceptance Gate Semantics

The evaluation gate result is recorded in `evaluation.json` and `reports/evaluation.json`.

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
> Calling `--final-test` without `--force` intentionally crashes with `RuntimeError: PROTECTED TEST SET VIOLATION`.
> 
> **Rule for Member 2**: Automated pipelines and regression tests must **always** invoke:
> ```powershell
> python -m src.models.evaluate --dry-run
> ```
> which evaluates strictly on Fold 3 within the 80% Train/Dev partition without touching the held-out test set.

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
  - `-m "not network"`: Executes 61 unit tests across data validation, aggregation, feature engineering, baselines, models, and deviation detector in ~56 seconds.

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
| **Model Training & Gating** | `python -m src.models.train` | 25–35 s | 3-fold CV tuning, Set A vs B benchmark, refit champion, save bundle |
| **Evaluation Dry-Run** | `python -m src.models.evaluate --dry-run` | 20–30 s | Full evaluation on Fold 3 within Train/Dev |
| **Deviation Calibration** | `python -m src.models.deviation --log-mlflow` | 10–15 s | Calibrates P1/P99 thresholds, benchmarks fault injection |
| **Pytest CI Suite** | `pytest -m "not network"` | ~56 s | 61 unit tests pass, 1 skipped cleanly |
| **Feast Test Suite** | `.venv-feast: pytest tests/test_feature_store.py` | ~33 s | 9 unit tests pass in dedicated Feast environment |

---

## 9. Clean-Clone Reproduction Checklist

To reproduce the entire pipeline from scratch on a clean machine:

- [ ] **1. Clone repository & checkout branch**:
  ```powershell
  git clone <repo-url>
  cd SolarCast
  git checkout m1/handoff-docs
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
- [ ] **8. Run evaluation dry-run benchmark**:
  ```powershell
  .\.venv\Scripts\python -m src.models.evaluate --dry-run
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
|       * Recalibrates deviation thresholds: python -m src.models.deviation --log-mlflow |
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
