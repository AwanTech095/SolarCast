# SolarCast Feast Feature Store Architecture & Operations

## 1. Purpose
The SolarCast Feature Store provides point-in-time correct feature extraction for offline training/backtesting and low-latency online feature serving for production inference (Mode F1 next-hour solar AC energy forecasting). It guarantees zero training-serving skew by serving features from a single unified feature pipeline.

## 2. Entity and Feature View Specifications
- **Entity**: `site`
  - Join Key: `site_id` (`ValueType.STRING`)
  - Configured Site: `"2107"` (System 2107 Farm Solar Array, Arbuckle, CA)
- **Feature View**: `solarcast_f1_set_b`
  - Owner: `member1`
  - Schema Version: `1.1.0`
  - Feature Count: 29 model features (Feature Set B)
  - Full schema, ordered column definitions, data types, physical units, and operational conventions are specified in [configs/feature_schema.json](file:///c:/projects/SolarCast/configs/feature_schema.json).
  - Summary of feature categories:
    1. *Production telemetry lags & aggregates* (7 features): `power_kw_lag_0m`, `power_kw_lag_15m`, `power_kw_lag_30m`, `power_kw_lag_45m`, `power_kw_last_4_mean`, `energy_kwh_latest_avail_hour`, `energy_kwh_prev_day_same_hour`, `energy_kwh_prev_week_same_hour`, `energy_kwh_rolling_mean_6h`, `energy_kwh_rolling_mean_24h`.
    2. *Solar geometry & clear-sky physics* (5 features): `hour_sin`, `hour_cos`, `day_of_year_sin`, `day_of_year_cos`, `solar_elevation_target_deg`, `clearsky_ghi_target_wm2`, `clearsky_energy_target_kwh`, `clearsky_index_latest_avail`.
    3. *Operational telemetry flags* (5 features): `production_age_minutes`, `power_kw_missing_count`, `latest_hour_missing`, `prev_day_missing`, `prev_week_missing`.
    4. *NOAA HRRR NWP forecast weather* (6 features): `tmp_c` (2m temperature), `tcdc_pct` (total cloud cover), `clearsky_ghi_cloud_adj_wm2` (Kasten-Czeplak cloud-adjusted GHI), `weather_run_age_hours`, `weather_fallback_used`, `weather_missing`.

## 3. Storage Architecture & Freshness Expectation
- **Offline Store (Parquet)**:
  - Path: `data/processed/feast/features_set_b.parquet`
  - Rows: 23,532 daylight issue hours (2018-07-12 14:00 UTC to 2023-11-09 23:00 UTC)
  - Primary Timestamp: `event_timestamp` (issue time $H$ UTC)
  - Created Timestamp: `created_timestamp`
  - Audit Non-Leakage Columns: `production_cutoff_utc`, `weather_available_at_utc`, `source_event_time_max_utc` (100% verified $\le event\_timestamp$).
- **Online Store (SQLite)**:
  - Path: `feature_store/feature_repo/data/online_store.db`
  - Table: `solarcast_solarcast_f1_set_b`
- **Freshness Expectation**:
  - Operational Cadence: 1-hour issue interval.
  - Production Telemetry Latency: Nominal 15 minutes before issue time ($H - 15\text{m}$).
  - Forecast Weather Latency: Nominal 2-hour HRRR publication lag ($H - 2\text{h}$ initialization).
  - Maximum Allowable Online Staleness: 2.0 hours.

## 4. Environment Separation
The project maintains two distinct virtual environments:
1. **Main Project Environment (`.venv`)**:
   - Python 3.11 with pinned scikit-learn 1.5.2, scipy 1.13.1, and `numpy==1.26.4`.
   - Used for core model training, baseline cross-validation, evaluation gating, and primary Pytest CI.
2. **Feast Environment (`.venv-feast`)**:
   - Python 3.11 with `feast==0.68.0`, `numpy==2.4.6`, `pandas==2.3.3`, `pyarrow==26.0.1`, and `mlflow==3.18.0`.
   - **Why separated**: Feast 0.68.0 strictly requires `numpy>=2.0.0` (installing Feast into `.venv` breaks scikit-learn/scipy pinned against NumPy 1.26). Isolating Feast protects the core modeling dependencies without touching `requirements.txt`.
   - All Feast tests auto-skip cleanly via `pytest.importorskip("feast")` when running in `.venv`.

## 5. TTL Semantics in Feast 0.68.0
*(Note: Findings below are derived directly from reading the Feast 0.68.0 source code and our empirical demonstration, not from official documentation)*:

- **Historical Retrieval (`get_historical_features`)**:
  - Feast evaluates `ttl` (set to `timedelta(days=7)` in `feature_views.py`) as a sliding lookback window $[t_{\text{entity}} - \text{ttl}, t_{\text{entity}}]$.
  - The offline query engine performs an as-of join selecting the latest feature event at or before $t_{\text{entity}}$ within that window. If no record exists within $[t_{\text{entity}} - \text{ttl}, t_{\text{entity}}]$, null values are returned.
- **Online Retrieval (`get_online_features`)**:
  - Feast's SQLite online store implementation **does not evaluate TTL, query timestamps, or staleness at query time**.
  - When calling `store.get_online_features(entity_rows=[{"site_id": "2107"}])`, SQLite simply executes a lookup on `entity_key` and returns the latest materialized row in the table, completely ignoring whether that record is 1 hour old, 1 week old, or in the future relative to the client's clock.
- **Serving Provider Safeguards (`src/features/feast_provider.py`)**:
  - To prevent serving future or stale features, our wrapper inspects the underlying `event_timestamps` protobuf field returned by Feast.
  - If stored record timestamp > requested issue time: raises `FutureFeatureError` (never answering past query times with future observations).
  - If $(t_{\text{query}} - t_{\text{record}}) > \text{max\_staleness\_hours}$ (default 2.0h): raises `StaleFeatureError`.

## 6. Architecture Implications for Replay & Production Serving
- **Online Store Replay Rule**:
  - Because the online store maintains **only the latest materialized value** per entity key, historical replay simulation must **not** query the online store without step-by-step clock updates. Batch historical replay must use offline retrieval (`replay_historical_features` via `get_historical_features`), which natively performs point-in-time joins.
- **Single-Replica Limitation for Member 3 (Serving & Deployment)**:
  - The SQLite online store (`data/online_store.db`) is a local file database. It cannot be concurrently written or shared across distributed containers/replicas in Kubernetes or multi-worker Docker services.
  - For deployment, Member 3 must either:
    1. Mount a shared read-only volume snapshot of `online_store.db` across inference pods, or
    2. Configure a centralized network online store backend (e.g. Redis, PostgreSQL, or DynamoDB) in `feature_store.yaml`.

## 7. Real Demonstration Results
All real-data demonstrations executed in `scripts/demo_feast.py` using `.venv-feast`:
- **Materialization Window**:
  - Materialized interval: `2023-06-13 15:00:00+00:00` to `2023-06-15 15:00:00+00:00`.
  - Stored record timestamp in online store: `2023-06-15 15:00:00+00:00`.
- **Training-Serving Skew Demonstration (Demo a)**:
  - 20 random issue times sampled between `2019-06-29 22:00:00+00:00` and `2023-09-27 16:00:00+00:00`.
  - Compared Feast offline retrieval against `build_features` output.
  - **Maximum absolute discrepancy**: `0.00e+00` across all 20 samples and all 29 features (tolerance: `1e-9`). Zero skew verified.
- **Online Query & Parity (Demo b)**:
  - Query at `2023-06-15 15:00:00+00:00`: retrieved `clearsky_ghi_target_wm2 = 427.45 W/m^2`, `power_kw_lag_0m = 105.60 kW`, `tcdc_pct = 1.0%`, `tmp_c = 22.85 degC`. Staleness: `0.00` hours.
- **Future Feature Rejection (Demo c)**:
  - Query at `2023-06-15 11:00:00+00:00` (past relative to 15:00 UTC stored record) raised `FutureFeatureError`: *"Online store feature event_timestamp (2023-06-15T15:00:00+00:00) is later than requested issue_time_utc (2023-06-15T11:00:00+00:00). Cannot serve future features for past issue times."*
- **Staleness Rejection (Demo c)**:
  - Query at `2023-06-15 20:00:00+00:00` (5.0 hours after 15:00 UTC stored record with `max_staleness_hours=2.0`) raised `StaleFeatureError`: *"Online store features are stale for issue_time_utc=2023-06-15T20:00:00+00:00. Feature event_timestamp=2023-06-15T15:00:00+00:00 is 5.00 hours old (exceeds max_staleness_hours=2.0h)."*
- **MLflow Tracking**:
  - Run Name: `feature_store_materialization`
  - Tags:
    - `feast_repo_git_commit`: `5cdb42819813df63e97ef301a9f4765deefb3dfd`
    - `feature_view_name`: `solarcast_f1_set_b`
    - `entity`: `site`
    - `entity_key`: `site_id`
    - `offline_format`: `parquet`
    - `online_store`: `sqlite`
    - `feast_version`: `0.68.0`
    - `owner`: `member1`
  - Parameters: `repo_path`, `feature_views`, `start_date_utc`, `end_date_utc`, `ttl_days` (7).
  - Metrics: `duration_seconds` (1.26s), `materialized_views_count` (1.0).

## 8. Exact Windows PowerShell Commands
Run all feature store operations in PowerShell using the dedicated `.venv-feast` environment:

```powershell
# 1. Generate the offline Feature Set B Parquet table
.\.venv-feast\Scripts\python -m src.features.build_feature_table

# 2. Register entity and feature view with the Feast local registry
cd feature_store/feature_repo
..\..\.venv-feast\Scripts\feast apply
cd ../..

# 3. Materialize features into SQLite online store with MLflow logging
.\.venv-feast\Scripts\python feature_store/materialize.py --start 2023-06-01T00:00:00+00:00 --end 2023-06-15T15:00:00+00:00 --log-mlflow

# 4. Run the end-to-end real data demonstration (skew test, online query, rejections, replay)
.\.venv-feast\Scripts\python -m scripts.demo_feast

# 5. Run Feast unit test suite (9 comprehensive tests)
.\.venv-feast\Scripts\pytest tests/test_feature_store.py -v

# 6. Verify core project tests in .venv (Feast tests auto-skip cleanly)
.\.venv\Scripts\pytest -m "not network"

# 7. Verify code formatting and linting
.\.venv\Scripts\ruff check .
```
