# SolarCast: Member 3 Handoff Specification (Model Serving, Feast Online Ingestion & Production APIs)

**Author:** Member 1 (Data, Features, Feast, Models, MLflow)  
**Target Audience:** Member 3 (Flask API, Serving Deployment, Replica Scaling, Logging & Monitoring)  
**Model Name:** `solarcast-f1` (Version `1.1.0`, Feature Set B)  
**Schema Versions:** Feature Schema `1.1.0`, Alert Event Schema `1.0.0`  
**Repository Branch:** `m1/handoff-docs`

---

## 1. Model Artifact & Bundle Specification

Member 1 exports a self-contained, physically bounded model artifact using `joblib`.

- **Artifact Path:** `models/solarcast_f1_bundle.joblib`
- **Bundle Class:** `src.models.bundle.ModelBundle`
- **Champion Estimator:** `sklearn.ensemble.HistGradientBoostingRegressor` (fit on 18,603 valid daylight training hours)

### 1.1 Loading the Bundle
```python
from pathlib import Path
from src.models.bundle import load_bundle

bundle = load_bundle("models/solarcast_f1_bundle.joblib")
print(f"Loaded model: {bundle.model_name} (v{bundle.feature_schema_version})")
print(f"Feature set: {bundle.feature_set} ({len(bundle.feature_names)} features)")
print(f"Capacity ceiling: {bundle.capacity_limit_kwh:.1f} kWh")
```

### 1.2 Bundle Attributes
| Attribute | Type | Value / Description |
|---|:---:|---|
| `estimator` | `Any` | Trained `HistGradientBoostingRegressor` instance |
| `feature_names` | `list[str]` | Exact ordered list of 29 feature names (Feature Set B) |
| `training_window` | `dict[str, str]` | `start_utc`, `end_utc`, and `training_samples` (`"18603"`) |
| `data_hashes` | `dict[str, str]` | SHA-256 hashes of `hourly.parquet`, `quarter_hourly.parquet`, and `hrrr_forecast_features.parquet` |
| `mlflow_run_id` | `str \| None` | MLflow tracking run ID associated with bundle export |
| `capacity_limit_kwh` | `float` | `893.0` (Plant DC capacity in kWh, physical ceiling) |
| `model_name` | `str` | `"solarcast-f1"` |
| `feature_schema_version` | `str` | `"1.1.0"` |
| `feature_set` | `str` | `"B"` |
| `needs_weather` | `bool` | `True` (incorporates NOAA HRRR NWP forecast features) |
| `created_at_utc` | `str` | ISO 8601 UTC creation timestamp |

### 1.3 `predict()` Signature & Physical Constraints
```python
def predict(
    self,
    features_df: pd.DataFrame,
    weather_df: pd.DataFrame | None = None,
) -> np.ndarray:
```
- **Required Columns:** All 29 feature columns specified in `bundle.feature_names` must be present across `features_df` and optional `weather_df`. Missing columns raise `KeyError`.
- **Column Ordering:** `predict()` automatically selects and aligns columns via `features_df[self.feature_names].values`.
- **NaN Handling:** `HistGradientBoostingRegressor` natively processes `NaN` values during inference using learned histogram split directions. Missing weather values (`weather_missing = 1` with `tmp_c = NaN`, `tcdc_pct = NaN`) are natively supported without throwing errors.
- **Physical Bounding Guarantees:**
  - **Night Hours:** If `solar_elevation_target_deg <= 0.0` (or `is_daylight == False`), predictions are deterministically forced to **identically $0.0\text{ kWh}$**.
  - **Output Clipping:** Predictions during daylight are strictly clipped to $[0.0, 893.0\text{ kWh}]$ (between zero and plant nameplate DC capacity). Negative or exceeding values are impossible.
- **`needs_weather` Behavior:** When `bundle.needs_weather == True`, the bundle expects weather features. If `weather_df` is passed as a separate DataFrame, `predict()` merges any missing weather columns into `features_df` before evaluating.

---

## 2. Feature Schema v1.1.0 Specification Table

*(Generated directly from `configs/feature_schema.json` via `python scripts/gen_feature_doc.py --set B`)*

### Feature Schema v1.1.0 (Feature Set B, 29 Features)

| # | Name | Data Type | Units | Description |
|---|---|:---:|:---:|---|
| 1 | `power_kw_lag_0m` | `float64` | `kW` | 15-minute average active power of the latest interval ending at or before cutoff (interval [H-30m, H-15m)) |
| 2 | `power_kw_lag_15m` | `float64` | `kW` | 15-minute average active power 15 minutes before cutoff (interval [H-45m, H-30m)) |
| 3 | `power_kw_lag_30m` | `float64` | `kW` | 15-minute average active power 30 minutes before cutoff (interval [H-60m, H-45m)) |
| 4 | `power_kw_lag_45m` | `float64` | `kW` | 15-minute average active power 45 minutes before cutoff (interval [H-75m, H-60m)) |
| 5 | `power_kw_last_4_mean` | `float64` | `kW` | Mean 15-minute active power across the last 4 available intervals preceding cutoff |
| 6 | `energy_kwh_latest_avail_hour` | `float64` | `kWh` | Total active energy of the latest fully available hour preceding cutoff (interval [H-2h, H-1h)) |
| 7 | `energy_kwh_prev_day_same_hour` | `float64` | `kWh` | Total active energy for the same hour on the previous day (interval [H-24h, H-23h)) |
| 8 | `energy_kwh_prev_week_same_hour` | `float64` | `kWh` | Total active energy for the same hour on the previous week (interval [H-168h, H-167h)) |
| 9 | `energy_kwh_rolling_mean_6h` | `float64` | `kWh` | Past-only rolling mean of hourly energy over the previous 6 fully available hours [H-7h, H-1h) |
| 10 | `energy_kwh_rolling_mean_24h` | `float64` | `kWh` | Past-only rolling mean of hourly energy over the previous 24 fully available hours [H-25h, H-1h) |
| 11 | `hour_sin` | `float64` | `dimensionless` | Sine of target hour H (sin(2 * pi * H / 24)) |
| 12 | `hour_cos` | `float64` | `dimensionless` | Cosine of target hour H (cos(2 * pi * H / 24)) |
| 13 | `day_of_year_sin` | `float64` | `dimensionless` | Sine of day-of-year (sin(2 * pi * DOY / 365.25)) |
| 14 | `day_of_year_cos` | `float64` | `dimensionless` | Cosine of day-of-year (cos(2 * pi * DOY / 365.25)) |
| 15 | `solar_elevation_target_deg` | `float64` | `degrees` | Solar elevation angle evaluated at midpoint of target hour [H, H+1h) (H + 30m) |
| 16 | `clearsky_ghi_target_wm2` | `float64` | `W/m^2` | pvlib clear-sky Global Horizontal Irradiance at midpoint of target hour (H + 30m) |
| 17 | `clearsky_energy_target_kwh` | `float64` | `kWh` | Clear-sky reference expected energy for target hour (clearsky_ghi * P95 ratio) |
| 18 | `clearsky_index_latest_avail` | `float64` | `dimensionless` | Clear-sky index of the latest fully available hour [H-2h, H-1h) (clipped [0, 2]; default 1.0 at night) |
| 19 | `production_age_minutes` | `float64` | `minutes` | Age of the latest available 15-minute production observation at issue time H (nominal 15 min) |
| 20 | `power_kw_missing_count` | `int64` | `count` | Count of missing 15-minute readings among the last 4 intervals preceding cutoff (0 to 4) |
| 21 | `latest_hour_missing` | `int64` | `flag` | Indicator flag: 1 if the latest fully available hour [H-2h, H-1h) was missing or invalid; 0 otherwise |
| 22 | `prev_day_missing` | `int64` | `flag` | Indicator flag: 1 if same hour previous day [H-24h, H-23h) was missing or invalid; 0 otherwise |
| 23 | `prev_week_missing` | `int64` | `flag` | Indicator flag: 1 if same hour previous week [H-168h, H-167h) was missing or invalid; 0 otherwise |
| 24 | `tmp_c` | `float64` | `degC` | HRRR NWP 2-meter air temperature forecast averaged over leads F1 and F2 for target hour [H, H+1h) |
| 25 | `tcdc_pct` | `float64` | `%` | HRRR NWP total cloud cover forecast percentage averaged over leads F1 and F2 for target hour [H, H+1h) |
| 26 | `clearsky_ghi_cloud_adj_wm2` | `float64` | `W/m^2` | Clear-sky GHI adjusted by forecast cloud cover via Kasten-Czeplak parameterization: clearsky_ghi_target_wm2 * (1 - 0.75 * (tcdc_pct/100)^3.4) |
| 27 | `weather_run_age_hours` | `float64` | `hours` | Age of the HRRR NWP initialization run relative to issue time H (nominal 2 hours, higher if fallback) |
| 28 | `weather_fallback_used` | `int64` | `flag` | Indicator flag: 1 if fallback to an earlier HRRR run (>2h age) was used; 0 if nominal 2h run used or weather missing |
| 29 | `weather_missing` | `int64` | `flag` | Indicator flag: 1 if forecast weather data is missing/unobtainable; 0 if weather data successfully ingested |

---

## 3. Request-to-Prediction Flow (Inference & Historical Replay)

### 3.1 Live Inference Request Flow
```
Client Request (POST /predict)
  { "site_id": "2107", "issue_time_utc": "2023-06-15T15:00:00Z" }
               │
               ▼
1. Input Validation:
   - Verify site_id == "2107"
   - Parse ISO 8601 UTC timestamp
               │
               ▼
2. Feast Online Store Lookup:
   src.features.feast_provider.get_online_features(
       site_id="2107",
       issue_time_utc="2023-06-15T15:00:00Z",
       max_staleness_hours=2.0
   )
   - Checks:
     * Entity exists? -> if not: raise EntityNotFoundError (HTTP 503)
     * Feature timestamp > issue_time? -> if yes: raise FutureFeatureError (HTTP 503 / 400)
     * Age > max_staleness_hours? -> if yes: raise StaleFeatureError (HTTP 503)
               │
               ▼
3. Model Inference:
   features_df = pd.DataFrame([features_dict])
   pred_kwh = bundle.predict(features_df)[0]
               │
               ▼
4. Response Assembly & Emission:
   Returns JSON with predicted energy, intervals, versions, and warning flags.
```

### 3.2 Historical Replay Simulation
Because the Feast online SQLite store holds only the latest materialized record per site, historical replay over multiple consecutive hours must **not** query the online store without updating the store at every step. Instead, batch historical replay should use `src.features.feast_provider.replay_historical_features`:

```python
from src.features.feast_provider import replay_historical_features
from src.models.bundle import load_bundle

bundle = load_bundle("models/solarcast_f1_bundle.joblib")
issue_times = ["2023-06-15T12:00:00+00:00", "2023-06-15T13:00:00+00:00", "2023-06-15T14:00:00+00:00"]

features_df = replay_historical_features(site_id="2107", issue_times=issue_times)
predictions = bundle.predict(features_df)
```

---

## 4. API Response Contract & HTTP Error Status Codes

### 4.1 Recommended JSON Response Fields
```json
{
  "site_id": "2107",
  "issue_time_utc": "2023-06-15T15:00:00+00:00",
  "interval_start_utc": "2023-06-15T15:00:00+00:00",
  "interval_end_utc": "2023-06-15T16:00:00+00:00",
  "predicted_energy_kwh": 384.21,
  "model_name": "solarcast-f1",
  "model_version": "1.1.0",
  "feature_schema_version": "1.1.0",
  "feature_staleness_hours": 0.0,
  "warnings": [],
  "created_at_utc": "2026-11-24T11:45:00.000000+00:00"
}
```
- **Warning values to populate**:
  - `"weather_missing"`: When `weather_missing == 1`.
  - `"weather_fallback_used"`: When `weather_fallback_used == 1` (forecast is older than nominal 2h).
  - `"night_forced_zero"`: When `solar_elevation_target_deg <= 0.0`.
  - `"approaching_staleness"`: When `feature_staleness_hours > 1.5`.

### 4.2 Standard HTTP Status Codes & Error Semantics
| HTTP Status | Error Type | Cause | Recommended Action |
|:---:|---|---|---|
| **`200 OK`** | Success | Features available, model prediction generated | Return prediction payload |
| **`400 Bad Request`** | Invalid Request Body | Malformed JSON, unparseable timestamp, or unrecognized `site_id` (only `"2107"` supported) | Client must correct request payload |
| **`422 Unprocessable`** | Out-of-Bounds Features | Client supplied explicit features with unphysical values (e.g., negative power, TCDC outside [0, 100]) | Reject bad telemetry |
| **`503 Service Unavailable`** | `EntityNotFoundError` | Entity `"2107"` has not been materialized in Feast online store | Trigger materialization pipeline |
| **`503 Service Unavailable`** | `StaleFeatureError` | Stored features are older than `max_staleness_hours` (default 2.0h) | Check plant telemetry stream / weather ingest |
| **`503 Service Unavailable`** | `FutureFeatureError` | Requested `issue_time_utc` is in the past relative to the latest online store record | Check clock synchronization or replay setup |

---

## 5. SQLite Online Store Scalability & Replica Volume Snapshots

### 5.1 SQLite Concurrency Limitation
The default Feast online store backend is SQLite (`feature_store/feature_repo/data/online_store.db`). SQLite uses file-level locking and **does not support concurrent writes** from distributed workers, multiple Docker containers, or Kubernetes pods.

### 5.2 Creating a Static Snapshot for Serving Replicas
For deploying multi-replica Flask services without running an external database:
1. In the build/CI environment, materialize the desired operational time window into the SQLite store using `.venv-feast`:
   ```powershell
   .\.venv-feast\Scripts\python feature_store/materialize.py --start 2023-06-01T00:00:00+00:00 --end 2023-06-15T15:00:00+00:00
   ```
2. Copy the resulting SQLite database into a dedicated deployment snapshot:
   ```powershell
   Copy-Item feature_store/feature_repo/data/online_store.db data/online_store_snapshot.db
   ```
3. Mount `data/online_store_snapshot.db` into the Flask Docker containers as a **read-only volume** (`:ro`). SQLite supports arbitrary concurrent readers when no write locks are requested.

*(Alternatively, for production clusters with dynamic writes, configure Redis or PostgreSQL in `feature_store/feature_repo/feature_store.yaml`)*.

---

## 6. Audit Logging Schema for Offline Analysis & Retraining Joins

To evaluate model accuracy after ground truth arrives and to feed the deviation detector, Member 3 must record two asynchronous log streams with matching join keys:

### 6.1 Prediction Log Record Schema (`predictions.jsonl`)
Emitted at inference time $H$ UTC:
```json
{
  "prediction_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "issue_time_utc": "2023-06-15T15:00:00+00:00",
  "interval_start_utc": "2023-06-15T15:00:00+00:00",
  "interval_end_utc": "2023-06-15T16:00:00+00:00",
  "site_id": "2107",
  "model_name": "solarcast-f1",
  "model_version": "1.1.0",
  "feature_schema_version": "1.1.0",
  "predicted_energy_kwh": 384.21,
  "inference_latency_ms": 14.8,
  "created_at_utc": "2023-06-15T15:00:01.025000+00:00"
}
```

### 6.2 Actuals / Ground-Truth Log Record Schema (`actuals.jsonl`)
Emitted when meter interval $[H, H+1\text{h})$ closes and is published:
```json
{
  "site_id": "2107",
  "interval_start_utc": "2023-06-15T15:00:00+00:00",
  "interval_end_utc": "2023-06-15T16:00:00+00:00",
  "measured_kwh": 379.50,
  "observed_at_utc": "2023-06-15T16:15:00+00:00",
  "quality_flag": "VALID"
}
```

### 6.3 Equi-Join Query
The join key for computing out-of-sample residuals is:
$$\text{Join Key} = \text{site\_id} + \text{interval\_start\_utc}$$
$$\text{residual} = \text{actuals.measured\_kwh} - \text{predictions.predicted\_energy\_kwh}$$

---

## 7. Recommended Prometheus Metrics (Low Cardinality)

To avoid cardinality explosion in Prometheus / Grafana, use fixed low-cardinality labels (`site_id`, `status`):

| Metric Name | Type | Labels | Description |
|---|:---:|---|---|
| `solarcast_prediction_requests_total` | Counter | `site_id`, `status` (`success`, `error_400`, `error_503`) | Total incoming inference requests |
| `solarcast_prediction_latency_seconds` | Histogram | `site_id` | End-to-end request processing latency |
| `solarcast_predicted_energy_kwh` | Histogram | `site_id` | Distribution of predicted energy values |
| `solarcast_feature_staleness_seconds` | Gauge | `site_id` | Age of features returned by Feast online store |
| `solarcast_weather_fallback_total` | Counter | `site_id` | Count of predictions served using fallback NWP runs |
| `solarcast_weather_missing_total` | Counter | `site_id` | Count of predictions served with missing NWP weather |
| `solarcast_deviation_alerts_total` | Counter | `site_id`, `direction`, `bin` | Count of production deviation alert events raised |

---

## 8. Production Alert Event Schema

When the deviation engine flags an anomalous episode, an alert event conforming to `configs/alert_event_schema.json` is emitted:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "schema_version": "1.0.0",
  "event_id": "DEV-202306151600-001",
  "site_id": "2107",
  "model_name": "solarcast-f1",
  "model_version": "1.1.0",
  "window_start_utc": "2023-06-15T15:00:00+00:00",
  "window_end_utc": "2023-06-15T16:00:00+00:00",
  "direction": "negative",
  "bin": "clear",
  "consecutive_hours": 2,
  "mean_residual_kwh": -210.45,
  "bound_low": -140.66,
  "bound_high": 127.94,
  "reason": "Production underperformed model forecast by 210.45 kWh over 2 consecutive hours in clear sky conditions",
  "created_at_utc": "2023-06-15T17:15:00.000000+00:00"
}
```
Member 3's service or Member 2's monitoring agent can publish this JSON payload directly to Slack, PagerDuty, or the retraining queue.
