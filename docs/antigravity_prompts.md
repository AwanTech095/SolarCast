# Antigravity prompt sequence (Member 1)

Rules: paste ONE prompt at a time. Wait for it to finish, run the commands it gives you, and only then paste the next one.
If anything fails, paste the full error back to Claude (chat) before moving on. Every prompt assumes `AGENTS.md` and
`docs/HANDOFF.md` are in the repo.

Prefix for every prompt:
> Read AGENTS.md and docs/HANDOFF.md first. Do only this task, then stop.

---
## P1. Scaffold
```
Create the folder layout from AGENTS.md (only the Member 1 paths, with empty __init__.py files in src packages and .gitkeep
in empty folders). Add: .gitignore (data/, .env, mlruns/, .venv/, __pycache__/, .pytest_cache/, .ruff_cache/), .env.example
(names only: MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD), a README.md skeleton (overview, setup,
how to run, tests), pyproject.toml with ruff config (line-length 100), and tests/test_smoke.py. Do not touch folders owned by
Members 2 and 3. Update docs/HANDOFF.md.
```

## P2. Environment
```
Create requirements.txt with pinned versions that work together on Python 3.11 for: pandas, numpy, pyarrow, scikit-learn,
scipy, pandera, pyyaml, requests, boto3, s3fs, joblib, matplotlib, mlflow, pytest, ruff, pvlib. Do NOT add feast, airflow,
kubeflow or bentoml yet. Tell me the exact PowerShell commands to create the venv, install, and run `pytest` and
`ruff check .`. Run them if you can and report the real output. Update docs/HANDOFF.md section 6.
```

## P3. Data feasibility (STOP at the end for my sign-off)
```
Task M1.1. Investigate PVDAQ on the OEDI data lake (s3://oedi-data-lake/pvdaq/, parquet, anonymous access; do NOT use the
decommissioned PVDAQ V3 API). Candidate systems: 2107 first, then 2105 and 9068. Prize datasets are stored differently from
regular systems (one directory per site, possibly split by channel) and can be very large: list files and sizes first and
download only metadata plus ONE small sample (a few days).
For each candidate report: nameplate kW, latitude/longitude, elevation, which channel gives AC power or AC energy, units and
scale/offset, interval length, meter/inverter time_interval alignment (left/center/right), local vs UTC timestamp behavior around
DST, date coverage, gap percentage, total size, and licence/terms.
Then fetch a small NASA POWER hourly sample for the best candidate's coordinates with time-standard=UTC for
ALLSKY_SFC_SW_DWN, T2M, RH2M, WS10M; confirm the parameters exist and note units.
Write docs/data_card.md and a short comparison table. Recommend ONE system with reasons and risks. Do NOT download
the full dataset. Stop and wait for my approval. Update docs/HANDOFF.md (data section + decision log). Report any step that
failed with the real error.
```

## P4. Ingestion (after I approve a site)
```
Task M1.1b. Write src/data/ingestion.py: functions to download the approved site's metadata and production data for a
configurable date range (configs/data.yaml: site_id, system_id, start, end, paths) and NASA POWER hourly weather for the same
coordinates and dates in UTC (request in chunks per year; retry transient errors twice with backoff). Save raw files
unchanged under data/raw/ and write data/raw/manifest.json: URLs, retrieval time, checksums (sha256), units, licence note,
row counts. Make it idempotent (skip existing files whose checksum matches). Tests with mocked HTTP/S3. Run it for a small
date range first and show me the manifest. Update docs/HANDOFF.md.
```

## P5. Validation + hourly aggregation
```
Task M1.2. Write src/data/validation.py (use Pandera schemas) and src/data/aggregation.py. Convert raw production to hourly
kWh: integrate kW over the actual sample duration, or difference cumulative energy with reset detection, depending on the
channel type found in docs/data_card.md. Respect the left/center/right interval alignment. Output
data/processed/hourly.parquet with columns: site_id, interval_start_utc, interval_end_utc, target_kwh, capacity_kw, latitude,
longitude, quality_flag, source_ids. Missing intervals stay NaN (never zero). Quarantine duplicates, impossible values
(negative, above capacity x tolerance), unit errors and big gaps into data/processed/quarantine.parquet with a reason column.
Join weather by UTC hour into a separate weather table with weather_event_time and available_at_utc (= event time + configurable
lag, default 14 days). Write tests/test_data.py with fixtures covering UTC, DST, duplicates, gaps, cumulative-meter reset, and
unit conversion. Print a data quality report (coverage, missing %, quarantined rows). Update docs/HANDOFF.md.
```

## P6. Features
```
Task M1.3a. Write src/features/preprocessing.py with feature functions shared by training and serving: lag_1h, same-hour-
yesterday, shifted rolling means (never including the target hour), hour and day-of-year sin/cos, solar elevation (pvlib),
availability-safe weather features with age and missingness flags, production-freshness flag. Given an issue_time_utc it predicts
the interval [issue_time, issue_time+1h) using only data published before issue_time. Write configs/feature_schema.json
(ordered names, dtypes, units, schema_version). Write tests/test_features.py that proves no feature uses information from the
target hour or later (e.g. perturb future rows and check features do not change) and that weather features respect available_at_utc.
Update docs/HANDOFF.md.
```

## P7. Baselines, tuning, evaluation, training
```
Task M1.4. Implement src/models/baselines.py (persistence: previous hour and same hour previous day, with explicit missing-history
fallbacks), src/models/tune.py, src/models/evaluate.py, src/models/train.py. Models: Ridge and HistGradientBoostingRegressor,
10 random-search configurations each, evaluated on 3 expanding-window chronological folds; final 20% test set opened once after
selection. Metrics: daylight MAE (solar elevation > 0), all-hour MAE, RMSE, capacity-normalized MAE, per-month slices. Candidate
gate (thresholds in configs/evaluation.yaml): at least 5% daylight-MAE improvement over the stronger baseline and normalized
daylight MAE <= 0.20 and finite outputs. Fixed seeds. Entrypoint: python -m src.models.train --config configs/evaluation.yaml
(also accept --data-path and --output-dir so Member 2 can call it from a pipeline). Run a 7/14/30-day weather-lag sensitivity.
Write tests/test_model.py (output shape, finite predictions, metric threshold, regression test vs a stored reference metric). Show me
the results table. Update docs/HANDOFF.md.
```

## P8. MLflow, evaluation.json, model bundle
```
Task M1.5. Add MLflow logging to train.py. Tracking URI from MLFLOW_TRACKING_URI, else local mlruns/. Each run logs: model type,
params, data hash (sha256 of hourly.parquet and the manifest), Git commit, feature schema version, metrics, TRAINING DURATION,
plots (actual vs predicted, residuals, error by hour/month) and artifacts. Register the selected model in the MLflow Model
Registry with a status tag (candidate/approved/rejected) set by the gate. Write evaluation.json: baseline vs candidate metrics,
thresholds, eligible flag, failure reasons. Write src/models/bundle.py to save one bundle (preprocessing + estimator, mode F1,
ordered feature schema, training window, data/Git/feature versions, MLflow run id, hashes) and a loader. Write docs/model_card.md
(limitations, one-site caveat, coverage by season, fairness note). Update docs/HANDOFF.md section 5 and 6.
```

## P9. Feast
```
Task M1.3b. First add feast to requirements in a way that does not break the rest of the stack (tell me if it conflicts). Create
feature_store/feature_repo/{feature_store.yaml, entities.py, features.py, feature_views.py} and feature_store/materialize.py.
Entity: site_id. Event timestamp: issue-state timestamp; keep source_event_time and available_at_utc columns for audit.
Offline store: Parquet files. Online store: SQLite. Document feature names, dtypes, source, transformation, freshness expectation
and owner in docs/feature_store.md. Demonstrate: historical retrieval, materialization, online query. Tests in
tests/test_feature_store.py: schema, entity key, event timestamps, freshness, nulls, offline/online parity (training-serving skew),
stale-feature rejection. Record the feature repo Git commit in the MLflow run. Remember the online store holds only the latest
values: replay must use offline point-in-time retrieval. Update docs/HANDOFF.md.
```

## P10. E1 estimator + deviation rule
```
Task M1.6. Add src/models/deviation.py: an E1 weather-to-output regression (target-hour observed irradiance and temperature,
calendar) logged as its own MLflow run. Calibrate residual 1st/99th percentile bounds per irradiance bin on a reserved
chronological calibration period; flag a deviation after 2 consecutive valid daylight hours outside bounds; suppress nights,
missing labels and invalid weather. Create labeled replay copies with injected 30% drops, zero-output daytime outages and bad
sensor values; report detection rate, detection delay and false-alert rate on untouched hours. Injected data must never enter
the final accuracy report. Tests included. Update docs/HANDOFF.md.
```

## P11. Handoffs and fixtures
```
Task M1.7. Create tests/fixtures/ with a small deterministic dataset (about 30 days) so other members can run everything without
the real data. Write docs/handoff_member2.md (data manifest schema, module interfaces, training entrypoint command and arguments,
fixture location, MLflow metadata keys) and docs/handoff_member3.md (bundle format and loader, ordered feature schema, unit/range
rules, Feast entity contract, 5 sample inputs and outputs, prediction/actual log fields). Final update of docs/HANDOFF.md:
status board, how-to-run, known issues, next steps.
```
