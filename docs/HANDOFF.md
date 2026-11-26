# SolarCast: Member 1 Handoff (living document)

**Owner:** Emad (Member 1: Data, features, Feast, models, MLflow). **Review partner:** Member 2.
**Last updated:** 2026-11-27 (raw meter validation, step 7). Update this file after every task (see AGENTS.md).
Anyone picking up this package should be able to continue from this file alone.

## 1. What this project is
Forecast the next hourly AC energy output (kWh) for **one** PV site (mode **F1**) using archived forecast irradiance,
temperature, production history and time features. Compare actual production with the stored F1 forecast to flag
deviations after labels arrive. NASA observed weather supports a separate retrospective benchmark (mode **E1**).
NOAA HRRR is proposed, not yet verified; follow the Member 1 verification gate in README.md before full training.
Demonstrate through chronological historical replay, not a live plant connection. Course goal: a reproducible, tested, deployed and monitored MLOps
system. Accuracy is secondary. Source documents: `MLOps_Course_Project_Requirements.docx` (13 pp.) and
`SolarCast_Project_and_Team_Plan.pdf` (14 pp.). **Where they conflict, the course requirements win.**
Known spec quirk: the example repo tree mentions FastAPI, but the endpoint section requires **Flask**. Keep Flask.

## 2. Status board
| ID | Item | Status | Notes |
|---|---|---|---|
| M1.review | Independent Member 2 integration acceptance | In progress | R1–R6 repairs tested. R7 P95 provenance, R8 weather availability assumptions, and environment/service verification remain. |
| M1.raw-validation | Raw meter quarantine (R6) | Done (tests pass) | Invalid values and timestamps quarantine before aggregation. Original power text is available as raw_power. Core suite 217 passed, 1 skipped, 1 deselected. |
| M1.evaluation-safeguards | Explicit calibration/monitoring modes, immutable final-test protection and corrected commands (step 6) | Done (tests pass) | Bare deviation CLI does no work. Model-bound dev-only purged OOF calibration and bounded stored-record monitoring write isolated outputs. Existing final audit cannot be forced or overwritten; repository-anchored guard runs before I/O. 26 new tests; 215 core tests passed, 1 skipped, 1 deselected. No real final-test evaluation or artifact replacement. |
| M1.alert-continuity | Consecutive-hour alerts, incremental confirmation and timing (step 5) | Done (tests pass) | Gaps/duplicates/suppressed intervals reset persistence; incremental detector emits once at second adjacent breach, stable context-based IDs, explicit label timing or null. 22 new regressions; full suite 189 passed, 1 skipped, 1 deselected. Historical performance not recomputed; durable live service not implemented. |
| M1.exact-feature-hour | Exact issue-hour identity in online/offline retrieval (step 4) | Done (tests pass) | All online feature timestamps must match requested time; offline checks exact registered-source keys and returned values. Missing night/hour rows reject, source time returned for audit. 23 new core tests; 167 passed, 1 skipped, 1 deselected overall. Real Feast integration not run. |
| M1.candidate-artifacts | Isolate candidate outputs and evaluate saved weights (step 3) | Done (tests pass) | Unique candidate directories, SHA-256 manifest, reloaded-model development validation, rejected exit 2, no rejected registration or approved-file overwrite. 17 new regression cases; full non-network suite 144 passed, 1 skipped, 1 deselected. No real-data retraining or deployment. |
| M1.packet | Receive and verify artifact packet (step 1) | Done (tests pass) | SolarCast_packet.zip: 31 payload hashes/sizes verified; 28 missing data/model files installed; three existing config/report copies match. Bundle data hashes and MLflow run ID checked; 64 tests passed, 2 skipped under the existing Python 3.12 review runtime. Separate Feast/Python 3.11 and bug fixes remain. Receipt: reports/packet_receipt_20261126.json. |
| M1.gate-validation | Fail-closed numerical quality gate (step 2) | Done (tests pass) | Confirmed latest main still accepted NaN metrics. Fixed evaluate_candidate_gate and added 44 regression cases: missing/nonfinite/negative metrics, zero baseline, invalid/empty predictions, invalid policy and exact threshold behavior. Full non-network suite: 127 passed, 1 skipped, 1 deselected; Ruff passed with local tools excluded. Does not complete candidate artifact/promotion integration. |
| M1.0 | Repo + environment setup | Done (tests pass) | Scaffold created on branch m1/scaffold; Member 1 folder layout, .gitkeep, __init__.py, pyproject.toml, .env.example, pinned requirements.txt, and tests passing (pytest & ruff clean) |
| M1.1 | Data feasibility and site choice | Done (tests pass) | Site 2107 approved by human owner. Deep verification audit completed in docs/data_card.md. |
| M1.1b | HRRR verification gate | Done (tests pass) | Full feasibility audit completed in docs/data_feasibility.md, scripts/step1_inventory.py, scripts/step2_coverage_scan.py, scripts/step3_step4_mixed_and_signal.py, and docs/tcdc_clearsky_proxy_correlation.png. Audited spatial mapping (334 m, chunk 1.4); instantaneous semantics mapped via trapezoidal averaging; operational latency verified with 2-h lag. Inventoried chunk 1.4 variables (TCDC, LCDC, MCDC, HCDC, TMP, DPT, RH, UGRD, VGRD, VIS, PRATE, APCP valid; DSWRF missing chunks due to iris-grib #265 bug). Completed 46,728-run full scan (2018-07-12 to 2023-11-09) showing 94.18% TMP and 94.07% TCDC availability. Selected optimal mixed week (2022-12-26 to 2023-01-01, atmospheric river storm, 100% valid, maximal kt_proxy std=0.400). Signal check on VALID high_sun hours demonstrated strong true forecast skill: Spearman rho = -0.7884 (p=5.46e-13), Pearson r = -0.9044 against clear-sky index proxy (distinguished from astronomical diurnal cycle). GRIB2 decoding verified in-memory with eccodes on Windows Python 3.11 and WSL2 confirmed; volume estimated at ~34.8 GB/yr (~185.7 GB full period). Gate decision: APPROVE ROUTE 1 (Hybrid Forecast TMP + Cloud Cover + pvlib Clear-Sky Irradiance for Mode F1; retain NASA POWER for retrospective Mode E1). |
| M1.2 | Validation + hourly dataset | Done (tests pass) | Implemented src/data/validation.py (Pandera) and src/data/aggregation.py. Quarantined 1 row > 893 kW (2018-02-18 12:30) into data/processed/quarantine.parquet. Ambiguous fall DST times mapped to 1st occurrence (PDT) with 2nd occurrence marked missing. Aggregated 60,096 continuous UTC hours (60,013 valid, 83 NaN = 0.1381%) into data/processed/hourly.parquet with pvlib solar elevation and quality flags. All 21 tests pass; ruff clean. |
| M1.3 | Features (zero-leakage, 23 past-only) | Done (tests pass) | Built continuous data/processed/quarter_hourly.parquet (240,384 rows) in src/data/aggregation.py. Implemented ONE shared build_features() function in src/features/preprocessing.py with configs/features.yaml and configs/feature_schema.json (23 features). Strict 15-minute operational cutoff enforced. Leakage-free behavior verified with perturbation tests, DST fall-back, missing hour fallback, and prehistory tests in tests/test_features.py. |
| M1.4 | Baselines & 3-fold chronological evaluation | Done (tests pass) | Implemented 5 baselines in src/models/baselines.py: (a) Latest Hour Persistence, (b) Same-Hour Previous Day, (c) Clear-Sky Persistence (hourly), (d) Last 15-Minute Persistence, and (e) Clear-Sky Scaled 15-Minute Persistence. Constructed configs/evaluation.yaml with 3 expanding-window chronological validation folds on Train/Dev (37,090 hours) while strictly preserving final 20% test set (9,273 hours) sealed. Champion is Clear-Sky Scaled 15-Minute Persistence (42.18 kWh Daylight MAE / 4.72% cap-MAE / 67.61 kWh RMSE). Generated docs/baseline_report.md including 0m, 15m, and 60m delay sensitivity. Full test suite (46 tests) passes; ruff clean. |
| M1.5 | MLflow lineage, evaluation.json, model bundle, Set A vs B benchmark, one-time final test | Done (tests pass) | Implemented Feature Set B (29 features including NOAA HRRR forecast TMP, TCDC, Kasten-Czeplak cloud-adjusted clear-sky GHI, run age, fallback, missingness). Sanitized out-of-range weather values (-10000 fill value in July 2020) to NaN with weather_missing flag. Benchmarked Set A vs Set B across 3 expanding folds on identical validation hours (N = 13,374): HistGradientBoosting Set B won with 31.43 kWh Daylight MAE (3.52% cap-MAE) and 53.35 kWh RMSE, achieving +2.52% improvement over Set A (32.24 kWh) and +25.33% improvement over baseline champion (42.09 kWh). Candidate gate passed (reports/evaluation.json and ./evaluation.json). Analyzed cloud cover bins (<20% low cloud: +5.59% gain; >80% overcast: +4.94% gain; 20-80% mixed cloud: -9.70% degradation due to NWP spatial/temporal phase timing errors). Computed permutation importance on validation folds (top: power_kw_lag_0m +144.04 kWh, hour_sin +51.94 kWh, energy_kwh_prev_day_same_hour +8.26 kWh, clearsky_ghi_cloud_adj_wm2 +4.92 kWh, tcdc_pct +3.29 kWh). Exported models/solarcast_f1_bundle.joblib with feature_set='B', needs_weather=True, schema 1.1.0. Logged runs to MLflow and registered solarcast-f1 v4. Executed authorized one-time final test evaluation on the 20% held-out test partition (2022-10-17 to 2023-11-09 UTC, 9,273 valid hours, 62 excluded) under strict safeguard logging (reports/final_test_audit.json). Champion HistGradientBoosting Set B confirmed decisive out-of-sample skill: Daylight MAE 32.69 kWh (3.66% cap-MAE) on primary daylight hours (+24.83% vs baseline champ, +3.84% vs Set A) and 33.48 kWh on all daylight hours (+24.33% vs baseline champ). Generalization gap was minimal (+1.26 kWh, +4.01%). Authored docs/final_test_report.md, logged MLflow run final_test_evaluation, updated evaluation.json and docs/model_card.md. All 53 tests pass; ruff clean. |
| M1.5b | Feast feature store integration (rubric 6 marks) | Done (tests pass) | Implemented isolated Feast 0.68.0 environment (.venv-feast) to resolve numpy 2 dependency conflict without altering core requirements.txt. Built shared feature table data/processed/feast/features_set_b.parquet (23,532 rows, 29 features + 3 audit columns, zero leakage verified). Defined feature repo feature_store/feature_repo/{feature_store.yaml, entities.py, features.py, feature_views.py} with SQLite online store and Parquet offline store. Implemented materialization pipeline feature_store/materialize.py with MLflow logging (run feature_store_materialization). Implemented src/features/feast_provider.py delivering zero-skew point-in-time historical retrieval and online feature retrieval strictly enforcing future and staleness rejection. Executed all 4 real-data demos in scripts/demo_feast.py. Comprehensive test suite in tests/test_feature_store.py (9/9 pass in .venv-feast; auto-skips cleanly in .venv); documented architecture in docs/feature_store.md; ruff clean. |
| M1.6 | Forecast deviations & alerting engine | Done (tests pass) | Implemented src/models/deviation.py, configs/deviation.yaml, and configs/alert_event_schema.json. Calibrated P1/P99 thresholds on 13,914 VALID daylight hours across 3 validation folds (clear: [-140.66, 127.94], mixed: [-231.18, 226.24], overcast: [-203.70, 216.30], no_weather: [-155.80, 191.86] kWh). Enforces 2-consecutive-hour persistence rule with night/missing/non-VALID reset. Injected fault benchmark: zero output 95% detected, bad values 90%, 30% drop 60%, isolated 1-h drop 0% (100% compliance). Unmodified test period baseline: 0.43 false alerts per 1,000h. Natural SUSPECT_ZERO evaluation: 6/43 detected (13.95%, median delay 3.0h; non-detected were sunrise/sunset low-power hours). Logged MLflow run deviation_detector. Authored docs/deviation_detector.md. All 8 tests pass in tests/test_deviation.py; ruff clean. |
| M1.7 | Handoff docs for Members 2 and 3 | Done (tests pass) | Authored docs/handoff_member2.md (DVC data snapshot tables, CI/CD runner setup, training/eval runbook, MLflow run lineage, evaluation.json schema, retraining DAG trigger with cooldown, dual-venv setup), docs/handoff_member3.md (model bundle spec, scripts/gen_feature_doc.py generated Schema 1.1.0 table, request-to-prediction replay flow, SQLite replica snapshot instructions, prediction/actuals join log schema, Prometheus metrics, alert schema), docs/samples/{predict_requests.json, predict_responses.json} (5 real examples including missing weather and stale rejection), tests/fixtures/features_sample.csv (48 rows, 24.5 KB, with public data attribution), tests/test_contract_samples.py (verifying bundle reproduces samples within 1e-6), docs/member1_defense_checklist.md (7 core demo explanations, commands, and expected outputs, plus 5 known limitations), and updated docs/deviation_detector.md. All 65 tests pass; ruff clean. |
| M1.8 | Monitoring signals, input drift, delayed-label performance & weather quality | Done (tests pass) | Implemented src/models/drift.py (8-feature 10-quantile PSI against season-matched reference, 2-consecutive-check rule, 200-row floor, weather freshness classifier), src/models/performance_monitor.py (virtual-clock delayed label join with 15m delay, 7-day daylight MAE > 1.25x reference with 30-label floor and data outage distinction; froze month-specific reference from validation evidence over fixed reference, cutting false alerts by >50%), configs/monitoring.yaml, updated configs/alert_event_schema.json to v1.1.0 with alert_type, scripts/replay_monitoring.py running 375-day chronological replay across 4 scenarios (weather shift detected in 4.0d, performance drop in 3.0d, label outage correctly suppressed with 0 accuracy alerts), logged MLflow run monitoring_replay (16ffd7429be642928941e95f932fe8b3), authored docs/monitoring_signals.md, and added tests/test_monitoring.py (12 tests). All 77 tests pass; ruff clean. |
| M1.9 | Artifact packet generation & verifier | Done (tests pass) | Built scripts/make_artifact_packet.py and scripts/verify_artifact_packet.py. Generated clean packet D:\SolarCast_packet and D:\SolarCast_packet.zip (31 payload files, 26.81 MB uncompressed, 12.91 MB deflated). Cryptographic verification via CHECKSUMS.sha256, PACKET_MANIFEST.json with pipeline commands and read-only MLflow lineage audit. Corrupted copy test verified failure. |
| M1.10 | MLflow lineage resolution & Model Registry | Done (tests pass) | Built stdlib-only src/models/lineage.py (log_lineage) logging git commit, git dirty status, data file hashes, manifest hash, feature schema version & hash, configuration hashes, and bundle sha256 across .venv and .venv-feast. Retroactively tagged historical runs with verified values and explicit unverified flags; documented champion working tree state (uncommitted changes at run time committed in 0afc5c6). Configured Model Registry alias champion -> solarcast-f1 v4, version tags (mode=F1, feature_set=B, bundle_sha256, source_run_id), status=superseded on v1-v3, and factual descriptions. Implemented scripts/resolve_run.py resolving runs to exact code, data, schema, and artifact hashes with [OK]/[MISMATCH]/[NOT RECORDED] verification against disk. Added @pytest.mark.network to live S3 regression test; full suite (83 tests) passes in 33.5s; ruff clean. |

Status values: Not started / In progress / Done (tests pass) / Blocked (say why).

## 3. Data sources (audited in M1.1; see docs/data_card.md for full report)
Production data was audited on 2026-11-16 via anonymous S3 access to `s3://oedi-data-lake/` and approved (System 2107).
Under the 2026-11-21 scope revision, NOAA HRRR archived forecasts are the candidate core forecast weather source
(pending the M1.1b verification gate in README.md), while NASA POWER supports retrospective analysis and the E1 benchmark.

**Target (measured production): NLR/NREL PVDAQ via OEDI data lake.**
- Citation / DOI: 10.25984/1846021 (DOE Solar Data Prize 2023 / NREL).
- Anonymous S3 access: `s3://oedi-data-lake/pvdaq/2023-solar-data-prize/{site}_OEDI/`.
- Audited candidates:
  - **2107 (Approved):** 893 kW DC fixed ground-mount, Arbuckle, CA ($38.9963^\circ\text{N}, -122.1341^\circ\text{W}$, 10 m elev). Dedicated revenue meter `2107_meter_15m_data.csv` (6.32 MB), 15-min instantaneous kW (interval average), left-aligned (`time_interval = 'L'`), 0.13% missingness over 7 years (2017–2023). Timestamps in local civil time (`PST8PDT`) with DST spring skip (must localize to `America/Los_Angeles` before UTC conversion).
  - **2105 (Backup):** 110 kW DC rooftop, Maui, HI. No single meter file; split across 11 inverter CSVs reporting cumulative kWh (requires differencing + reset handling). Mixed tilts/azimuths across roof planes. HST (UTC-10, no DST).
  - **9068 (Backup):** 4.74 MW DC single-axis tracker, Kersey, CO (1407 m elev). Single-axis tracker with 285 tracker angles. Inverter clipping near 3.82 MW AC limit. Massive dataset (2.0 GB across 15 files).
- Note: The regular `pvdaq/parquet/` metadata tables contain standard systems, while prize datasets reside under `pvdaq/2023-solar-data-prize/`. Old PVDAQ V3 API is decommissioned.

**Weather: NASA POWER hourly API (supporting retrospective analysis & E1 benchmark).**
- API: `/api/temporal/hourly/point` at https://power.larc.nasa.gov/ (tested and verified with `time-standard=UTC`).
- Confirmed parameters & units for 2107 coordinates:
  - `ALLSKY_SFC_SW_DWN`: All Sky Surface Shortwave Downward Irradiance (`Wh/m^2`, hourly insolation)
  - `T2M`: Temperature at 2 Meters (`C`)
  - `RH2M`: Relative Humidity at 2 Meters (`%`)
  - `WS10M`: Wind Speed at 10 Meters (`m/s`)
- Licence: CC BY 4.0 (cite NASA POWER).
- Update lag: meteorological ~2-3 days, solar ~5-7 days. It is an archive of observations, **not an advance forecast**.
- Scope update: The earlier plan used a simulated 14-day lag for F1. That is superseded as the core forecast-data route;
  NASA POWER now supports retrospective analysis and a separately evaluated E1 benchmark.

**Forecast weather: NOAA HRRR archive (core forecast candidate; pending M1.1b verification gate).**
- Archive: https://registry.opendata.aws/noaa-hrrr-pds/. Retrieve only a bounded plant/date sample first.
- Verify irradiance/downward shortwave radiation and near-surface temperature, units, valid intervals,
  spatial extraction and overlap with production. Initialization time is not publication time.
- Require forecast-field availability no later than the prediction issue time. Distinguish verified release
  evidence from a simulated latency assumption; flag/exclude unresolved cases. Never use later runs or analyses.
- See README.md, "Special note for Member 1", for the ten checks, sample schema and report deliverables.

**Fallback if no usable measured site is found:** another public measured-generation source with known coordinates and terms.
Synthetic data is allowed only as a clearly labeled pipeline-demo fallback.

**Honest expectation:** forecast-weather skill and data feasibility are not established. Compare weather-enhanced
and production/time-only models on the same eligible timestamps. The proposed 5% improvement gate may fail;
report actual results and retain the eligible baseline/champion. Do not promise accuracy or a course grade.

## 4. Key design decisions (from the team plan; proposed values, not course-mandated)
- F1 = weather-driven next-hour forecast using eligible archived forecasts (HRRR proposed). Archived forecasts
  are now in the proposed core scope, subject to verification. E1 = separate retrospective expected-output benchmark.
- Split: last 20% held out as a final test set. Earlier 80%: 3 expanding-window folds. Test is opened once.
- Primary metric: daylight MAE (kWh). Candidate gate: >=5% better than the stronger persistence baseline and
  capacity-normalized daylight MAE <= 0.20. Retraining gate: >=2% better than champion and no seasonal slice >5% worse.
- Core deviation rule: residual = actual - stored F1 prediction. Calibrate bounds on earlier chronological
  validation data and freeze before testing; investigate forecast error as well as plant behavior. Keep the
  older irradiance-bin E1 residual rule separate if implemented. No verified equipment-fault claims.
- Required models: persistence (previous hour, same hour previous day), Ridge, HistGradientBoostingRegressor. No LSTM needed.
- Tools for Member 1 in the spec: Pandera or Great Expectations (validation, optional), Feast (required, rubric 6 marks),
  MLflow (9 marks, shared with registry), Evidently or a custom drift detector (PSI >0.20 in >=2 features is the proposed rule).

## 5. Contracts I own (fill in when built; link the file)
| Contract | File | Status |
|---|---|---|
| Hourly data schema + manifest | `configs/data.yaml`, `configs/data_manifest.json`, `docs/data_quality_report.md` | Built (quarantine.parquet, hourly.parquet [12 cols incl. high_sun & midpoint solar elevation], data_manifest.json) |
| Test sample fixture (teammates) | `tests/fixtures/hourly_sample.csv`, `tests/fixtures/README.md` | Built (720 rows, July 2022 excerpt, <200 KB, DOI 10.25984/1846021) |
| HRRR feasibility report + preview schema | `docs/data_feasibility.md`, `docs/data_feasibility_preview.csv`, `scripts/hrrr_feasibility.py` | Built (audited spatial mapping, GRIB2 semantics, 2-h release lag, Zarr iris-grib #265 bug, multi-year coverage, correlations r=0.945/0.927) |
| Feature schema (ordered names/types/units) | `configs/feature_schema.json`, `configs/features.yaml` | Built (Schema 1.1.0: 23 features in Set A, 29 features in Set B with HRRR weather) |
| Weather data quality report | `docs/weather_data_quality.md` | Built (join coverage 95.91%, out-of-range sanitization for -10000 fill value) |
| Set A vs Set B comparison | `scripts/compare_set_a_b.py`, `reports/set_a_vs_b_summary.json`, `docs/model_report.md` | Built (Set B beats Set A by +2.52% MAE, beats baseline champion by +25.33%; cloud-bin and monthly breakdowns) |
| Baseline report & evaluation config | `docs/baseline_report.md`, `configs/evaluation.yaml` | Built (3-fold expanding-window cross-validation, Clear-Sky Scaled 15-Min Persistence champion, locked 20% test set) |
| Model bundle format | `src/models/bundle.py`, `models/solarcast_f1_bundle.joblib` | Built (`ModelBundle` with `feature_set='B'`, `needs_weather=True`, `feature_schema_version='1.1.0'`, bounded `.predict()`) |
| Evaluation result | `reports/evaluation.json`, `./evaluation.json` | Built (HistGradientBoosting Set B champion vs Clear-Sky Scaled 15m Persistence: 31.43 vs 42.09 kWh on val CV, 32.69 vs 43.48 kWh on final test; eligible: true, final_test_opened: true) |
| Model card & documentation | `docs/model_card.md`, `docs/model_report.md` | Built (full documentation of Set A vs B benchmark, limitations, cloud cover bin degradation, and MLflow lineage) |
| Final test evaluation report & audit | `docs/final_test_report.md`, `reports/final_test_summary.json`, `reports/final_test_audit.json` | Built (One-time test evaluation: HGB Set B achieves 32.69 kWh Daylight MAE [3.66% cap], +24.83% vs baseline champ, +3.84% vs Set A; generalization gap +1.26 kWh / 4.01%; audited in reports/final_test_audit.json) |
| Feast entity/feature views & architecture doc | `feature_store/feature_repo/`, `feature_store/materialize.py`, `src/features/feast_provider.py`, `docs/feature_store.md` | Built (`solarcast_f1_set_b` view over `features_set_b.parquet`, SQLite online store, zero-skew provider with future/stale rejection, MLflow run `feature_store_materialization`; complete architecture and run guide in `docs/feature_store.md`) |
| Deviation thresholds & detector | `src/models/deviation.py`, `configs/deviation.yaml`, `configs/alert_event_schema.json`, `docs/deviation_detector.md` | Built (2-hour persistence rule, cloud-bin P1/P99 thresholds, MLflow run `deviation_detector`; full report in `docs/deviation_detector.md`) |
| Training entrypoint | `python -m src.models.train --config configs/evaluation.yaml` | Built (full pipeline: tuning Set A & B, benchmarking, gating, full dev refit, plots, MLflow tracking, bundle) |
| Member 2 Handoff Specification | `docs/handoff_member2.md` | Built (DVC tracking paths/hashes, training/eval runbook, MLflow run lineage, evaluation.json schema, retraining DAG trigger, clean-clone checklist) |
| Member 3 Handoff Specification | `docs/handoff_member3.md` | Built (Model bundle contract, Schema 1.1.0 table via scripts/gen_feature_doc.py, request-to-prediction flow, SQLite replica snapshot, join logs, Prometheus metrics) |
| Real Prediction Sample Contracts | `docs/samples/predict_requests.json`, `docs/samples/predict_responses.json` | Built (5 real pipeline examples including clear daylight, noon peak, nighttime 0.0 kWh, missing weather, and stale rejection HTTP 503) |
| Feature Set B Sample Fixture & Contract Test | `tests/fixtures/features_sample.csv`, `tests/test_contract_samples.py` | Built (48 daylight rows, 24.5 KB, public data attribution; unit tests verify bundle reproduces sample predictions within 1e-6) |
| Member 1 Defense Checklist | `docs/member1_defense_checklist.md` | Built (7 core demonstration explanations, exact tested PowerShell commands, expected outputs, and 5 known limitations) |
| Feature Schema Markdown Generator | `scripts/gen_feature_doc.py` | Built (Generates Schema 1.1.0 Markdown table directly from configs/feature_schema.json) |
| Artifact Packet & Verifier | `scripts/make_artifact_packet.py`, `scripts/verify_artifact_packet.py`, `D:\SolarCast_packet`, `D:\SolarCast_packet.zip` | Built (31 payload files, 26.81 MB uncompressed / 12.91 MB deflated, SHA-256 verified, read-only MLflow lineage audit, PACKET_README.md, install CLI) |
| MLflow Lineage & Provenance Resolver | `src/models/lineage.py`, `docs/mlflow_lineage.md`, `scripts/resolve_run.py`, `reports/model_registry_status.json` | Built (Resolves runs or champion alias to code commit, dirty status, data/manifest hashes, schema version/hash, bundle hash; reports [OK]/[MISMATCH]/[NOT RECORDED]) |

## 6. How to run (fill in as built)

### Raw meter quarantine

Missing required columns fail with a clear error. Power is coerced once, and invalid text,
missing values and infinities use the existing `non_finite` reason. Quarantine records
include `raw_power` alongside numeric `power_kw`. Invalid or offset-bearing timestamps
use `invalid_timestamp` when power is otherwise valid. Local civil times retain the
existing timezone conversion policy. Duplicate detection keeps the first usable reading.
An end-to-end run on the supplied site 2107 meter CSV used temporary outputs and produced
240,061 accepted readings, one quarantine row, 60,096 hourly rows and 240,384 quarter-hourly rows.

```powershell
.\tmp\review_env\Scripts\python -m pytest tests/test_raw_meter_validation.py tests/test_data.py -m "not network" -q
.\tmp\review_env\Scripts\python -m pytest -m "not network" -q
.\tmp\review_env\Scripts\python -m ruff check . --exclude tools
```

### Evaluation/runbook safeguards (27 November 2026, step 6 only)

Branch `m1/evaluation-safeguards`, review R5. Eight initial regression tests failed against the original
entrypoints: bare deviation CLI performed implicit calibration/final evaluation, final-data helpers read
without protection, and force/audit overwrite could replace one-time evidence. These paths are corrected:

- Final evaluation checks authorization before configs/data/model I/O. Any existing audit (including a
  partial/failed claim) blocks repeat evaluation, even with `--force --reason`. The default marker is
  anchored to this repository, not the caller's working directory. First-ever use requires force/reason
  outside known CI/scheduler contexts; exclusive audit creation is the atomic access claim. Failed runs
  retain it. These are application guards, not a filesystem security boundary; never remove/change the audit.
- The default deviation command now fails with usage error. `--calibrate` and `--evaluate` are mutually
  exclusive and require an explicit model bundle. Irrelevant monitoring/window flags cannot be silently
  ignored during calibration. `--reports-dir` and all current runbook flags are implemented.
- Calibration clones the supplied estimator recipe across three development-only expanding folds, purges
  unavailable training labels, and verifies site/hour uniqueness, model hourly-data lineage and feature
  availability audits. No final targets enter fitting, residual calculation or thresholds. These are OOF
  recipe forecasts, not predictions from different final saved weights. Configs record model/data/features/
  config hashes, source run ID, code commit, fold windows and protected calendar. Observed bin counts are
  recorded; absent/degenerate bins are not silently calibrated. Outputs remain unapproved experiment artifacts.
- Monitoring reads explicitly supplied stored-forecast/actual records in a bounded UTC window at a virtual
  clock. It never loads raw target datasets, refits or recalibrates. It checks site/model/threshold identity,
  excludes unarrived labels and rejects the historical final-test period and issues before training labels
  were available. No visible labels yields `no_visible_labels`, not a claim of healthy accuracy.
- Outputs go to fresh `reports/deviation_runs/<id>/` directories (Git-ignored). Neither mode overwrites
  approved thresholds, bundles or historical reports. Legacy unbound threshold YAML fails in this workflow;
  direct retrospective detector/contract examples retain their old config compatibility.
- The two old implicit final/natural-benchmark readers are retired and fail before I/O. This also retires
  the misleading all-years future-trained-model natural comparison; no corrected natural fault score is
  invented. Packet metadata marks historical model/evaluation/audit files as restore-only, with null
  regeneration commands. Import their verified bytes rather than retraining/reopening them.

Complete calibration/monitoring commands and input columns are in `docs/handoff_member2.md` section 2.7.
The monitor template requires joined Member 3 records; post-holdout observations are not in the packet.
For a self-contained synthetic end-to-end proof, run:

```powershell
$env:PYTHONPATH=''
.\tmp\review_env\Scripts\python -m pytest tests/test_evaluation_safeguards.py tests/test_deviation_workflow.py -q
.\tmp\review_env\Scripts\python -m src.models.deviation --help
.\tmp\review_env\Scripts\python -m pytest -m "not network" -q --tb=short
$env:CI='true'
.\tmp\review_env\Scripts\python -m pytest -m "not network" -q --tb=short
Remove-Item Env:CI
.\tmp\review_env\Scripts\python -m ruff check . --exclude tools
```

Observed core suite, normally and with `CI=true`: **215 passed, 1 skipped (Feast), 1 deselected (network)**.
All 31 installed packet artifact hashes remain unchanged. New tests exercise both real
CLI handlers with temporary synthetic Parquet/bundle inputs, exclusive-write failures, final/test-calendar
rejections, data/model/site identity, delayed labels, chronology and immutable outputs. The `--help` command
also ran through `python -m src.models.deviation`. Logs: `tmp/evaluation_safeguards_before.txt` (8 failures),
`tmp/evaluation_safeguards_focused.txt`, `tmp/evaluation_safeguards_full_suite.txt`,
`tmp/evaluation_safeguards_ci_suite.txt`, `tmp/deviation_cli_help.txt`.

Runtime remains the Python 3.12.5 review environment. Clean Python 3.11, real Feast and a shared MLflow
server are not verified. No new dependencies, real-data retraining/recalibration, final-test benchmark,
threshold promotion, historical packet regeneration or GitHub push. By eye, confirm the selected bundle
hash matches both thresholds and stored forecasts, and compare actual label arrival with the virtual clock.
Next: R6 raw-value quarantine; R7/R8 evidence limitations remain explicit. Stop after this task.

### Alert continuity (26 November 2026, step 5 only)

Branch: `m1/alert-continuity`, review R4. Initial regression suite: **17 failed, 1 passed** against the
old detector. It incorrectly treated 16:00 and 19:00, or repeated 16:00 rows, as consecutive breaches.
The fix requires exactly one hour between eligible hourly target intervals in the same direction.
Gaps, all copies of duplicated within-batch hours, off-hour timestamps, night/invalid/missing/nonfinite
measurements and direction changes reset pending persistence. Frozen calibrated bounds are unchanged.

`detect_deviations(frame)` remains the complete-batch retrospective summary API. For incremental use,
keep one `DeviationDetector(config, site_id, model_name, model_version)` instance and call `process(frame)`
on strictly later, nonoverlapping batches. It emits one independent snapshot when the second adjacent
breach confirms an episode, without waiting for closure. Further breaches extend `summaries()` but do not
emit again. Overlapping/late batches reject before modifying state. State is in memory, not a deployed
live monitor: late-label buffering, durable recovery and delivery deduplication remain service work.

Episode IDs hash site/model/version, start/direction and frozen bounds, independent of prior unrelated
episodes or wall-clock record creation. A truncated replay without the original episode start cannot
reconstruct that episode's ID. Consumers must preserve complete episode context when replaying.

Optional timing fields were added to `configs/alert_event_schema.json` without changing existing required
fields. `confirmation_issue_time_utc` and target interval start/end identify the second breach. Real
`actual_available_at_utc` inputs establish `triggered_at_utc` (when both initial labels have arrived) and
`detection_delay_hours` (from first anomalous target start). Without both timestamps, trigger/delay is null.
The old 1-hour difference between target starts is not actual label-based latency. `created_at_utc` is
record creation time, while legacy window endpoints remain anomalous target STARTS. See updated
`docs/deviation_detector.md` and `docs/handoff_member3.md` for state and payload contracts.

```powershell
$env:PYTHONPATH=''
.\tmp\review_env\Scripts\python -m pytest tests/test_deviation.py tests/test_deviation_continuity.py -q
.\tmp\review_env\Scripts\python -m pytest -m "not network" -q --tb=short
.\tmp\review_env\Scripts\python -m ruff check . --exclude tools
```

Verification: **30 focused tests passed**; full suite **189 passed, 1 skipped (Feast), 1 deselected
(network)**; scoped Ruff clean. New events were validated against the full shared JSON schema, including
nullable timing. Logs: `tmp/alert_continuity_before.txt`, `tmp/alert_continuity_after.txt`,
`tmp/alert_continuity_full_suite.txt`. Runtime remains Python 3.12.5; clean 3.11 and real Feast not verified.
No dependency installation, threshold recalibration, final-test benchmark, production retraining or push.
Existing model/data/reports are unchanged. Historical alert rates and delays were not rerun and must not
be presented as validation of the corrected detector. By eye, confirm that 16:00/19:00 cannot alert and
that second-hour target start, label arrival and event creation are distinct timestamps.
Next step: R5 runbook commands and protection against automated final-test reuse. Stop here until asked.

### Exact feature hour (26 November 2026, step 4 only)

Branch: `m1/exact-feature-hour`. Fixes review **R3** (earlier step-3 next-step prose mislabeled it R4).
The original provider accepted a 15:00 feature row for a 16:00 forecast inside its 2-hour freshness limit.
That changes target-hour solar geometry, calendar, lags and weather. New synthetic tests first showed
**17 failures, 3 passes**; after the fix, 23 provider contract cases pass without requiring Feast.

- Online retrieval validates the requested site, all requested feature records and their timestamps.
  Exact equality is required, including subsecond precision. Older-within-limit rows now raise
  `TargetHourMismatchError`; existing future/stale errors remain. A null weather value is permitted
  when its record exists for the exact hour. Missing records are not interpreted as weather nulls.
- All requests require timezone-aware timestamps and normalize to UTC. No silent local-time assumption.
  Staleness limits must be finite/nonnegative and cannot authorize a different target hour.
- Offline retrieval checks `(site_id, event_timestamp)` against the registered view's existing local
  Parquet FileSource. Source keys must be unique. Every request must exist exactly before calling Feast;
  every returned value must match that exact row afterward. Source weather NaNs stay NaN.
  Output adds `source_feature_time_utc`. It cannot use an as-of echo of the request time as source evidence.
- This source verifier supports the current unmapped local Parquet source only. Remote/mapped backends
  require another adapter and fail explicitly here. Reading projected source columns adds offline I/O.
  Data tables, registry definitions, model features and installed artifacts were not rebuilt or changed.
- Nights absent from the daylight-only table return feature-unavailable errors; never reuse prior daylight
  rows. A future on-demand path must rebuild target-specific features with the shared builder.
- Member 3 handoff now defines an explicit Feast-process -> JSON/Parquet -> core-model-process boundary
  with site/time checks. A live bridge/API is not implemented in this step.

```powershell
$env:PYTHONPATH=''
.\tmp\review_env\Scripts\python -m pytest tests/test_feast_provider_contract.py -q
.\tmp\review_env\Scripts\python -m pytest -m "not network" -q --tb=short
.\tmp\review_env\Scripts\python -m ruff check . --exclude tools

# Required follow-up in an existing isolated Feast environment (not available locally).
.\.venv-feast\Scripts\python -m pytest tests/test_feature_store.py -q
```

Full non-network suite: **167 passed, 1 skipped (Feast), 1 deselected (network)**; scoped Ruff passed.
Logs: `tmp/exact_feature_hour_before.txt`, `tmp/exact_feature_hour_after.txt`,
`tmp/exact_feature_hour_full_suite.txt`. Verification uses Python 3.12.5 and mocked backend responses plus
temporary Parquet sources. The real Feast backend, clean Python 3.11 environment and live service boundary
are not verified. No real model retraining, final-test evaluation, materialization, new dependencies or push.
Before deployment, inspect requested versus source feature time (same UTC instant), site identity, and the
feature-unavailable response for a missing night/hour; do not convert missing rows into zero production.
Next step: review **R4**, continuity in deviation alerts. Stop here until asked to continue.

### Isolated candidate artifacts (26 November 2026, step 3 only)

Branch: `m1/candidate-artifacts`. Synthetic regression tests first confirmed approved-file overwrites and
rejected CLI success (3 failed, 1 passed). The corrected training pipeline writes only new candidate outputs:

- `models/candidates/<candidate-id>/bundle.joblib`, `validation.parquet`, `evaluation.json`, `manifest.json`.
- `reports/candidates/<candidate-id>/plots/` for that exact candidate's diagnostics.
- The manifest binds model/validation/evaluation SHA-256 hashes, code revision, config/data hashes, feature
  metadata, training window and MLflow run ID. It always says `deployment_approved: false`.
- Rechecking reloads the saved bundle and uses `predict`, never `fit`. Changed/missing files, inappropriate
  chronology, failed CV or failed exact-model metrics prevent eligibility. A manifest is integrity evidence
  from a trusted run, not a cryptographic signature or independent proof of data provenance.
- Training/evaluation CLI exit `0` means eligible, `2` means rejected; exceptions fail nonzero. Rejected
  candidates may be logged but are not registered. Registration errors propagate. No champion alias or
  approved serving file is changed, including for passing candidates. Promotion remains future work.

To obtain honest validation of the distributed weights without reopening the already-used final test,
future candidates are fitted only on the training side of the last development fold. Rows whose hourly
labels plus reporting delay are unavailable at validation start are excluded. The frozen model is scored
on development Fold 3. CV selection already uses this fold, so this is development evidence, **not a new
unbiased test estimate**. Unlike the supplied historical full-dev model, new candidates stop before Fold 3;
refitting on all development rows creates different weights and invalidates that candidate's gate evidence.
The historical weather-present daylight mask and quality thresholds are retained in this step.

```powershell
# Verified here using synthetic candidate training and mocked MLflow (no production retraining).
$env:PYTHONPATH=''
.\tmp\review_env\Scripts\python -m pytest tests/test_candidate_artifacts.py -q
.\tmp\review_env\Scripts\python -m pytest -m "not network" -q --tb=short
.\tmp\review_env\Scripts\python -m ruff check . --exclude tools

# Future approved training run: prints its unique manifest path; it does not deploy.
.\.venv\Scripts\python -m src.models.train
# Use the actual manifest path from that run. Do not use the historical root evaluation.json.
.\.venv\Scripts\python -m src.models.evaluate --candidate-manifest "models/candidates/<candidate-id>/manifest.json"
```

Full suite: **144 passed, 1 skipped (Feast unavailable), 1 deselected (network)**; scoped Ruff passed.
Log: `tmp/candidate_artifacts_tests.txt`. Python is still the temporary 3.12.5 review runtime; clean 3.11,
separate Feast and an actual shared MLflow registry were not verified. The existing approved bundle and
three historical evaluation/audit files match their packet receipt hashes. No real-data model was retrained,
no final-test benchmark was run and no GitHub push or Member 2 infrastructure was performed.
Next priority is exact target-hour feature retrieval (R4); stop here until the owner requests that step.

### Quality-gate validation fix (26 November 2026, step 2 only)

Integrated GitHub main `5ac8ed0` on feature branch `m1/quality-gate`. Preserved local review and packet notes
in a separate documentation commit, retaining both upstream and local decision/changelog entries during
handoff conflict resolution. The latest remote gate still contained the numerical-validation bug.

Changed only quality-gate evaluation behavior in `src/models/evaluate.py` and added
`tests/test_quality_gate.py` (44 cases). Before the fix, the new tests reported **37 failed, 7 passed**.
After the fix, gate/model-focused tests reported **53 passed**; the complete non-network suite reported
**127 passed, 1 skipped, 1 deselected**. Ruff passed using the existing project configuration plus exclusion
of the untracked local `tools/` PDF helper. Runtime remains the existing Python 3.12 review environment;
this does not claim separate Feast or Python 3.11 environment verification.

Behavior contract:
- Missing, negative, nonnumeric or nonfinite error metrics reject the candidate with explicit failure reasons.
- Relative improvement requires a finite, strictly positive baseline MAE.
- Predictions must be numeric, finite, nonempty, one-dimensional and within the existing physical bounds.
- Invalid policy settings raise `ValueError` rather than disabling checks.
- Unknown metric/improvement/threshold values use JSON `null`, never NaN/Infinity. Valid-result fields and
  threshold behavior remain compatible. Downstream code must check `eligible`, not interpret missing values as zero.

```powershell
.\tmp\review_env\Scripts\python -m pytest tests/test_quality_gate.py tests/test_model.py -q
.\tmp\review_env\Scripts\python -m pytest -m "not network" -q
.\tmp\review_env\Scripts\python -m ruff check . --exclude tools
```

Logs: `tmp/quality_gate_before.txt`, `tmp/quality_gate_after.txt`, `tmp/quality_gate_full_suite.txt`.
No model was retrained and no final-test benchmark was invoked. This step does not change model quality
thresholds, prediction units or data scope. It also does not fix the remaining R2 candidate-bundle overwrite,
exact-artifact evaluation or deployment/promotion controls. Alert, feature retrieval and Member 2 pipeline
implementation are separate next tasks. Stop after this step as requested.

### Artifact packet receipt (26 November 2026, step 1 only)

Opened `SolarCast_packet.zip` and checked all 31 payload files against both `CHECKSUMS.sha256` and
`PACKET_MANIFEST.json` (28,116,883 payload bytes). Safe staging location:
`tmp/artifact_packets/solarcast_20261124/`. Installed 28 missing `data/` and `models/` files after verifying
their hashes; did not replace source/config/report files. The three existing config/report payload files
were byte-identical. Receipt: `reports/packet_receipt_20261126.json`.

The supplied model loads as `solarcast-f1`, Feature Set B, 29 features, schema 1.1.0, MLflow run
`f7fef3f7bdf241c5a53dc50d8e4713f9`. Its recorded hashes match the installed hourly, quarter-hourly and
HRRR datasets. Packet metadata reports local MLflow only; shared tracking/storage still needs Member 2 setup.
The supplied registry/lineage audit is historical metadata, not verification of an accessible shared service.

```powershell
# Integrity/install verifier created for this local receipt; refuses differing artifact overwrites.
python tmp/verify_solarcast_packet.py

# Existing artifact/contract tests; no retraining or final-test benchmark invocation.
.\tmp\review_env\Scripts\python -m pytest -m "not network" -q --tb=short
```

Observed result: **64 passed, 2 skipped, 17 warnings**. Log: `tmp/packet_pytest_20261126.txt`.
Runtime remains the existing Python 3.12.5 review environment; separate Feast tests and a clean Python 3.11
build have not been verified here. Passing the existing suite does not resolve the independent review's
uncovered bugs. No quality-gate/alert/feature changes, model retraining, final-test benchmark, DVC setup,
GitHub push or Member 2 pipeline implementation was performed in this step.

Fetched newer Git refs for provenance: packet source commit `177c655522af91a0d17ee086e5eeae9b61d60817`
exists, and current remote main is `5ac8ed0`. Current reviewed code stays at `d8c907f` to preserve local
review notes. Next step: reconcile those newer monitoring/lineage commits, then address one priority bug
at a time, starting with fail-closed model quality gates.

### Independent review checkpoint (24 November 2026)

Read `docs/member1_review_20261124.md` before following the older automated handoff commands below.
Reviewed main `dbd5f40` plus the unmerged handoff branch `d8c907f` on local `m1/member1-review`.
Core review environment: available Python 3.12.5 with pinned core requirements, not the required clean 3.11 build.
`tmp/review_env/Scripts/python -m pytest -m "not network" -q --tb=short` produced **48 passed, 1 failed,
14 errors, 3 skipped**; missing data/model artifacts caused the failure/errors. Ruff passed with the existing
scripts exclusion and the local untracked PDF helper excluded (`--exclude tools`). Separate Feast tests and
real-data/model-score reproduction were not run; final holdout was not reopened.

Independent synthetic probes reproduced NaN-metric gate acceptance, a gap incorrectly counted as consecutive
hours, acceptance of an old target-specific online feature row, unparseable-power quarantine omission and an
unsupported documented deviation CLI flag. No application fixes were made. Probe: `tmp/member1_review_probes.py`;
logs/results: `tmp/review_pytest.txt`, `tmp/review_ruff_tracked.txt`, `tmp/member1_review_probe_results.json`.
The full report records source references, verification limitations, owners and acceptance checks.

**Historical prerequisite, resolved 26 November:** the datasets, weather features, model bundle and packet
tracking/hash references have now been supplied and verified above. DVC/shared MLflow are not configured yet. Do not schedule
the default deviation benchmark: it accesses the final test period. Member 2 implementation awaits this review's
priority fixes and artifact bootstrap; DVC/fixture-CI preparation can proceed as a separate next task.

### Existing Member 1 run instructions
```powershell
# Environment setup (PowerShell on Windows)
py -3.11 -m venv .venv
# Note: If the py launcher reports "No installed Python found!", use the direct Python 3.11 executable:
# & "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" -m venv .venv

.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run data ingestion pipeline (PVDAQ registration & NASA POWER weather fetch)
python -m src.data.ingestion --config configs/data.yaml
# (or directly: .\.venv\Scripts\python -m src.data.ingestion --config configs/data.yaml)

# Run data validation, quarantine, and dataset aggregation (hourly and quarter-hourly)
python -m src.data.aggregation --config configs/data.yaml
# (or directly: .\.venv\Scripts\python -m src.data.aggregation --config configs/data.yaml)

# Run baseline model evaluation (3-fold cross-validation and report generation)
python -m src.models.baselines --config configs/evaluation.yaml --report docs/baseline_report.md
# (or directly: .\.venv\Scripts\python -m src.models.baselines --config configs/evaluation.yaml --report docs/baseline_report.md)

# Train candidate models, evaluate gate, generate diagnostic plots, and log to MLflow
python -m src.models.train --config configs/evaluation.yaml
# (or directly: .\.venv\Scripts\python -m src.models.train)

# Run Feature Set A vs Feature Set B comparison (cloud bins, monthly breakdown, permutation importance)
python scripts/compare_set_a_b.py
# (or directly: .\.venv\Scripts\python scripts/compare_set_a_b.py)

# Run HRRR forecast weather extraction pipeline (Mode F1 features)
# Process in separate PowerShell window (resumable monthly parts in data/processed/hrrr_parts/):
python scripts/hrrr_extract.py --threads 16
# (or directly: .\.venv\Scripts\python scripts/hrrr_extract.py --threads 16)

# Flexible date filtering and variable selection:
# Priority 1: 2022-10-17 to 2023-11-09 (Test & Dev fold 3)
python scripts/hrrr_extract.py --start 2022-10-17 --end 2023-11-09 --variables tmp,tcdc --threads 16

# Run in Google Colab (recommended for ~50-100x higher throughput):
# Open notebooks/colab_hrrr_extract.ipynb in Google Colab, copy D:\SolarCast_inputs to Drive, and execute.

# Watch live extraction progress:
Get-Content logs\hrrr_extract.log -Tail 5 -Wait

# Merge monthly parts into consolidated dataset:
python scripts/hrrr_extract.py --merge
# (or directly: .\.venv\Scripts\python scripts/hrrr_extract.py --merge)

# Run model evaluation dry-run on last validation fold (Fold 3, strictly Train/Dev):
.\.venv\Scripts\python -m src.models.evaluate --dry-run

# Official one-time final test evaluation (requires --force and --reason; audit in reports/final_test_audit.json):
# .\.venv\Scripts\python -m src.models.evaluate --final-test --force --reason "one-time final evaluation after model selection"

# Run NOAA HRRR forecast data feasibility audit (Task M1.1b)
python scripts/hrrr_feasibility.py
python scripts/step1_inventory.py
python scripts/step2_coverage_scan.py
python scripts/step3_step4_mixed_and_signal.py
# (or directly with .\.venv\Scripts\python ...)

# Feast Feature Store Integration (Step 1 to Step 5; see docs/feature_store.md for complete architecture guide)
# Set up isolated Feast virtual environment (avoids numpy 2 conflict in main .venv):
py -3.11 -m venv .venv-feast
.\.venv-feast\Scripts\pip install -r requirements-feast.txt

# Build shared Feature Set B Parquet table (23,532 daylight issue hours with audit columns):
.\.venv-feast\Scripts\python -m src.features.build_feature_table

# Apply feature definitions to Feast registry:
cd feature_store/feature_repo
..\..\.venv-feast\Scripts\feast apply
cd ../..

# Materialize offline features into SQLite online store with MLflow tracking:
.\.venv-feast\Scripts\python feature_store/materialize.py --start 2023-06-01T00:00:00+00:00 --end 2023-06-15T15:00:00+00:00 --log-mlflow

# Run end-to-end Feast real-data demonstrations (skew test, online serving, stale/future rejection, replay):
.\.venv-feast\Scripts\python -m scripts.demo_feast

# Run Feast unit test suite (9 comprehensive tests in .venv-feast):
.\.venv-feast\Scripts\pytest tests/test_feature_store.py -v

# Development-only calibration; produces separate unapproved artifacts, not a final-test benchmark
.\.venv\Scripts\python -m src.models.deviation --calibrate --bundle models/solarcast_f1_bundle.joblib --reports-dir reports/deviation_runs --log-mlflow

# Run deviation detector unit test suite
.\.venv\Scripts\pytest tests/test_deviation.py -v

# Run chronological virtual-clock monitoring replay across all 4 scenarios with MLflow tracking:
.\.venv\Scripts\python -m scripts.replay_monitoring --log-mlflow

# Run monitoring unit test suite (PSI hand-computed, seasonal ref, 200-row/30-label floors, causality, schema v1.1.0):
.\.venv\Scripts\pytest tests/test_monitoring.py -v

# Main project test suite (verifying Feast tests auto-skip cleanly in main .venv):
pytest -m "not network"
# (or directly: .\.venv\Scripts\pytest -m "not network")

# Code quality and style checks
ruff check .
# (or directly: .\.venv\Scripts\ruff check .)

# Generate standalone artifact packet outside repo with checksums, manifest, and zip:
python scripts/make_artifact_packet.py --out D:\SolarCast_packet --zip
# (or directly: .\.venv\Scripts\python scripts/make_artifact_packet.py --out D:\SolarCast_packet --zip)

# Verify artifact packet against CHECKSUMS.sha256:
python scripts/verify_artifact_packet.py D:\SolarCast_packet
# (or directly: .\.venv\Scripts\python scripts/verify_artifact_packet.py D:\SolarCast_packet)

# Install verified packet into a clean repository clone:
python scripts/verify_artifact_packet.py D:\SolarCast_packet --install C:\path\to\SolarCast
# (or directly: .\.venv\Scripts\python scripts/verify_artifact_packet.py D:\SolarCast_packet --install C:\path\to\SolarCast)

# Resolve MLflow run or champion alias to cryptographic code/data/schema lineage:
python scripts/resolve_run.py champion
python scripts/resolve_run.py 620faa341a8a4c14824f31e40b157fd6
# (or directly: .\.venv\Scripts\python scripts/resolve_run.py champion)
```

For the new data-feasibility task, follow README.md's Member 1 checklist and submit `docs/data_feasibility.md`
plus a reusable extraction/validation script and sample lineage. No HRRR extraction command exists yet.

Scope-documentation verification on 2026-11-21: `python -m pytest tests -q` passed all 4 scaffold tests
using the available Python 3.12 host; this does not verify the pinned Python 3.11 ML stack or any data pipeline.
README local links, code fences, checklist count and `git diff --check` passed. `python -m ruff check .`
could not run because Ruff is not installed on this host (`No module named ruff`); no Python code was changed.

## 7. Spec traceability (Member 1)
Course spec 3.1 modular code | 3.2 DVC tracking (with Member 2) | 3.3 MLflow runs: model type, params, data version, metrics,
duration, artifacts, Git commit | 3.5 data/model/regression tests | 12.1 Feast: entities, feature views, offline retrieval,
materialization, online query, skew/stale/null tests, feature repo commit in MLflow | 3.13 dataset licence + fairness (site/season coverage)
+ risk register entries. Final report needs an individual contribution statement.

## 8. Open questions for the team (tick when answered)
- [ ] Repo URL and collaborator access (Member 2)
- [ ] Python version and pinned stack (Member 2 decides; Feast/MLflow/Airflow/KFP/BentoML may need separate envs)
- [ ] DagsHub MLflow tracking URI and DVC remote (Member 2)
- [ ] Data snapshot URI/hash contract and MLflow metadata keys (Member 2)
- [ ] Request/response schema and feature retrieval mode for the API (Member 3)
- [ ] Model bundle format Member 3 needs (joblib vs MLflow pyfunc)
- [ ] Final deadline date, machine specs, AWS budget (all)
- [x] Site/dataset sign-off by all three after M1.1 (Approved: System 2107 Farm Solar Array, Arbuckle, CA)

## 9. Risks / known issues
- Data access and size (prize datasets can be tens of GB): sample first, then scale.
- Timestamp/unit errors (local vs UTC, DST, left/center/right interval alignment, kW vs kWh): verify with fixtures.
- HRRR forecast fields, publication timing and interval alignment remain unverified; do not begin full training
  or claim an operationally faithful backtest before the feasibility gate is reviewed.
- Dependency conflicts between Feast, MLflow, Airflow, KFP, BentoML: keep separate environments.

## 10. Decision log
| Date | Decision | Why |
| 2026-11-27 | Validate raw meter fields before timezone conversion | Keep malformed samples out of aggregation and retain raw power text for audit. |
| 2026-11-27 | Separate model-bound dev calibration from bounded monitoring; make final-test audit immutable | Remove implicit recalibration/holdout access and arbitrary champion parameters. Preserve approved artifacts and one-time evidence; new workflows use isolated outputs, actual model identity and delayed-label checks. Historical natural/final deviation readers are retired. |
| 2026-11-26 | Require true hourly adjacency; add incremental detector without changing retrospective wrapper | Gaps/duplicates cannot confirm persistence. Keep per-site/model in-memory state across ordered batches; emit once at second breach, use stable IDs, and report unknown arrival timing as null. Frozen bounds and historical results preserved; durable monitoring remains future work. |
| 2026-11-26 | Require exact feature issue time in both retrieval paths; verify existing registered Parquet source without schema changes | Target-specific vectors cannot be reused within a freshness window. Missing/ambiguous rows reject, source time is returned for audit, weather nulls are preserved. Real Feast integration and live process bridge remain unverified. |
| 2026-11-26 | Isolate future candidates and bind eligibility to saved weights; no automatic promotion | Prevent rejected runs from overwriting approved artifacts. Frozen fits stop before last development fold with label-availability purge; exact bundle is reloaded and scored. Existing full-dev model and historical results remain unchanged. |
|---|---|---|
| 2026-11-26 | Step 2: reject invalid numerical gate evidence and invalid policy; preserve valid decisions | Added 44 regression cases, reproduced 37 failures on latest main, then passed focused and full suites. Invalid values become null in rejection records. Candidate artifact/promotion safeguards remain a separate task; no retraining/final-test benchmark. |
| 2026-11-24 | Standalone Artifact Packet, Checksum Verification & Read-Only MLflow Lineage Audit | Packaged 31 verified data, model, and convenience files to D:\SolarCast_packet (26.81 MB uncompressed) and created D:\SolarCast_packet.zip (12.91 MB, sha256 93504e87d870e7881235d3de1fbedbff712492ae567abb05107af5536e797e21). Excluded hrrr_parts/ and exploratory samples june_week_extracted / mixed_week_extracted. Excluded mlruns/ due to absolute Windows file URIs breaking cross-machine portability (runs will be re-logged to the shared server by running python -m src.models.train once tracking URI exists). Generated CHECKSUMS.sha256, PACKET_MANIFEST.json mapping each file to its pipeline generation command, and performed read-only MLflow lineage audit across the champion run, registered model solarcast-f1 v1-v4 (confirming v4 is champion, stages/aliases unset in local store), and operational runs without modifying any runs or results. Built scripts/verify_artifact_packet.py, verified on the packet, and tested deliberate 1-byte corruption asserting detection and abort. |
| 2026-11-24 | Monitoring Signals, Month-Specific Performance Reference, and Input Drift Calibration | Evaluated input drift (src/models/drift.py) across 8 key numeric features using 10-quantile PSI against season-matched Train/Dev daylight observations (matching calendar month +/- 1 month and daylight hours). Enforced 200-row sample minimum; documented that 14-day daylight windows at 39°N have <200 hours outside mid-summer, correctly outputting insufficient_data rather than false drift alarms. Compared performance reference definitions on 1,551 Train/Dev validation days: fixed reference (31.43 kWh) triggered 258 false alert days (170 in winter/spring: 57 in Jan, 36 in Nov), whereas month-specific reference cut false alarms by >50% to 127 days, eliminating winter seasonal distortion. Froze month-specific reference in configs/monitoring.yaml. Enforced virtual clock label visibility with 15-minute operational delay; missing actuals trigger label_data_outage, never accuracy degradation. Bumped alert event schema to v1.1.0 supporting alert_type (residual_deviation, input_drift, performance_degradation, data_quality). Replay over 375 test days demonstrated sensitivity to weather shifts (detected in 4.0d) and performance drops (detected in 3.0d), while correctly suppressing alerts during a 5-day label outage. Logged MLflow run monitoring_replay (16ffd7429be642928941e95f932fe8b3). Authored docs/monitoring_signals.md. |
| 2026-11-26 | Step 1 complete: accept supplied packet integrity; preserve separate code-review conditions | All 31 payload checksums/sizes and bundle data hashes match. Installed missing artifacts; existing tests now 64 passed/2 skipped. No scope change, code fixes or automatic deployment approval. |
| 2026-11-24 | Independent integration review: conditional acceptance, no deployment sign-off | Latest code/handoff pulled and inspected. Synthetic probes exposed gate, target-time and alert gaps; artifact absence prevents full reproduction. See docs/member1_review_20261124.md. No dataset/scope change or full Member 2 implementation authorized by this review. |
| 2026-11-24 | Deviation Detector Threshold Calibration & 2-Hour Persistence Alerting Rule | Calibrated frozen 1st and 99th percentile residual thresholds strictly across the 3 expanding validation folds (13,914 VALID daylight hours; clear: [-140.66, 127.94], mixed: [-231.18, 226.24], overcast: [-203.70, 216.30], no_weather: [-155.80, 191.86] kWh). Evaluated clear-sky index cross-stratification and confirmed rejection because no_weather x csi_low (85 samples) and no_weather x csi_high (154 samples) violated the 200-sample threshold rule; merged into cloud bins. Enforced 2-consecutive-hour persistence rule to suppress transient spikes while reliably flagging sustained faults (injected zero output: 95% detected with 1.0h median delay; 0% false alerts on isolated single-hour drops). Unmodified test false-alert rate was 0.43/1000h. Natural SUSPECT_ZERO evaluation detected 6/43 episodes (13.95%, median delay 3.0h; non-detected were low-power sunrise/sunset events where expected power < 140 kWh). Logged MLflow run deviation_detector. Authored docs/deviation_detector.md. |
| 2026-11-24 | Feast Environment Isolation (.venv-feast) & TTL Semantics | Verified pip install --dry-run feast requires numpy>=2.0.0, conflicting with pinned numpy==1.26.4 / scipy==1.13.1 in .venv. Established isolated .venv-feast with requirements-feast.txt, leaving core requirements.txt untouched. Read Feast 0.68.0 TTL behavior: (1) Historical retrieval (get_historical_features) evaluates TTL as lookback window [entity_timestamp - ttl, entity_timestamp] during point-in-time as-of join; (2) Online SQLite store does not evaluate query timestamps or staleness at query time. Authored custom provider src/features/feast_provider.py enforcing future-timestamp rejection and staleness threshold rejection (max_staleness_hours=2.0). |
| 2026-11-23 | One-Time Final Test Evaluation & Safeguard Compliance | Executed official one-time final test evaluation across the 20% held-out test partition (2022-10-17 01:00 to 2023-11-09 23:00 UTC, 9,273 valid hours, 62 excluded: 35 SUSPECT_ZERO, 27 MISSING_DATA) under strict check_final_test_safeguard() logging with --force and --reason "one-time final evaluation after model selection". No models were selected or tuned on test results; all models were fit strictly on the 80% Train/Dev partition. Champion HistGradientBoosting Set B confirmed decisive superiority on primary weather-present daylight hours (N = 4,387): 32.69 kWh Daylight MAE (3.66% cap-MAE) and 56.85 kWh RMSE (+24.83% vs Clear-Sky Scaled 15m Persistence baseline champion, +3.84% vs HGB Set A). On all daylight hours (N = 4,601 including 214 missing-weather hours: cause unknown; observed only as missing files/chunks in the archive), HGB Set B achieved 33.48 kWh Daylight MAE (3.75% cap-MAE), outperforming the baseline champion by +24.33% and Set A by +3.46%. Generalization gap vs validation CV was minimal (+1.26 kWh, +4.01%), smaller than Set A (+1.75 kWh, +5.44%). Updated evaluation.json, reports/evaluation.json (final_test_opened: true), reports/final_test_audit.json, and MLflow run final_test_evaluation. Authored docs/final_test_report.md. |
| 2026-11-23 | Champion Model Selection: HistGradientBoosting Set B (+2.52% vs Set A, +25.33% vs Champion Baseline) | Compared Feature Set A (23 features) and Feature Set B (29 features including NOAA HRRR forecast TMP, TCDC, Kasten-Czeplak cloud adjustment, run age, fallback, missingness) on identical validation daylight hours (N = 13,374) across 3 expanding-window folds. HistGradientBoosting Set B achieved 31.43 kWh Daylight MAE (3.52% cap-MAE) and 53.35 kWh RMSE, outperforming Set A (32.24 kWh MAE, 54.88 kWh RMSE) by +0.81 kWh (+2.52%) and Clear-Sky Scaled 15m Persistence (42.09 kWh) by +25.33%. Meeting the pre-registered selection rule (>= 1.0% improvement over Set A), Set B was selected as the champion model. Candidate gate passed (reports/evaluation.json and ./evaluation.json, final_test_opened: false). Exported models/solarcast_f1_bundle.joblib with feature_set='B', needs_weather=True, schema 1.1.0. Logged runs to MLflow and registered model solarcast-f1 v4. |
| 2026-11-23 | Data Sanitization for Upstream Weather Fill Values (-10,000 in July 2020) | Auditing revealed one record (2020-07-29 21:00 UTC) with tcdc_pct = -10000.0 (upstream fill value in MesoWest Zarr). Updated src/features/preprocessing.py to enforce meteorological valid bounds (TCDC in [0, 100]%, TMP in [-40, 60]°C), treating out-of-bounds values as NaN with weather_missing = 1. Added test_weather_valid_range_filtering in tests/test_features.py and documented in docs/weather_data_quality.md. |
| 2026-11-23 | Empirical Cloud-Cover Stratification & NWP Mixed-Cloud Degradation Analysis | Stratifying identical validation hours by forecast total cloud cover revealed weather features help under clear sky (<20% TCDC: +5.59% MAE gain) and overcast sky (>80% TCDC: +4.94% MAE gain), but degrade by -9.70% (54.26 vs 49.46 kWh) under mixed cloud (20-80% TCDC). Identified cause as NWP 3-km spatial/temporal phase timing errors, where rapid cloud formation is mistimed relative to compact PV collector fields; local 15-minute generation telemetry (power_kw_lag_0m) is superior in mixed regimes. Documented in docs/model_report.md and docs/model_card.md. |
| 2026-11-23 | Fair 15-minute baseline benchmark & candidate gate re-evaluation | Added (d) Last 15-Minute Persistence and (e) Clear-Sky Scaled 15-Minute Persistence to src/models/baselines.py, giving baselines the identical 15-minute operational telemetry available to the ML models. Redefined baseline champion as Clear-Sky Scaled 15-Minute Persistence (42.18 kWh Daylight MAE / 4.72% cap-MAE / 67.61 kWh RMSE; delay 0m: 33.09 kWh, delay 60m: 67.50 kWh). Recomputed candidate gate: HistGradientBoosting won decisively with 32.24 kWh Daylight MAE (23.56% improvement over champion baseline, passing gate, registered as solarcast-f1 v2); Ridge achieved 43.52 kWh (-3.16%, rejected). Computed permutation importance for HistGradientBoosting (power_kw_lag_0m +116.92 kWh, hour_sin +40.97 kWh, energy_kwh_prev_day_same_hour +15.03 kWh). |
| 2026-11-23 | Candidate Model Selection & Acceptance Gate Pass: HistGradientBoosting (57.36% improvement) | Trained and benchmarked Ridge Regression and HistGradientBoostingRegressor on 3 expanding-window folds of the 80% Train/Dev partition (37,090 hours) under Feature Set A (23 features). HistGradientBoosting won decisively with 32.24 kWh Daylight MAE (3.61% cap-MAE) and 54.76 kWh RMSE, outperforming Clear-Sky Persistence (75.62 kWh) by 57.36% and Ridge (43.52 kWh) by 25.9%. Candidate gate passed (threshold >= 5.0% improvement, <= 20% cap-MAE). Written to reports/evaluation.json and ./evaluation.json with final_test_opened: false. Guarded final test with check_final_test_safeguard(). Logged experiment solarcast-model-training in MLflow, registered model solarcast-f1 v1 (status: candidate), and packaged models/solarcast_f1_bundle.joblib. Authored docs/model_report.md and docs/model_card.md. |
| 2026-11-23 | HRRR extraction order, stopping rule, and Google Colab offload | Rigorous diagnostic on local host demonstrated that low extraction rate (~0.1 req/s) is driven entirely by residential network bandwidth and transoceanic latency to AWS S3 us-west-1 (11.8 KB/s on 1MB chunk, median TTFB 1.74s, 4 threads 60.9 KB/s, 16 threads collapsed to 18.9 KB/s due to TCP socket contention). Code does not serialize I/O (thread-local S3FileSystem, lock on int counter only). Created Google Colab extraction kit (requirements-extract.txt, notebooks/colab_hrrr_extract.ipynb) leveraging cloud S3 bandwidth. Staged extraction priorities and stopping rules: Priority 1: 2022-10-17 to 2023-11-09 (Test & latest Dev fold 3); Priority 2: 2021-10-17 to 2022-10-16 (Prior training fold); Priority 3: older months as time permits. Filtered default variables to tmp,tcdc (reducing chunk requests by 60%). Prepared inputs in D:\SolarCast_inputs\. |
| 2026-11-22 | Forecast weather route approval (author's decision pending team confirmation) | forecast weather route = HRRR forecast 2 m temperature + total/low/mid/high cloud cover from the Zarr archive + pvlib clear-sky irradiance, for 2018-07-12 to 2023-11-09, availability lag 2 h, fallback to earlier runs; proceeding on the author's decision pending team confirmation. |
| 2026-11-22 | Baseline champion selection: Clear-Sky Persistence (75.62 kWh Daylight MAE) | Evaluated three baseline models on 3 expanding-window chronological validation folds within Train/Dev (37,090 hours) while strictly locking final 20% test set (9,273 hours). ClearSkyPersistence achieved lowest Daylight MAE (75.62 kWh, 8.47% cap-MAE) and superior Daylight RMSE (113.80 kWh), outperforming Same-Hour Previous Day (75.65 kWh MAE, 128.37 kWh RMSE) and Latest Available Hour (175.73 kWh MAE). Candidate ML models must beat 71.84 kWh Daylight MAE (5% gate). |
| 2026-11-22 | Feature pipeline architecture and 15-minute operational cutoff | Unified training and serving via single build_features(issue_times_utc, ...) function in src/features/preprocessing.py. Enforced operational availability cutoff of H - 15m (latest usable 15-min interval [H-30m, H-15m), latest fully available hour [H-2h, H-1h)). Generated 23 features with zero future leakage verified by perturbation tests, DST fall-back handling, missing hour fallback, and prehistory tests. Tracked via configs/feature_schema.json and configs/features.yaml. |
| 2026-11-22 | NOAA HRRR Route 1 Gate Approval (Hybrid TMP + TCDC + pvlib Clear-Sky) | Completed variable inventory (chunk 1.4 non-empty for TCDC, LCDC, MCDC, HCDC, TMP, DPT, RH, UGRD, VGRD, VIS, PRATE, APCP; DSWRF chunks missing), full 46,728-run multi-year scan (TMP 94.18%, TCDC 94.07% available), and mixed-weather signal check on winter atmospheric river week (Spearman rho = -0.7884, p=5.46e-13 vs clear-sky index proxy). Verified eccodes GRIB2 decoding in-memory on Windows Python 3.11 and confirmed WSL2 Ubuntu. Selected Route 1 as primary Mode F1 strategy; rejected GRIB2 DSWRF due to high network/request overhead (~185 GB estimate, 187k HTTP range requests); retained NASA POWER for retrospective Mode E1. |
| 2026-11-22 | Diurnal cycle vs weather forecast skill distinction | Raw correlations (0.945 for clear-sky GHI, 0.927 for HRRR analysis DSWRF) are driven by astronomical day/night cycles, not advance weather forecast skill. Normalizing by clear-sky GHI isolates true cloud attenuation skill. |
| 2026-11-22 | NOAA HRRR feasibility gate evaluation (M1.1b) | Reject standalone HRRR Zarr irradiance due to missing forecast DSWRF chunks (iris-grib #265 bug) and missing pre-July 2018 forecast runs in Zarr; recommend CONDITIONAL path utilizing HRRR forecast 2m temperature with operational 2-h lag, pvlib astronomical clear-sky irradiance (r=0.945), and past-only production lags for Mode F1; retain NASA POWER for retrospective Mode E1. |
| 2026-11-22 | Daylight and solar elevation use interval midpoint (t + 30m); high_sun (>5 deg) added | Evaluating solar elevation at the midpoint of each hourly interval [t, t+1h) accurately captures daytime production during sunrise/sunset hours. Hours with high_sun and target_kwh == 0 in runs of 2+ consecutive hours are flagged SUSPECT_ZERO without mutating target_kwh. |
| 2026-11-21 | DST fall-back handling, quarantine rules, and hourly aggregation threshold | Assign ambiguous local fall-back timestamps to the first occurrence (PDT); treat unrecorded second occurrence (09:00 UTC) as missing data (NaN target). Require min_intervals=4 (all 4 fifteen-minute intervals) for a valid hourly kWh target (sum(kW)*0.25); otherwise set target to NaN with explanatory quality_flag. Quarantine values > 893 kW, negatives, and non-finite without silent clipping into quarantine.parquet. |
| 2026-11-21 | User authorized revised weather-driven F1 scope; HRRR is a candidate, NASA supports retrospective analysis | Preserve the assigned weather/irradiance/temperature objective without future-observation leakage. Member 1 must verify and report before source approval. |
| 2026-11-16 | System 2107 approved as project PV site | Approved by human owner. Detailed verification confirmed local civil time with DST (America/Los_Angeles), 15-minute left-aligned average active power (kW), 893 kW DC capacity / 662.4 kW nominal AC capacity |
| 2026-11-16 | Recommend System 2107 (Farm Solar Array, Arbuckle, CA) for production | Clean revenue meter (instantaneous kW, 6.32 MB, 0.13% missingness over 7 years), fixed 25° ground mount (no tracker/multi-roof complications), left-aligned, verified against NASA POWER hourly UTC weather |
| 2026-11-15 | Pinned stack in requirements.txt | Pinned 16 core Member 1 packages for Python 3.11 (pandas 2.2.2, numpy 1.26.4, scikit-learn 1.5.2, scipy 1.13.1, mlflow 2.17.2, pvlib 0.11.1, pandera 0.20.4, etc.), excluding feast/airflow/kubeflow/bentoml to avoid early dependency conflicts |
| 2026-11-15 | Member 1 scaffold & directory layout | Created Member 1 package structure on branch m1/scaffold with empty __init__.py files, .gitkeep in empty folders, ruff config (line-length 100), .env.example, and test_smoke.py |
| 2026-11-13 | First candidate system 2107; backups 2105, 9068 | Mid-size fixed system with long history; not yet validated |
| 2026-11-13 | Preferred route: PVDAQ via OEDI + NASA POWER (UTC) | Named in team plan; sources reviewed |

## 11. Changelog

- 2026-11-27 (step 7): Closed R6 with required-column checks, numeric and timestamp quarantine, and first-usable duplicate handling. Added two regression tests. Core suite 217 passed, 1 skipped, 1 deselected. Scoped Ruff passed.

- 2026-11-27 (step 6): Closed R5 entrypoint/runbook bugs with immutable repository-anchored first-use final-test claims, dev-only model-recipe calibration, bounded stored-record monitoring, explicit source/model/site identity and isolated outputs. Added `src/models/deviation_workflow.py`, 26 regression/end-to-end tests, CLI help verification and restore-only historical packet metadata; updated Member 2/3 runbooks. Core suite 215 passed, 1 skipped, 1 deselected; historical artifacts preserved. No actual final-test evaluation, production recalibration, new dependency or push. See section 6 for commands and verification limits.

- 2026-11-26 (step 5): Fixed R4 continuity and confirmation on `m1/alert-continuity`. Added `DeviationDetector.process()` for state across ordered nonoverlapping batches, stable episode IDs and explicit/nullable label-based trigger timing; preserved `detect_deviations()` batch summaries. Added optional schema fields and 22 regressions. 30 focused tests passed; full non-network suite 189 passed, 1 skipped, 1 deselected; scoped Ruff clean. Updated monitoring handoff and marked historical rates/latencies as pre-fix evidence. No recalibration, final-test evaluation, retraining or push.

- 2026-11-26 (step 4): Fixed R3 target-hour retrieval on `m1/exact-feature-hour`: exact timestamps for every online feature, exact site/hour source membership and returned-value checks offline, explicit source timestamp output, timezone/policy validation, and no substitution for missing night rows. Added 23 deterministic contract tests; full non-network suite 167 passed, 1 skipped, 1 deselected; scoped Ruff passed. Updated feature-store and Member 3 environment/serving guidance. Artifacts preserved; no real Feast run, retraining, final-test access or push.

- 2026-11-26 (step 3): Fixed R2 artifact overwrite/exact-candidate evaluation on `m1/candidate-artifacts`. Added unique manifest-bound candidate outputs, no-refit evaluator CLI, exit 2 for rejection, no rejected registration and propagated registration errors. 17 new tests; full non-network suite 144 passed, 1 skipped, 1 deselected; scoped Ruff passed. Protected historical model/reports retained; no real-data retraining, final-test benchmark or push. See section 6 for exact commands, fit-window change and verification limits.
- 2026-11-26 (Step 2 - Numerical Quality Gate): Integrated main 5ac8ed0 on m1/quality-gate, preserving review/packet notes. Hardened evaluate_candidate_gate against missing/nonfinite/negative metrics, invalid baseline, empty/malformed predictions and invalid thresholds. Added tests/test_quality_gate.py (44 cases; before: 37 failed/7 passed; after focused: 53 passed; full non-network: 127 passed/1 skipped/1 deselected). Ruff passed with local tools excluded. No model retraining, final-test benchmark or Member 2 implementation.
- 2026-11-24 (M1.10 - MLflow Lineage Tracking, Model Registry Setup, Run Resolver CLI & Test Suite Network Isolation): Resolved all cryptographic lineage gaps identified in the artifact packet audit. (1) Implemented stdlib-only provenance helper src/models/lineage.py (log_lineage) logging git commit, git dirty status, data file hashes (hourly, hrrr, qh, feast), manifest hash, feature schema version & hash, configuration hashes (evaluation.yaml, model_params.yaml), and bundle sha256; integrated into train.py, evaluate.py, deviation.py, replay_monitoring.py, and materialize.py without third-party dependencies (compatible across .venv and .venv-feast). Added unit test suite in tests/test_lineage.py (7 passed). (2) Implemented scripts/tag_existing_runs.py retroactively tagging historical runs with verified provenance tags (prefix lineage_added_later. and timestamp lineage_added_later_at): verified champion artifact bundle sha256 (f72de152727fd406ac35a08cfa83289b6a61b031e0b3f4eb52008288e3980619) exactly matches models/solarcast_f1_bundle.joblib; verified logged data hashes match current disk files; honestly tagged unlogged hashes as unverified; documented champion run working-tree state (uncommitted changes at run time committed in 0afc5c6). (3) Implemented scripts/setup_model_registry.py: tested and confirmed MLflow 2.17.2 local file store supports aliases, tags, and descriptions; set alias champion on solarcast-f1 v4; set version tags (mode=F1, feature_set=B, schema_version=1.1.0, gate_eligible=true, final_test_opened=true, bundle_sha256, source_run_id); tagged v1-v3 status=superseded; updated descriptions with factual run history; saved status to reports/model_registry_status.json. (4) Implemented scripts/resolve_run.py resolving any run ID or champion to code commit, dirty flag, data hashes, schema version/hash, bundle hash, params, and metrics, verifying each against local disk with [OK], [MISMATCH], or [NOT RECORDED]. (5) Authored docs/mlflow_lineage.md documenting lineage architecture, run audit table, tuning run comparison, and shared server migration runbook. (6) Identified tests/test_data.py::TestHRRRExtraction::test_regression_june_week_sample as making live AWS S3 network calls via s3fs; added @pytest.mark.network and registered marker in pyproject.toml. Full test suite (83 tests) passes in 33.5s with -m "not network"; ruff clean. Branch m1/lineage.
- 2026-11-24 (M1.9 - Artifact Packet Generation, Verification CLI, Lineage Audit & Zip Archive): Implemented scripts/make_artifact_packet.py and scripts/verify_artifact_packet.py. Generated D:\SolarCast_packet containing 31 payload files (28,116,883 bytes, ~26.81 MB) preserving relative directory structure. Generated CHECKSUMS.sha256 and PACKET_MANIFEST.json with pipeline commands from docs/handoff_member2.md, git facts (commit, branch), python 3.11.9 and 20 key package versions, and local MLflow tracking facts. Conducted read-only MLflow lineage audit across champion run (f7fef3f7bdf241c5a53dc50d8e4713f9), solarcast-f1 registered versions (v1-v4, v4 champion), final_test_evaluation, deviation_detector, monitoring_replay, and feature_store_materialization. Created D:\SolarCast_packet.zip (13,535,820 bytes, ~12.91 MB, sha256 93504e87d870e7881235d3de1fbedbff712492ae567abb05107af5536e797e21). Verified integrity with scripts/verify_artifact_packet.py and verified that a deliberately corrupted copy fails with exit code 1. Updated docs/handoff_member2.md and docs/HANDOFF.md. All 77 unit tests pass; ruff clean. Branch m1/packet.
- 2026-11-24 (M1.8 - Operational Monitoring Engine, Input Drift, Delayed-Label Performance & Weather Quality): Implemented end-to-end monitoring and alerting architecture. (1) Built src/models/drift.py: implemented 10-quantile Population Stability Index (PSI) with hand-computed test example (PSI = 0.716704); implemented season-matched reference builder subsetting Train/Dev daylight observations by month (+/- 1 month) and daylight hours; created InputDriftMonitor monitoring 8 key numeric features (power_kw_lag_0m, energy_kwh_prev_day_same_hour, energy_kwh_rolling_mean_24h, clearsky_ghi_target_wm2, clearsky_index_latest_avail, tmp_c, tcdc_pct, clearsky_ghi_cloud_adj_wm2) with 200-row minimum floor, 2-consecutive-check persistence rule, and per-feature NaN reporting. (2) Built src/models/performance_monitor.py: implemented join_predictions_with_delayed_labels enforcing virtual-clock label visibility with 15-minute operational telemetry delay (arrival = interval_end + 15m <= T); implemented PerformanceMonitor evaluating 7-day daylight MAE with 30-label floor and distinguishing label_data_outage from model degradation (missing labels never treated as zero error). (3) Empirically compared fixed reference (31.43 kWh, 258 validation alert days, 170 in winter) vs. month-specific reference (127 validation alert days, 0 in Jan/Mar/Dec) and froze month-specific reference in configs/monitoring.yaml based on validation evidence. (4) Implemented weather freshness classifier (fresh <= 2h, fallback 2-8h, stale > 8h, missing) and WeatherFreshnessMonitor alerting when 7-day non-fresh weather exceeds 15%; strictly rejected illegal future weather (ValueError on age < 0). (5) Bumped configs/alert_event_schema.json to v1.1.0 with backward-compatible alert_type support; updated src/models/deviation.py, tests/test_deviation.py, docs/handoff_member2.md, and docs/handoff_member3.md. (6) Implemented scripts/replay_monitoring.py running 375-day chronological replay (2022-10-31 to 2023-11-09) across 4 controlled scenarios (seed=42): baseline unmodified run (drift: 8.8 alerts/100d, perf: 27.2 alerts/100d, DQ: 13.6 alerts/100d); weather shift (+8°C temp, x0.6 cloud, 21d: detected in 4.0d / 96h); performance drop (forecast x0.7, 10d: detected in 3.0d / 72h); label outage (actuals removed 5d: 0 accuracy alerts raised, status label_data_outage). Generated reports/monitoring_events.jsonl (329 records) and reports/monitoring_summary.json. (7) Logged MLflow run monitoring_replay (run_id: 16ffd7429be642928941e95f932fe8b3). (8) Authored docs/monitoring_signals.md and comprehensive test suite in tests/test_monitoring.py (12 tests). All 77 tests pass; ruff clean. Branch m1/monitors.
- 2026-11-26 (Step 1 - Artifact Packet): Verified supplied SolarCast_packet.zip, all 31 payload hashes/sizes and safe paths, installed 28 data/model files without overwrites, checked three matching existing config/report files and bundle lineage/data hashes. Added reports/packet_receipt_20261126.json and ignored the binary ZIP. Existing tests: 64 passed, 2 skipped under Python 3.12 review runtime. Fetched newer remote refs for source provenance; did not change code, reopen final benchmark, fix bugs or start Member 2 implementation. Stop after this step as requested.
- 2026-11-24 (Independent integration review): Pulled origin/main through dbd5f40, then reviewed d8c907f from m1/handoff-docs on local m1/member1-review. Added docs/member1_review_20261124.md with code references, test/probe evidence, scope alignment, artifact prerequisites and Member 2 work order. Core tests: 48 passed, 1 failed, 14 errors, 3 skipped (missing artifacts); scoped Ruff passed. No code fixes, model retraining, final-test reopening or GitHub push performed.
- 2026-11-24 (M1.7 - Complete Handoff Documentation, Real Pipeline Contract Samples & Defense Checklist): Delivered comprehensive handoff package for Members 2 and 3. (1) Created docs/handoff_member2.md containing exact CLI commands and arguments for ingestion, hourly dataset build, HRRR extraction, Feast feature table build, training, and deviation calibration; exit codes, failure handling, and idempotency guarantees; complete data snapshot tables (file paths, exact byte sizes, SHA-256 hashes matching configs/data_manifest.json and local disk parquets); detailed MLflow run lineage across experiments solarcast-model-training and solarcast; evaluation.json schema field-by-field breakdown with gate semantics; strict final-test safeguard instructions (final test is already opened: automation must NEVER call --force, use --dry-run instead); dual virtual environment architecture (.venv vs .venv-feast); CI pytest command; measured step runtimes; clean-clone reproduction checklist; and retraining DAG trigger workflow with 7-day cooldown. (2) Created docs/handoff_member3.md containing model bundle format and load_bundle/predict signatures; Feature Schema v1.1.0 Markdown table generated programmatically by scripts/gen_feature_doc.py; request-to-prediction inference and replay flow; API response contract and HTTP status codes (400, 422, 503); SQLite single-writer limitation and volume snapshot guide for replicas; prediction/actuals join log schema; low-cardinality Prometheus metrics; and alert event schema. (3) Implemented scripts/generate_contract_samples.py generating real contract sample files: docs/samples/predict_requests.json and docs/samples/predict_responses.json (5 real examples: clear morning, solar noon peak, nighttime 0.0 kWh forcing, missing weather fallback, and stale rejection HTTP 503), tests/fixtures/features_sample.csv (48 rows of features_set_b, 24.5 KB, with public data attribution), and tests/test_contract_samples.py (verifying bundle reproduces sample predictions within 1e-6 tolerance; 4/4 tests pass). (4) Authored docs/member1_defense_checklist.md providing one-paragraph explanations, exact tested PowerShell commands, and expected outputs for 7 core oral defense demonstrations, plus 5 documented limitations. (5) Updated docs/deviation_detector.md clarifying issue time H convention (start of target hour), production cutoff (H - 15m), alert trigger timing (when second anomalous hour has closed), and explicitly labeling fault injection metrics as synthetic. All 65 tests pass; ruff clean. Branch m1/handoff-docs.
- 2026-11-24 (M1.6 - Production Deviation Detector, Calibrated Thresholds & Fault Injection Benchmark): Implemented production deviation detection engine in src/models/deviation.py. (Step 1) Extracted out-of-sample validation residuals across 3 expanding folds (27,816 validation hours, 13,914 VALID daylight hours) using frozen champion HistGradientBoosting Set B specification. (Step 2) Calibrated frozen 1st and 99th percentile residual thresholds saved to configs/deviation.yaml (clear: [-140.66, 127.94], mixed: [-231.18, 226.24], overcast: [-203.70, 216.30], no_weather: [-155.80, 191.86] kWh); merged clear-sky index bins due to sub-200 sample counts in no_weather. Established configs/alert_event_schema.json with 2-hour persistence rule and reset on night/missing/non-VALID. (Step 3) Benchmarked on final test period copy: injected 30% drop (60% detected, 1.5h delay), zero output (95% detected, 1.0h delay), bad values (90% detected, 1.0h delay), isolated 1-hour drop (0% false triggers, 100% compliance); unmodified test baseline showed 0.43 false alerts per 1,000 daylight hours (2 alert events over 4,601 hours). Evaluated 43 natural SUSPECT_ZERO episodes across evaluation window (6/43 detected, 13.95%, median delay 3.0h; non-detected were sunrise/sunset low-sun hours where expected power < 140 kWh). (Step 4) Implemented tests/test_deviation.py (8/8 pass) and logged MLflow run deviation_detector (cce94db3448e405ab0fee01415e82da1). (Step 5) Authored docs/deviation_detector.md. All tests pass; ruff clean. Branch m1/deviation.
- 2026-11-24 (M1.5b - Feast Feature Store Integration, Parquet/SQLite Stores, Zero-Skew Provider & Tests): Implemented complete Feast feature store pipeline for Mode F1 next-hour solar forecasting. (Step 0) Purged unsupported causal claims across docs and documented P95 clear-sky ratio in model card. (Step 1) Checked feast dependencies, isolated Feast 0.68.0 to gitignored .venv-feast (numpy 2.4.6) with requirements-feast.txt, keeping core requirements.txt strictly untouched. (Step 2) Created src/features/build_feature_table.py generating data/processed/feast/features_set_b.parquet (23,532 daylight issue hours, 29 model features, 3 audit columns), confirming 100% compliance with audit timestamp <= event_timestamp non-leakage condition. (Step 3) Configured feature_store/feature_repo/{feature_store.yaml, entities.py, features.py, feature_views.py} and feature_store/materialize.py. Applied schema to local registry and verified Feast 0.68.0 TTL behavior. (Step 4) Implemented src/features/feast_provider.py and scripts/demo_feast.py executing 4 real-data demos: zero-skew point-in-time historical retrieval across 20 samples (max diff: 0.00e+00 vs build_features), SQLite online materialization, safe online serving with future and staleness rejection, and point-in-time replay helper. (Step 5) Implemented tests/test_feature_store.py with 9 comprehensive tests (auto-skipping in .venv, passing in .venv-feast) and logged MLflow materialization run (feature_store_materialization, run_id 56f349ec19f64a62917396b79a8e9fa9). Branch m1/feast.
- 2026-11-23 (Final Test Set Evaluation - One-Time Audited Run, Generalization Analysis & Report): Implemented unified evaluation orchestrator in src/models/evaluate.py supporting both dry-run (validation Fold 3) and official final test evaluation with CLI entrypoint (--dry-run, --final-test, --force, --reason). Verified end-to-end dry-run execution on Fold 3 without touching the test partition. Executed authorized one-time evaluation on the final 20% test partition (9,273 valid hours, 62 excluded) using models fit strictly on the 80% Train/Dev partition (reports/final_test_audit.json, MLflow run final_test_evaluation). All 5 baselines and 4 ML pipelines evaluated atomically. HistGradientBoosting Set B achieved 32.69 kWh Daylight MAE (3.66% cap-MAE) and 56.85 kWh RMSE on primary daylight hours (N = 4,387), outperforming baseline champion by +24.83% and Set A by +3.84%. On all daylight hours (N = 4,601), achieved 33.48 kWh Daylight MAE (+24.33% vs base). Evaluated cloud-cover bins: <20% clear (+6.15% gain for B), >80% overcast (+5.46% gain for B), 20-80% mixed (-3.21% difference; hypothesis: timing errors of forecast cloud on broken cloud; not tested). Evaluated monthly breakdown (Set B won in 12/14 months, reaching +11.81% in Nov 2023 [79 daylight hours]). Measured generalization gap: validation-to-test gap was +1.26 kWh (+4.01%) for HGB B, +1.75 kWh (+5.44%) for HGB A, and +1.39 kWh (+3.31%) for the baseline champion; differences are small. Analyzed 214 missing-weather hours (including 101 daylight hours in August 2023: cause unknown; observed only as missing files/chunks in the archive; note subsets differ: set A shows +0.69 kWh shift between the same subsets). Authored docs/final_test_report.md, updated docs/model_card.md, evaluation.json, reports/evaluation.json (final_test_opened: true), and docs/HANDOFF.md. All 53 tests pass; ruff clean. Branch m1/hrrr-forecast-features.
- 2026-11-23 (Prompt I Completion - Weather Feature Set B, Set A vs B Benchmark, Champion Selection & Bundling): Completed full evaluation of Feature Set B (29 features) incorporating NOAA HRRR forecast weather. Implemented meteorological valid range filtering (TCDC in [0, 100]%, TMP in [-40, 60]°C) in src/features/preprocessing.py sanitizing upstream fill value (-10,000 on 2020-07-29 21:00 UTC) to NaN with weather_missing=1, added unit test in tests/test_features.py, and documented in docs/weather_data_quality.md. Benchmarked Set A vs Set B across identical 3 chronological folds on 13,374 weather-present daylight hours (scripts/compare_set_a_b.py, reports/set_a_vs_b_summary.json): HistGradientBoosting Set B won with 31.43 kWh Daylight MAE (3.52% cap-MAE) and 53.35 kWh RMSE, outperforming Set A (32.24 kWh) by +2.52% (+0.81 kWh) and Clear-Sky Scaled 15m Persistence (42.09 kWh) by +25.33%, satisfying the >=1.0% preference rule and >=5.0% candidate gate (reports/evaluation.json and ./evaluation.json with final_test_opened: false). Stratified cloud-cover bins: <20% clear (+5.59% gain), >80% overcast (+4.94% gain), 20-80% mixed (-9.70% degradation due to NWP spatial/temporal phase timing errors). Computed permutation importance on validation folds (power_kw_lag_0m +144.04 kWh, hour_sin +51.94 kWh, clearsky_ghi_cloud_adj_wm2 +4.92 kWh, tcdc_pct +3.29 kWh, tmp_c +0.50 kWh). Packaged models/solarcast_f1_bundle.joblib with feature_set='B', needs_weather=True, schema 1.1.0, and added round-trip tests in tests/test_model.py. Logged runs to MLflow (experiment solarcast-model-training) and registered solarcast-f1 v4. Updated docs/model_report.md and docs/model_card.md. All 53 tests pass; ruff clean. Branch m1/hrrr-forecast-features.
- 2026-11-23 (M1.4 & M1.5 Refinement - Fair 15-Minute Information Set Baselines, Delay Sensitivity, and Permutation Importance): Added (d) Last 15-Minute Persistence (kw_lag0 * 1h) and (e) Clear-Sky Scaled 15-Minute Persistence (kt15 * clearsky_energy_target_kwh) to src/models/baselines.py with cascading fallbacks, night forced to 0, and clipping to [0, 893]. Evaluated all 5 baselines across 3 validation folds at delays 0m, 15m (default), and 60m: Clear-Sky Scaled 15-Minute Persistence is the undisputed champion (Delay 0m: 33.09 kWh / 3.71% cap; Delay 15m: 42.18 kWh / 4.72% cap / 67.61 kWh RMSE; Delay 60m: 67.50 kWh / 7.56% cap). Redefined champion baseline in src/models/train.py. Recomputed candidate gate: HistGradientBoosting achieved 32.24 kWh Daylight MAE (23.56% improvement over champion baseline, passing gate, registered as solarcast-f1 v2); Ridge achieved 43.52 kWh (-3.16%, rejected). Computed permutation importance on validation folds (top: power_kw_lag_0m +116.92 kWh, hour_sin +40.97 kWh, energy_kwh_prev_day_same_hour +15.03 kWh). Updated evaluation.json, reports/evaluation.json, docs/baseline_report.md, docs/model_report.md, docs/model_card.md. Added unit tests in tests/test_model.py. All 46 tests pass; ruff clean. Branch m1/hrrr-forecast-features.
- 2026-11-23 (M1.4 & M1.5 - Model Training, Benchmarking, Candidate Gating, MLflow Tracking, and Bundling): Implemented model hyperparameter tuning (src/models/tune.py), evaluation and gating (src/models/evaluate.py), model bundling (src/models/bundle.py), and end-to-end training orchestrator (src/models/train.py) with configs/model_params.yaml. Confirmed quarter_hourly.parquet and hourly.parquet timestamps and added test_hourly_kwh_equals_sum_of_quarter_hourly_kw in tests/test_data.py. Evaluated baseline delay sensitivity (0m, 15m default, 60m) in docs/baseline_report.md. Tuned Ridge Regression and HistGradientBoostingRegressor (10 iterations each) across 3 chronological expanding folds on 80% Train/Dev partition (37,090 hours). HistGradientBoosting won with 32.24 kWh Daylight MAE (3.61% cap-MAE) and 54.76 kWh RMSE, outperforming Clear-Sky Persistence (75.62 kWh) by 57.36% and Ridge (43.52 kWh) by 25.9%. Candidate gate passed (reports/evaluation.json and ./evaluation.json with final_test_opened: false). Enforced strict check_final_test_safeguard() protecting locked 20% test partition (9,273 hours). Trained final winner on complete Train/Dev dataset; generated diagnostic plots (actual_vs_predicted, residuals, error_by_hour, error_by_month); saved standalone bundle models/solarcast_f1_bundle.joblib with load_bundle() and bounded .predict(); logged metrics, params, data sha256 hashes, Git commit, schema version, and artifacts to MLflow (file:./mlruns); registered model solarcast-f1 v1 (status: candidate). Authored docs/model_report.md and docs/model_card.md. Added unit test suite in tests/test_model.py (6 tests). All 44 tests pass; ruff clean. Branch m1/hrrr-forecast-features.
- 2026-11-23 (M1.3c - HRRR Extractor Diagnosis, Flexible CLI Options & Google Colab Kit): Completed network diagnostic isolating throughput bottleneck: 1MB chunk GET took 84.2s (11.8 KB/s) with 1.74s median TTFB; 4 threads achieved 60.9 KB/s while 16 threads collapsed to 18.9 KB/s due to packet drops and socket contention; verified code does not serialize work (thread-local S3FileSystem, lock only on int counter, 10 S3 round-trips/hour for 5 vars). Added CLI options to scripts/hrrr_extract.py: --start, --end, --variables (default 'tmp,tcdc'; allows lcdc,mcdc,hcdc), --parts-dir (default 'data/processed/hrrr_parts'), and --threads. Preserved schema with unrequested columns set to NaN and extraction_status recording requested variables (e.g. 'SUCCESS:tmp,tcdc'). Built Google Colab kit: created requirements-extract.txt and notebooks/colab_hrrr_extract.ipynb with 8 structured cells (Drive mount, git clone & checkout, dependencies, copy inputs from Drive, self-test verifying 2022-06-13 00:00 UTC at 25.6°C, configurable extraction to Drive parts, progress tracking, and merge). Staged inputs (hourly.parquet and hrrr_coverage_scan.parquet) to D:\SolarCast_inputs\. Established staged extraction order (Priority 1: 2022-10-17 to 2023-11-09, Priority 2: 2021-10-17 to 2022-10-16, Priority 3: older). Added 2 unit tests in tests/test_data.py. All 42 tests pass; ruff clean. Branch m1/hrrr-forecast-features.
- 2026-11-22 (M1.3b - HRRR Forecast Feature Extraction Pipeline & Multi-Threading): Implemented src/data/hrrr.py and scripts/hrrr_extract.py to extract 2m temperature (TMP, converted K to C) and cloud cover (TCDC, LCDC, MCDC, HCDC) from MesoWest HRRR Zarr (s3://hrrrzarr/sfc/) for all daylight hours between 2018-07-12 and 2023-11-09. Adheres to operational availability lag of 2 hours (issue time H, eligible run init <= H - 2h) with fallback up to 6 hours (H-8h max). Fast-paths run existence via precomputed data/processed/hrrr_coverage_scan.parquet. Uses 16 worker threads, zero disk chunk caching (in-memory extraction), and saves monthly resumable parts under data/processed/hrrr_parts/ with CLI merge flag. Added 6 comprehensive unit/integration tests in tests/test_data.py (selection rule, 2h lag, fallback, max fallback exceeded, unit conversion, part merging, and live regression test against june_week_extracted.parquet within 0.1 float tolerance). Benchmarked sample execution (~0.075-0.10 req/s remote residential throughput; ETA ~320 hours residential vs ~3-5 hours on EC2 us-west-1). Added logs/ to .gitignore. All 41 tests pass; ruff clean. Branch m1/hrrr-forecast-features.
- 2026-11-22 (M1.4 - Baseline Models & Chronological 3-Fold Evaluation): Implemented BaseBaseline, LatestHourPersistence, SameHourPreviousDayPersistence, and ClearSkyPersistence in src/models/baselines.py with cascading missing-history fallbacks and strict nighttime zeroing. Configured configs/evaluation.yaml with 80/20 chronological split (Train/Dev: 37,090 hours; locked Final Test: 9,273 hours) and 3 expanding-window folds. Evaluated baselines on validation folds: ClearSkyPersistence is the champion with 75.62 kWh Daylight MAE (8.47% cap-MAE), 113.80 kWh RMSE, and 37.87 kWh All-Hour MAE. Wrote docs/baseline_report.md including complete monthly breakdowns. Added tests/test_model.py. All 35 tests pass; ruff clean. Branch m1/features-baselines.
- 2026-11-22 (M1.3 - Quarter-Hourly Aggregation & Zero-Leakage Feature Engineering): Extended src/data/aggregation.py with aggregate_to_quarter_hourly() producing data/processed/quarter_hourly.parquet (240,384 intervals). Created src/features/preprocessing.py with ONE shared build_features() function producing 23 past-only features (4 quarter-hourly lags, 4-lag mean, latest fully available hourly kWh, same hour prev day, same hour prev week, past-only rolling means, hour/DOY sin/cos, target hour solar elevation & pvlib clearsky energy, latest clear-sky index, age and missingness flags). Enforced 15-minute operational cutoff (cutoff = H - 15m). Configured configs/features.yaml and configs/feature_schema.json (schema_version 1.0.0). Created tests/test_features.py with rigorous future-data perturbation test, DST fall-back test, missing hour fallback test, and prehistory test. All tests pass; ruff clean. Branch m1/features-baselines.
- 2026-11-22 (M1.1b - HRRR Gate Finalization: Steps 1-6 Variable Inventory, Multi-Year Scan, Signal Check, GRIB2 Audit & Gate Approval): Executed full 6-step feasibility protocol on branch m1/hrrr-gate. (1) Variable inventory on run 2022-06-13 12z confirmed chunk 1.4 has non-empty chunks for TCDC (476 KB), LCDC, MCDC, HCDC, TMP (1.01 MB), DPT, RH, UGRD, VGRD, VIS, PRATE, APCP (12 KB, 1-hr acc); confirmed DSWRF is missing chunks due to iris-grib #265 bug. (2) Executed full multi-year coverage scan across all 46,728 hourly runs (2018-07-12 to 2023-11-09) with 16 threads and cheap HEAD requests; verified 94.18% availability for TMP and 94.07% for TCDC (v3: 93.65% TMP / 93.52% TCDC; v4: 94.61% TMP / 94.51% TCDC); saved resumable data/processed/hrrr_coverage_scan.parquet (424 KB). (3) Selected mixed-weather week (2022-12-26 to 2023-01-01, 168 hours, 100% VALID, atmospheric river storm, maximal kt_proxy std=0.400); extracted plant TMP and TCDC with lag=2h into data/processed/mixed_week_extracted.parquet (16 KB) and June sample into june_week_extracted.parquet (17 KB) in-memory with zero disk chunk caching. (4) Conducted signal check for VALID high_sun hours, demonstrating strong true weather forecast skill: Spearman rho = -0.7884 (p=5.46e-13), Pearson r = -0.9044 between forecast TCDC and clear-sky index proxy; generated docs/tcdc_clearsky_proxy_correlation.png; documented diurnal cycle distinction. (5) Tested pip install eccodes cfgrib in Windows Python 3.11 with pre-built wheels, successfully decoded a byte-range DSWRF GRIB2 message in-memory without disk caching (night 2.10 MB, noon 1.88 MB, 1010 W/m² at solar noon); confirmed WSL2 Ubuntu exists; estimated GRIB2 DSWRF volume at ~34.8 GB/yr (~185.7 GB full period). (6) Updated docs/data_feasibility.md with all findings labeled verified/assumed/unresolved, purged 404 citations (SCN 18-18 and SCN 20-91) retaining only active primary URLs, and approved Route 1 (Hybrid Forecast TMP + Cloud Cover + pvlib Clear-Sky Irradiance) as primary Mode F1 strategy. All 23 tests pass; ruff clean.
- 2026-11-22 (M1.1b - NOAA HRRR Verification Gate & Feasibility Audit): Completed comprehensive feasibility audit of candidate forecast weather source NOAA HRRR across AWS GRIB2 archive (s3://noaa-hrrr-bdp-pds/) and MesoWest Zarr archive (s3://hrrrzarr/). Authored docs/data_feasibility.md, scripts/hrrr_feasibility.py (reproducible, idempotent), docs/data_feasibility_preview.csv (48 daylight rows with exact 15-column schema), and docs/hrrr_pv_correlation.png. Verified spatial mapping (334 m to plant, chunk 1.4, offsets 48, 42); audited GRIB2 .idx semantics confirming instantaneous point forecasts mapped to hourly targets via trapezoidal numerical integration; verified 2-hour operational release lag (81/104 daylight hours show fresh updates vs lag 3). Discovered critical upstream flaw in MesoWest Zarr (all forecast DSWRF chunks omitted due to iris-grib #265 bug) and absence of pre-July 2018 forecast runs in Zarr. Demonstrated strong physical correlation between measured AC energy and pvlib clear-sky GHI (r=0.945) and HRRR analysis DSWRF (r=0.927). Issued CONDITIONAL recommendation (use HRRR Zarr 2m TMP + pvlib clear-sky irradiance + past-only production lags for Mode F1; retain NASA POWER for retrospective Mode E1). All 23 tests pass; ruff clean. Branch m1/hrrr-gate.
- 2026-11-22 (M1.2 Refinement - Midpoint Solar Elevation, High-Sun & Zero Generation Audit): Shifted solar elevation evaluation to interval midpoint (+30m), added high_sun (>5 deg) boolean indicator, and updated Pandera hourly_schema. Conducted systematic daytime zero audit by elevation bin and year (1,439 daytime zeros), cataloging 55 multi-hour zero runs (>10 deg) and confirming absence of daytime frozen readings. Added SUSPECT_ZERO quality flag for 448 hours in runs of 2+ high-sun zero hours. Created tests/fixtures/hourly_sample.csv (720 rows, July 2022 excerpt, <200 KB) with DOI attribution in README.md. Replaced unsupported cause statements in docs with objective data observations. All 23 tests pass; ruff clean.
- 2026-11-21 (M1.2 - Validation, Quarantine & Hourly Dataset Aggregation): Implemented Pandera schemas in src/data/validation.py and aggregation logic in src/data/aggregation.py. Quarantined single erroneous reading (1,783.68 kW on 2018-02-18 12:30) into data/processed/quarantine.parquet. Localized naive civil timestamps (America/Los_Angeles) to UTC, resolving ambiguous autumn fall-back timestamps to PDT and logging the unrecorded hour as missing. Aggregated 15-minute average power into continuous 60,096-row hourly dataset (data/processed/hourly.parquet) with pvlib solar elevation, boolean daylight flags, and quality_flag auditing (60,013 valid, 80 missing, 3 incomplete). Updated src/data/ingestion.py to output POSIX paths and verified units ('kW, 15-minute average'); regenerated data/raw/manifest.json and copied to configs/data_manifest.json (tracked). Authored docs/data_quality_report.md. Added unit test suite in tests/test_data.py (10 tests; 21 total tests in repo pass; ruff clean). Branch m1/hourly-dataset.
- 2026-11-21: Updated README scope and ten-item Member 1 HRRR verification checklist, preserved upstream setup instructions, and synchronized agent/handoff scope. No HRRR files were downloaded and no dataset acceptance or model result is claimed. Reconcile with existing m1/data-feasibility research before continuing.
- 2026-11-16 (M1.1 Approved & Ingestion Implemented): System 2107 approved. Completed deep verification audit in docs/data_card.md (DST 75-min spring skip explanation, solar-noon alignment test proving America/Los_Angeles timezone, AC statistics and capacity normalization rationale, aggregation type 'avg'). Archived one-off scripts to scripts/exploration/. Created configs/data.yaml and src/data/ingestion.py with retry/backoff, idempotent download check, and manifest generator. Fetched complete 2017–2023 NASA POWER hourly weather and generated data/raw/manifest.json. Added tests/test_ingestion.py (11 tests pass, ruff clean). Committed on branch m1/data-ingestion.
- 2026-11-16 (M1.1 - Data Feasibility): Completed feasibility audit of PVDAQ prize systems 2107, 2105, 9068 on OEDI S3 bucket and NASA POWER Hourly API. Created docs/data_card.md with full comparison matrix, DST analysis, parameter verification, and recommendation for System 2107. Downloaded metadata and small sample (~6.3 MB meter data, well under 500 MB budget) in data/raw/. Committed on branch m1/data-feasibility.
- 2026-11-15 (P2 - Environment & Requirements): Added pinned requirements.txt (16 packages for Python 3.11). Created .venv, installed requirements, verified pytest (4 passed) and ruff check . (clean). Fixed import sorting in tests/test_smoke.py and updated run instructions in docs/HANDOFF.md.
- 2026-11-15 (P1 - Scaffolding): Completed Member 1 package scaffold on branch m1/scaffold: created package directories (configs, data/{raw,processed}, src/{data,features,models}, feature_store/feature_repo, tests/fixtures, evidence), added empty __init__.py in src packages and .gitkeep in empty folders, added .env.example, configured pyproject.toml with ruff line-length 100, added tests/test_smoke.py (tests pass), updated .gitignore with data/ and mlruns/, extended README.md with Member 1 setup section.
- 2026-11-13: File created. Sources reviewed. No code written yet.

## 12. Next steps
0. Artifact handoff, numerical gate validation, isolated exact-candidate checks, provider target-hour
   safeguards and alert continuity/confirmation are complete; latest main
   5ac8ed0 is integrated. Continue `docs/member1_review_20261124.md` priorities one at a time:
   P95 provenance (R7), remaining R8 assumptions and real Feast/environment verification. Future
   promotion must separately verify the exact eligible candidate and never refit under its approval. The existing
   Member 1 handoffs are implementation guidance, not independent proof of clean-clone readiness.
1. **Member 2 Review & Integration:**
   - Review `docs/handoff_member2.md` and verify DVC tracking tables and clean-clone checklist.
   - Configure CI/CD pipeline using `pytest -m "not network"`.
   - Implement Airflow / KFP retraining DAG with trigger webhook consuming `configs/alert_event_schema.json` (v1.1.0) and retraining trigger policy (24-hour cooldown, at most one active run per site, minimum new labels).
2. **Member 3 Review & Serving Deployment:**
   - Review `docs/handoff_member3.md`, bundle loader signatures, and `docs/samples/predict_requests.json`.
   - Implement Flask serving API and load-balancer mounting `online_store_snapshot.db` read-only.
   - Configure prediction and actuals logging schemas (`predictions.jsonl`, `actuals.jsonl`) for continuous evaluation.
3. **Oral Defense Rehearsal:**
   - Review demonstrations and commands in `docs/member1_defense_checklist.md`.
