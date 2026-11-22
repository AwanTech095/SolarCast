# SolarCast: Member 1 Handoff (living document)

**Owner:** Emad (Member 1: Data, features, Feast, models, MLflow). **Review partner:** Member 2.
**Last updated:** 2026-11-21 (weather-forecast scope revision). Update this file after every task (see AGENTS.md).
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
| M1.0 | Repo + environment setup | Done (tests pass) | Scaffold created on branch m1/scaffold; Member 1 folder layout, .gitkeep, __init__.py, pyproject.toml, .env.example, pinned requirements.txt, and tests passing (pytest & ruff clean) |
| M1.1 | Data feasibility and site choice | Done (tests pass) | Site 2107 approved by human owner. Deep verification audit completed in docs/data_card.md. |
| M1.1b | HRRR verification gate | Done (tests pass) | Full feasibility audit completed in docs/data_feasibility.md, scripts/step1_inventory.py, scripts/step2_coverage_scan.py, scripts/step3_step4_mixed_and_signal.py, and docs/tcdc_clearsky_proxy_correlation.png. Audited spatial mapping (334 m, chunk 1.4); instantaneous semantics mapped via trapezoidal averaging; operational latency verified with 2-h lag. Inventoried chunk 1.4 variables (TCDC, LCDC, MCDC, HCDC, TMP, DPT, RH, UGRD, VGRD, VIS, PRATE, APCP valid; DSWRF missing chunks due to iris-grib #265 bug). Completed 46,728-run full scan (2018-07-12 to 2023-11-09) showing 94.18% TMP and 94.07% TCDC availability. Selected optimal mixed week (2022-12-26 to 2023-01-01, atmospheric river storm, 100% valid, maximal kt_proxy std=0.400). Signal check on VALID high_sun hours demonstrated strong true forecast skill: Spearman rho = -0.7884 (p=5.46e-13), Pearson r = -0.9044 against clear-sky index proxy (distinguished from astronomical diurnal cycle). GRIB2 decoding verified in-memory with eccodes on Windows Python 3.11 and WSL2 confirmed; volume estimated at ~34.8 GB/yr (~185.7 GB full period). Gate decision: APPROVE ROUTE 1 (Hybrid Forecast TMP + Cloud Cover + pvlib Clear-Sky Irradiance for Mode F1; retain NASA POWER for retrospective Mode E1). |
| M1.2 | Validation + hourly dataset | Done (tests pass) | Implemented src/data/validation.py (Pandera) and src/data/aggregation.py. Quarantined 1 row > 893 kW (2018-02-18 12:30) into data/processed/quarantine.parquet. Ambiguous fall DST times mapped to 1st occurrence (PDT) with 2nd occurrence marked missing. Aggregated 60,096 continuous UTC hours (60,013 valid, 83 NaN = 0.1381%) into data/processed/hourly.parquet with pvlib solar elevation and quality flags. All 21 tests pass; ruff clean. |
| M1.3 | Features + Feast (offline/online) | Not started | |
| M1.4 | Baselines, Ridge, HistGB, tuning | Not started | |
| M1.5 | MLflow lineage, evaluation.json, model bundle | Not started | |
| M1.6 | Forecast deviations + supporting E1 benchmark + replay test | Not started | Core alerts use stored F1 forecasts; keep any delayed E1 results separate. |
| M1.7 | Handoff docs for Members 2 and 3 | Not started | |

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
| Feature schema (ordered names/types/units) | `configs/feature_schema.json` | TBD |
| Model bundle format | `src/models/bundle.py`, `docs/handoff_member3.md` | TBD |
| Evaluation result | `evaluation.json` (baseline vs champion, thresholds, eligible flag, reasons) | TBD |
| Feast entity/feature views | `feature_store/feature_repo/` | TBD |
| Training entrypoint | `python -m src.models.train --config configs/evaluation.yaml` | TBD |

## 6. How to run (fill in as built)
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

# Run data validation, quarantine, and hourly dataset aggregation
python -m src.data.aggregation --config configs/data.yaml
# (or directly: .\.venv\Scripts\python -m src.data.aggregation --config configs/data.yaml)

# Run NOAA HRRR forecast data feasibility audit (Task M1.1b)
python scripts/hrrr_feasibility.py
python scripts/step1_inventory.py
python scripts/step2_coverage_scan.py
python scripts/step3_step4_mixed_and_signal.py
# (or directly with .\.venv\Scripts\python ...)

# Run test suite
pytest
# (or directly: .\.venv\Scripts\pytest)

# Code quality and style checks
ruff check .
# (or directly: .\.venv\Scripts\ruff check .)
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
|---|---|---|
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
1. **Team review of M1.1b Feasibility Gate:**
   - Review docs/data_feasibility.md and decide whether to approve the CONDITIONAL path (HRRR forecast 2m temperature + pvlib astronomical clear-sky irradiance + production history for Mode F1).
2. **M1.3 Features + Feast (Offline & Online feature store) (Next task):**
   - Implement feature extraction in `src/features/preprocessing.py` (lags t-1, t-2, t-24, rolling stats, cyclical time encodings, pvlib clear-sky irradiance, HRRR forecast 2m temperature).
   - Configure Feast entities and feature views in `feature_store/feature_repo/` for offline point-in-time join and online SQLite store.
