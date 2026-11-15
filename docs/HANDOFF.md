# SolarCast: Member 1 Handoff (living document)

**Owner:** Emad (Member 1: Data, features, Feast, models, MLflow). **Review partner:** Member 2.
**Last updated:** 2026-11-13 (project start). Update this file after every task (see AGENTS.md).
Anyone picking up this package should be able to continue from this file alone.

## 1. What this project is
Forecast the next hourly AC energy output (kWh) for **one** PV site (mode **F1**), and estimate expected output from observed
weather to flag production deviations (mode **E1**). Course goal: a reproducible, tested, deployed and monitored MLOps
system. Accuracy is secondary. Source documents: `MLOps_Course_Project_Requirements.docx` (13 pp.) and
`SolarCast_Project_and_Team_Plan.pdf` (14 pp.). **Where they conflict, the course requirements win.**
Known spec quirk: the example repo tree mentions FastAPI, but the endpoint section requires **Flask**. Keep Flask.

## 2. Status board
| ID | Item | Status | Notes |
|---|---|---|---|
| M1.0 | Repo + environment setup | Done (tests pass) | Scaffold created on branch m1/scaffold; Member 1 folder layout, .gitkeep, __init__.py, pyproject.toml, .env.example, pinned requirements.txt, and tests passing (pytest & ruff clean) |
| M1.1 | Data feasibility and site choice | Not started | Candidates in section 3 |
| M1.2 | Validation + hourly dataset | Not started | |
| M1.3 | Features + Feast (offline/online) | Not started | |
| M1.4 | Baselines, Ridge, HistGB, tuning | Not started | |
| M1.5 | MLflow lineage, evaluation.json, model bundle | Not started | |
| M1.6 | E1 estimator + deviation rule + replay test | Not started | |
| M1.7 | Handoff docs for Members 2 and 3 | Not started | |

Status values: Not started / In progress / Done (tests pass) / Blocked (say why).

## 3. Data sources (proposed; final site NOT yet chosen)
Sources read on 2026-11-13. Nothing has been downloaded or validated yet.

**Target (measured production): NLR/NREL PVDAQ via OEDI data lake.**
- Docs: https://github.com/openEDI/documentation/blob/main/pvdaq.md. DOI 10.25984/1846021 (cite it).
- Parquet on S3: `s3://oedi-data-lake/pvdaq/` (tables: system, site, metrics, meters, inverters, mount...) and
  `s3://oedi-data-lake/pvdata/system_id=/year=/month=/day=/*.parquet` for the time series.
- Typical resolution is 15 minutes, but it varies by system. Schema may change across years.
- Key tables: `pvdaq_system` (power = DC nameplate kW), `pvdaq_site` (latitude, longitude, elevation),
  `pvdaq_metrics` (units, scale/offset, `common_name` such as "AC energy"), `pvdaq_meters`/`pvdaq_inverters`
  (`time_interval` = left/center/right alignment), `pvdaq_pvdata` (`measured_on` local, `utc_measured_on`, `metric_id`, `value`).
- The `utc_measured_on` field "could include DST" per the docs, so verify it against the local timestamp.
- Old PVDAQ V3 API is **decommissioned** (https://developer.nlr.gov/docs/solar/). Bulk download tool: PVDAQ Access
  (https://github.com/NatLabRockies/pvdaq_access). Data map: https://openei.org/wiki/PVDAQ/PVData_Map
- Prize datasets (5+ years, at least 15-min), stored differently (one directory per site, may be split by channel, some tens of GB):
  - **2107**: 893 kW fixed ground-mount, California. **First candidate.**
  - **2105**: 237 kW rooftop, several orientations, Hawaii. Backup (mixed orientations make it messier).
  - **9068**: 4.7 MW single-axis tracker, Colorado. Backup.
  - 7333 (257 MW, 10 s data) and 9069 (38.7 MW) are too large; avoid.
- Open checks before accepting a site: AC energy/power channel exists, units, interval alignment, timezone, coverage and
  gaps, licence, whether the S3 bucket needs no credentials.

**Weather: NASA POWER hourly API.**
- API: `/api/temporal/hourly/point` at https://power.larc.nasa.gov/ (docs: https://power.larc.nasa.gov/docs/services/api/temporal/hourly/).
- **Default time standard is Local Solar Time. Always pass `time-standard=UTC`.** Hourly data from 2001 to near real time.
- Max 15 parameters per request. Candidate fields: `ALLSKY_SFC_SW_DWN`, `T2M`, `RH2M`, `WS10M` (verify availability and units).
- Registry page: https://registry.opendata.aws/nasa-power/. Licence: no restrictions, **cite NASA POWER**. It is CC BY 4.0.
  Update lag: meteorological ~2-3 days, solar ~5-7 days. So it is an archive of observations, **not a next-hour forecast**.
  The plan's simulated 14-day lag is conservative. Run 7/14/30-day sensitivity.

**Fallback if no usable measured site is found:** another public measured-generation source with known coordinates and terms.
Synthetic data is allowed only as a clearly labeled pipeline-demo fallback.

**Honest expectation:** with lagged weather, F1 skill will come mostly from production lags (close to persistence). The
5% improvement gate over the stronger persistence baseline may fail. If so, report it and keep the baseline as champion.
The spec rewards the process (gates, blocked promotion), not high accuracy.

## 4. Key design decisions (from the team plan; proposed values, not course-mandated)
- F1 = next-hour forecast. E1 = weather-adjusted expected output for delayed deviation analysis. F2 (archived forecasts) is out of scope.
- Split: last 20% held out as a final test set. Earlier 80%: 3 expanding-window folds. Test is opened once.
- Primary metric: daylight MAE (kWh). Candidate gate: >=5% better than the stronger persistence baseline and
  capacity-normalized daylight MAE <= 0.20. Retraining gate: >=2% better than champion and no seasonal slice >5% worse.
- Deviation rule: residual = actual - expected; 1st/99th percentile bounds per irradiance bin, calibrated on a reserved
  chronological period; flag after 2 consecutive valid daylight hours outside bounds; frozen after calibration.
- Required models: persistence (previous hour, same hour previous day), Ridge, HistGradientBoostingRegressor. No LSTM needed.
- Tools for Member 1 in the spec: Pandera or Great Expectations (validation, optional), Feast (required, rubric 6 marks),
  MLflow (9 marks, shared with registry), Evidently or a custom drift detector (PSI >0.20 in >=2 features is the proposed rule).

## 5. Contracts I own (fill in when built; link the file)
| Contract | File | Status |
|---|---|---|
| Hourly data schema + manifest | `configs/data.yaml`, `data/processed/manifest.json` | TBD |
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

# Run test suite
pytest
# (or directly: .\.venv\Scripts\pytest)

# Code quality and style checks
ruff check .
# (or directly: .\.venv\Scripts\ruff check .)
```

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
- [ ] Site/dataset sign-off by all three after M1.1

## 9. Risks / known issues
- Data access and size (prize datasets can be tens of GB): sample first, then scale.
- Timestamp/unit errors (local vs UTC, DST, left/center/right interval alignment, kW vs kWh): verify with fixtures.
- Weather lag makes F1 close to persistence; gate may fail (acceptable if reported).
- Dependency conflicts between Feast, MLflow, Airflow, KFP, BentoML: keep separate environments.

## 10. Decision log
| Date | Decision | Why |
|---|---|---|
| 2026-11-13 | Preferred route: PVDAQ via OEDI + NASA POWER (UTC) | Named in team plan; sources reviewed |
| 2026-11-13 | First candidate system 2107; backups 2105, 9068 | Mid-size fixed system with long history; not yet validated |
| 2026-11-15 | Member 1 scaffold & directory layout | Created Member 1 package structure on branch m1/scaffold with empty __init__.py files, .gitkeep in empty folders, ruff config (line-length 100), .env.example, and test_smoke.py |
| 2026-11-15 | Pinned stack in requirements.txt | Pinned 16 core Member 1 packages for Python 3.11 (pandas 2.2.2, numpy 1.26.4, scikit-learn 1.5.2, scipy 1.13.1, mlflow 2.17.2, pvlib 0.11.1, pandera 0.20.4, etc.), excluding feast/airflow/kubeflow/bentoml to avoid early dependency conflicts |

## 11. Changelog
- 2026-11-13: File created. Sources reviewed. No code written yet.
- 2026-11-15 (P1 - Scaffolding): Completed Member 1 package scaffold on branch m1/scaffold: created package directories (configs, data/{raw,processed}, src/{data,features,models}, feature_store/feature_repo, tests/fixtures, evidence), added empty __init__.py in src packages and .gitkeep in empty folders, added .env.example, configured pyproject.toml with ruff line-length 100, added tests/test_smoke.py (tests pass), updated .gitignore with data/ and mlruns/, extended README.md with Member 1 setup section.
- 2026-11-15 (P2 - Environment & Requirements): Added pinned requirements.txt (16 packages for Python 3.11). Created .venv, installed requirements, verified pytest (4 passed) and ruff check . (clean). Fixed import sorting in tests/test_smoke.py and updated run instructions in docs/HANDOFF.md.

## 12. Next steps
1. Run M1.1: Data feasibility and site choice (PVDAQ candidate system 2107 vs backups 2105, 9068 + NASA POWER hourly API) and stop for site sign-off.
