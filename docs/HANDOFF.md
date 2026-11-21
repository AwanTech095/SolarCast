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
| M1.1 | Data feasibility and site choice | In progress | Partner research exists on m1/data-feasibility; System 2107 remains a candidate. HRRR access, fields, release timing and production alignment need the README verification report before sign-off. |
| M1.2 | Validation + hourly dataset | Not started | |
| M1.3 | Features + Feast (offline/online) | Not started | |
| M1.4 | Baselines, Ridge, HistGB, tuning | Not started | |
| M1.5 | MLflow lineage, evaluation.json, model bundle | Not started | |
| M1.6 | Forecast deviations + supporting E1 benchmark + replay test | Not started | Core alerts use stored F1 forecasts; keep any delayed E1 results separate. |
| M1.7 | Handoff docs for Members 2 and 3 | Not started | |

Status values: Not started / In progress / Done (tests pass) / Blocked (say why).

## 3. Data sources (proposed; final site NOT yet chosen)
Sources below were originally reviewed on 2026-11-13. Member 1 has since reported a production audit on
`m1/data-feasibility`; preserve and reconcile that branch's data card rather than treating its work as not started.
Raw data has not been independently validated in this workspace. The new HRRR forecast-data route is pending verification.

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
  The earlier plan used a simulated 14-day lag for F1. That is superseded as the core forecast-data route;
  NASA now supports retrospective analysis and a separately evaluated E1 benchmark.

**Forecast weather: NOAA HRRR archive (new candidate).**
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
- [ ] Site/dataset sign-off by all three after M1.1

## 9. Risks / known issues
- Data access and size (prize datasets can be tens of GB): sample first, then scale.
- Timestamp/unit errors (local vs UTC, DST, left/center/right interval alignment, kW vs kWh): verify with fixtures.
- HRRR forecast fields, publication timing and interval alignment remain unverified; do not begin full training
  or claim an operationally faithful backtest before the feasibility gate is reviewed.
- Dependency conflicts between Feast, MLflow, Airflow, KFP, BentoML: keep separate environments.

## 10. Decision log
| Date | Decision | Why |
|---|---|---|
| 2026-11-21 | User authorized revised weather-driven F1 scope; HRRR is a candidate, NASA supports retrospective analysis | Preserve the assigned weather/irradiance/temperature objective without future-observation leakage. Member 1 must verify and report before source approval. |
| 2026-11-13 | Preferred route: PVDAQ via OEDI + NASA POWER (UTC) | Named in team plan; sources reviewed |
| 2026-11-13 | First candidate system 2107; backups 2105, 9068 | Mid-size fixed system with long history; not yet validated |
| 2026-11-15 | Member 1 scaffold & directory layout | Created Member 1 package structure on branch m1/scaffold with empty __init__.py files, .gitkeep in empty folders, ruff config (line-length 100), .env.example, and test_smoke.py |
| 2026-11-15 | Pinned stack in requirements.txt | Pinned 16 core Member 1 packages for Python 3.11 (pandas 2.2.2, numpy 1.26.4, scikit-learn 1.5.2, scipy 1.13.1, mlflow 2.17.2, pvlib 0.11.1, pandera 0.20.4, etc.), excluding feast/airflow/kubeflow/bentoml to avoid early dependency conflicts |

## 11. Changelog
- 2026-11-21: Updated README scope and ten-item Member 1 HRRR verification checklist, preserved upstream setup instructions, and synchronized agent/handoff scope. No HRRR files were downloaded and no dataset acceptance or model result is claimed. Reconcile with existing m1/data-feasibility research before continuing.
- 2026-11-13: File created. Sources reviewed. No code written yet.
- 2026-11-15 (P1 - Scaffolding): Completed Member 1 package scaffold on branch m1/scaffold: created package directories (configs, data/{raw,processed}, src/{data,features,models}, feature_store/feature_repo, tests/fixtures, evidence), added empty __init__.py in src packages and .gitkeep in empty folders, added .env.example, configured pyproject.toml with ruff line-length 100, added tests/test_smoke.py (tests pass), updated .gitignore with data/ and mlruns/, extended README.md with Member 1 setup section.
- 2026-11-15 (P2 - Environment & Requirements): Added pinned requirements.txt (16 packages for Python 3.11). Created .venv, installed requirements, verified pytest (4 passed) and ruff check . (clean). Fixed import sorting in tests/test_smoke.py and updated run instructions in docs/HANDOFF.md.

## 12. Next steps
1. Bring this scope update into Member 1's existing research branch without discarding its data-card findings.
2. Complete the README HRRR verification checklist on one overlapping week, then assess seasonal coverage and cost.
3. Submit docs/data_feasibility.md and a reproducible script/sample, marking verified/assumed/unresolved findings;
   recommend GO, CONDITIONAL or NO-GO and stop for team review before full training/site sign-off.
