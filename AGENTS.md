# AGENTS.md: Rules for the AI coding agent (read this first, every session)

Project: **SolarCast**, an MLOps course project (3 students). Forecast next-hour solar AC energy (kWh) for one PV site,
and flag production deviations. This file covers the **Member 1 package**: data, features, Feast, models, MLflow.
The human owner of this package does **not write code**. Write complete, runnable code and explain how to run it.

## Working rules
1. Read `docs/HANDOFF.md` before starting. **Update it at the end of every task** (status table, decisions, how-to-run,
   changelog). A new teammate must be able to continue from that file alone.
2. Do one task at a time. Finish it, run the tests, then stop and report. Do not start the next task unasked.
3. Never invent data, numbers, metrics, file contents or download results. If a download, command or test fails, say so
   and show the real error. Do not claim something works unless you ran it.
4. Ask before making a decision that changes scope: choice of dataset/site, a new dependency, renaming a contract file.
5. Small commits on a feature branch (`m1/<topic>`). Message format: `type(scope): summary`, e.g. `feat(data): add PVDAQ loader`.
   Never commit to `main`. Never commit secrets, `.env`, raw data, `mlruns/`, or `.venv/`.
6. Python 3.11, type hints, docstrings, `ruff`-clean, pinned `requirements.txt`. Config in `configs/*.yaml`, not hard-coded.
7. Every module gets pytest tests (fixtures in `tests/fixtures/`, small and deterministic). Notebooks are exploration only.
8. The user is on Windows. Give commands that work in PowerShell, and use `pathlib`, not hard-coded `/` or `\` paths.

## Data and ML rules (non-negotiable)
- All timestamps are **UTC**. Canonical key: `site_id` + `interval_start_utc`. Also store `interval_end_utc`.
- Read power/energy metadata first. Integrate kW over sample duration to get kWh, or difference cumulative meters with reset
  detection. **Missing intervals are NOT zero production.**
- Respect the source's interval alignment flag (left/center/right) when building intervals. Never pair one site's weather
  with another site's power.
- **No random splits.** Reserve the latest 20% of valid time as a final test set (opened once, after selection). Within the earlier
  80% use 3 expanding-window folds. Fit imputers/scalers only on training folds.
- Lag/rolling features use strictly earlier observations. No centered windows, no backward fill from future rows.
- NASA POWER is an observational archive, not a forecast. For the next-hour forecast (F1), weather is allowed only if
  `available_at_utc <= issue_time_utc`. Default simulated weather lag is **14 days** (config value; also run 7 and 30).
- Two separate modes: **F1** forecast (core) and **E1** expected-output estimate (uses observed weather, delayed analysis only).
  Never mix their inputs or claims.
- Metrics: daylight MAE (solar elevation > 0, independent of target values) is primary; also all-hour MAE, RMSE,
  capacity-normalized MAE. No MAPE near zero output.
- Training and serving must use the **same** feature functions (`src/features/preprocessing.py`).
- Each MLflow run records: model type, params, DVC/data version (hash), metrics, training duration, Git commit,
  feature schema version, feature-repo commit, artifacts and plots.

## Repository layout (Member 1 owns the bold paths)
```
AGENTS.md  README.md  REPORT.md  requirements.txt  .env.example  .gitignore
configs/        data.yaml  evaluation.yaml  model_params.yaml  feature_schema.json
data/           raw/  processed/                (DVC-tracked by Member 2; never in Git)
src/data/       ingestion.py  validation.py  aggregation.py
src/features/   preprocessing.py
src/models/     baselines.py  tune.py  train.py  evaluate.py  bundle.py  deviation.py
feature_store/  feature_repo/{feature_store.yaml,entities.py,features.py,feature_views.py}  materialize.py
tests/          test_data.py  test_features.py  test_model.py  test_feature_store.py  fixtures/
docs/           HANDOFF.md  data_card.md  model_card.md  handoff_member2.md  handoff_member3.md
evidence/       (screenshots, run IDs, logs)
src/serving/ src/monitoring/ pipelines/ docker/ k8s/ monitoring/ infra/ .github/   (owned by Members 2 and 3; do not edit)
```

## Definition of done for any task
Tests pass, `ruff check .` is clean, commands to run are written in `docs/HANDOFF.md`, and a short summary of what was
done, what was not done, and what the user should check by eye (units, time alignment, leakage) is reported.
