# MLflow Lineage and Cryptographic Provenance Resolution

This document details the lineage tracking, Model Registry configuration, and cryptographic provenance
audit for the **SolarCast** machine learning pipeline (Mode F1).

---

## 1. Overview & Requirements

The SolarCast project requirements dictate that any selected model run must resolve to:
1. **Exact code commit** and working tree dirty/clean status.
2. **Cryptographic dataset hashes** (SHA-256 of processed training datasets, forecast features, and data manifest).
3. **Feature schema version and hash** (`configs/feature_schema.json`).
4. **Configuration hashes** (`configs/evaluation.yaml`, `configs/model_params.yaml`).
5. **Model artifact / bundle SHA-256** matching the binary deployed to serving (`models/solarcast_f1_bundle.joblib`).

To satisfy this without introducing third-party dependency bloat or breaking isolated virtual environments
(`.venv` and `.venv-feast`), a standard-library-only helper module was implemented at `src/models/lineage.py`.

---

## 2. Lineage Architecture (`src/models/lineage.py`)

The helper function `log_lineage(run=None)` extracts immutable environment and file metadata using Python's
`hashlib`, `subprocess`, `json`, and `sys` modules:

| Tag Key | Source / Description | Verified in Champion Run? |
|---|---|---|
| `git_commit` | `git rev-parse HEAD` | Yes (`4eb8276...`) |
| `git_dirty` | `git status --porcelain` | Yes (`true`, uncommitted tree at run time) |
| `python_version` | `sys.version.split()[0]` | Yes (`3.11.9`) |
| `mlflow_version` | `mlflow.__version__` | Yes (`2.17.2`) |
| `data_hash_hourly` | SHA-256 of `data/processed/hourly.parquet` | Yes (`f9157a5dd5c43788...`) |
| `data_hash_hrrr` | SHA-256 of `data/processed/hrrr_forecast_features.parquet` | Yes (`fbc02d8bbbbdb9de...`) |
| `data_hash_qh` | SHA-256 of `data/processed/quarter_hourly.parquet` | Yes (`7c488a111f2eb7b5...`) |
| `data_hash_feast` | SHA-256 of `data/processed/feast/features_set_b.parquet` | Added in Step 1 for future runs |
| `data_manifest_hash` | SHA-256 of `configs/data_manifest.json` | Added in Step 1 for future runs |
| `feature_schema_version`| `schema_version` from `configs/feature_schema.json` | Yes (`1.1.0`) |
| `feature_schema_hash` | SHA-256 of `configs/feature_schema.json` | Added in Step 1 for future runs |
| `config_hash_evaluation`| SHA-256 of `configs/evaluation.yaml` | Added in Step 1 for future runs |
| `config_hash_model_params`| SHA-256 of `configs/model_params.yaml` | Added in Step 1 for future runs |
| `bundle_sha256` | SHA-256 of `models/solarcast_f1_bundle.joblib` | Yes (`f72de152727fd406...`) |

### Pipeline Integration
`log_lineage(run)` is called inside the active MLflow context in:
- `src/models/train.py` (candidate runs and champion bundle export)
- `src/models/evaluate.py` (evaluation and final test audit)
- `src/models/deviation.py` (residual deviation detector runs)
- `scripts/replay_monitoring.py` (virtual-clock monitoring simulation)
- `feature_store/materialize.py` (Feast online store materialization in `.venv-feast`)

---

## 3. Audit of Existing Historical Runs

Historical runs logged prior to Step 1 exhibited gaps where certain SHA-256 hashes or git commits were
not logged at execution. In accordance with course rules, no numbers were invented or back-filled without
direct verification. All retroactively added tags carry the prefix `lineage_added_later.` and the timestamp
tag `lineage_added_later_at`.

### Run Audit Table

| Pipeline Stage | MLflow Run ID | Git Commit | Dirty Flag | Data Hashes | Schema Hash | Bundle SHA | Notes |
|---|---|---|---|---|---|---|---|
| **Champion Training** | `f7fef3f7bdf241c5a53dc50d8e4713f9` | `4eb8276` | `true` | Verified (hourly, hrrr, qh) | unverified | Verified (`f72de1...`) | Working tree uncommitted at run time; committed in `0afc5c6`. |
| **Final Test Evaluation**| `620faa341a8a4c14824f31e40b157fd6` | `763b391` | unverified | unverified | unverified | unverified | One-time test sealed; data hashes not logged in tags. |
| **Deviation Detector** | `cce94db3448e405ab0fee01415e82da1` | `95e0fc0`* | unverified | unverified | unverified | unverified | *Captured in system tags, not user tags. |
| **Monitoring Replay** | `16ffd7429be642928941e95f932fe8b3` | unverified | unverified | unverified | unverified | unverified | Launched via `-c` CLI; no git commit captured. |
| **Feast Materialization**| `56f349ec19f64a62917396b79a8e9fa9` | `5cdb428` | unverified | unverified | unverified | unverified | `feast_repo_git_commit` logged; data hashes not logged. |

### Champion Run Working-Tree State
The champion run recorded git commit `4eb82768058447721fca93a5a41c29333c6acafd` (`fix(features): sanitize out-of-range weather values`).
Git log analysis demonstrates that the benchmarking and bundling code in `src/models/train.py`, `src/models/bundle.py`,
and `src/models/tune.py` was in progress in the working tree when the run was executed at `2026-11-23 20:49:04 UTC`.
Those changes were committed shortly afterwards in `0afc5c6ca100623e325080ed085ecce5238563d2` (`feat(models): benchmark Set A vs B...`).
Thus, `lineage_added_later.git_dirty = "true"` and `lineage_added_later.git_working_tree` accurately documents this provenance.

### Champion Bundle Cryptographic Match
- Disk bundle (`models/solarcast_f1_bundle.joblib`): `f72de152727fd406ac35a08cfa83289b6a61b031e0b3f4eb52008288e3980619`
- Champion artifact (`mlruns/.../artifacts/bundle/solarcast_f1_bundle.joblib`): `f72de152727fd406ac35a08cfa83289b6a61b031e0b3f4eb52008288e3980619`
- Match confirmed: `[OK]`. Tagged `lineage_added_later.bundle_sha256 = f72de152727fd406...`.

---

## 4. MLflow Model Registry Configuration

MLflow 2.17.2 local file-store (`file:./mlruns`) fully supports aliases, model version tags, and version descriptions.
The registry was configured using `scripts/setup_model_registry.py` and exported to `reports/model_registry_status.json`:

```
Registered Model: solarcast-f1
├── Version 1 (Run: 88c57491...)
│   ├── Status Tag : superseded
│   └── Description: HistGradientBoosting Set A (23 features). CV MAE: 32.24 kWh. +57.36% vs ClearSkyPersistence.
├── Version 2 (Run: 7b64f8c9...)
│   ├── Status Tag : superseded
│   └── Description: HistGradientBoosting Set A (23 features). CV MAE: 32.24 kWh. +23.56% vs Clear-Sky Scaled 15m Persistence.
├── Version 3 (Run: 555ba709...)
│   ├── Status Tag : superseded
│   └── Description: HistGradientBoosting Set B (29 features, HRRR weather). CV MAE: 31.46 kWh. +25.25% vs baseline.
└── Version 4 (Run: f7fef3f7...)
    ├── Alias      : champion
    ├── Status Tag : champion
    ├── Tags       : mode=F1, feature_set=B, schema_version=1.1.0, gate_eligible=true, final_test_opened=true,
    │                bundle_sha256=f72de1..., source_run_id=f7fef3f7bdf241c5a53dc50d8e4713f9
    └── Description: HistGradientBoosting Set B champion with sanitized weather bounds (29 features).
                     CV MAE: 31.43 kWh (+25.33% vs baseline, +2.52% vs Set A). Final test MAE: 32.69 kWh.
```

---

## 5. Resolving Runs with `scripts/resolve_run.py`

The resolver script `scripts/resolve_run.py` validates any run ID or the `"champion"` alias against the local filesystem.

### Usage
```powershell
# Resolve the champion model via the Model Registry alias
python scripts/resolve_run.py champion

# Resolve an explicit run ID (e.g. final test evaluation)
python scripts/resolve_run.py 620faa341a8a4c14824f31e40b157fd6
```

### Resolver Output Example (Champion)
```
================================================================================
 SOLARCAST RUN LINEAGE RESOLVER: f7fef3f7bdf241c5a53dc50d8e4713f9
================================================================================
Run Name       : HistGradientBoosting_Set_B_15mDelay
Experiment ID  : 754448501154260354
Status         : FINISHED
Registry Model : solarcast-f1 (version 4)
Registry Alias : champion

--- 1. CODE PROVENANCE & GIT REPOSITORY STATE ---
  Git Commit SHA       : 4eb82768058447721fca93a5a41c29333c6acafd
  Working Tree Dirty   : true
  Provenance Note      : uncommitted_working_tree_at_execution; changes subsequent to 4eb8276 committed in 0afc5c6

--- 2. DATASET CRYPTOGRAPHIC INTEGRITY ---
  [OK]             Hourly Processed (hourly.parquet)
                   Logged : f9157a5dd5c43788...
                   Disk   : f9157a5dd5c43788...
  [OK]             HRRR Weather Features (hrrr_forecast_features.parquet)
                   Logged : fbc02d8bbbbdb9de...
                   Disk   : fbc02d8bbbbdb9de...
  [OK]             Quarter-Hourly Power (quarter_hourly.parquet)
                   Logged : 7c488a111f2eb7b5...
                   Disk   : 7c488a111f2eb7b5...
  [NOT RECORDED]   Feast Feature View Store (features_set_b.parquet)
                   Logged : unverified / not recorded
                   Disk   : 07fbb11feca94c4b...
  [NOT RECORDED]   Data Pipeline Manifest (configs/data_manifest.json)
                   Logged : unverified / not recorded
                   Disk   : ba950b3bfed32846...

--- 3. FEATURE SCHEMA & CONFIGURATION INTEGRITY ---
  Feature Schema Version : 1.1.0
  [NOT RECORDED]   configs/feature_schema.json
                   Logged : unverified / not recorded
                   Disk   : 8a5c56e4d702cf00...

--- 4. MODEL BUNDLE CRYPTOGRAPHIC PROVENANCE ---
  [OK]             models/solarcast_f1_bundle.joblib
                   Logged : f72de152727fd406...
                   Disk   : f72de152727fd406...
```

---

## 6. How to Compare the 12 Hyperparameter Tuning Runs

In experiment `754448501154260354` (`solarcast-model-training`), 12 hyperparameter tuning runs were evaluated across
the 3 expanding validation folds:
- **Ridge Regression (Set A and Set B):** Grid search across regularization parameter `alpha` in `[0.1, 1.0, 10.0, 100.0]`.
- **HistGradientBoosting (Set A and Set B):** Grid search tuning combinations of `learning_rate` (`[0.05, 0.10]`),
  `max_iter` (`[100, 200]`), and `max_leaf_nodes` (`[31, 63]`).

### Comparison Methods:
1. **Via Python Client:**
   ```python
   from mlflow.tracking import MlflowClient
   client = MlflowClient(tracking_uri="file:./mlruns")
   runs = client.search_runs(
       experiment_ids=["754448501154260354"],
       order_by=["metrics.daylight_mae_kwh ASC"]
   )
   for r in runs:
       print(f"{r.info.run_name:<35} | MAE: {r.data.metrics.get('daylight_mae_kwh'):.2f} kWh | Model: {r.data.params.get('model_type')}")
   ```
2. **Via MLflow UI:**
   ```powershell
   mlflow ui --backend-store-uri file:./mlruns --port 5000
   ```
   Navigate to experiment `solarcast-model-training`, select runs, and use the **Compare** view to inspect parallel
   coordinates, parameter correlation scatter plots, and metric bar charts.

---

## 7. Migration to Shared MLflow Server (Member 2 Runbook)

When Member 2 deploys the shared team MLflow Tracking Server (e.g. SQLite/PostgreSQL backend with artifact storage):

1. **Set Environment Variable:**
   ```powershell
   $env:MLFLOW_TRACKING_URI = "http://<shared-server-host>:5000"
   ```
2. **Re-log Training Runs:**
   Execute standard training pipeline:
   ```powershell
   python -m src.models.train
   ```
   This automatically invokes `log_lineage(run)` to capture all live SHA-256 hashes, git commit, and dirty state.
3. **Register Champion Alias and Tags:**
   Execute:
   ```powershell
   python scripts/setup_model_registry.py
   ```
4. **DO NOT RE-RUN FINAL TEST:**
   > [!IMPORTANT]
   > The final test partition (`reports/final_test_audit.json`) was opened exactly once on 2026-11-24 under strict
   > anti-leakage course safeguards. **Never re-run `python -m src.models.evaluate --final-test` with `--force`.**
   > To record the final test metrics on the shared server, migrate the logged artifact `reports/final_test_audit.json`
   > or register the run record without re-opening the test partition.
