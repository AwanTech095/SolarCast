# Member 1 integration review - 24 November 2026

## Decision

**Update, 26 November 2026:** R1's missing-file transfer is now resolved locally. Received
`SolarCast_packet.zip`, verified all 31 payload hashes/sizes, installed the data/model files and confirmed the
bundle's recorded input hashes. Existing tests now report **64 passed, 2 skipped** in the Python 3.12 review
environment. See `reports/packet_receipt_20261126.json` and the handoff's step-1 entry. This does not resolve
the code findings below or establish shared DVC/MLflow, separate Feast or clean Python 3.11 verification.
The original 24 November evidence below is preserved as a dated review, not the current artifact inventory.

**Conditional acceptance of the foundation; not yet ready for automated promotion or serving.**
The project is moving in the intended direction. Member 1 has delivered substantial modular code,
documentation and reported experiments. Keep this work; do not restart the project or replace the dataset
solely because of these findings. Resolve the handoff, gate and target-time issues before treating the
pipeline as deployable. Member 2 can begin artifact transfer/versioning and fixture-based CI preparation;
full training/deployment automation should follow the acceptance checks below.

This review does not change the approved project scope, approve new data sources or implement fixes.

## Checkout and review coverage

- Fetched GitHub updates from `https://github.com/AwanTech095/SolarCast`.
- Fast-forwarded the previous checkout to `origin/main` at `dbd5f40`.
- Found six additional handoff commits on `origin/m1/handoff-docs`. That branch contains main, but its
  handoff additions are not merged into main at review time.
- Reviewed that latest branch at **`d8c907f`** on local branch **`m1/member1-review`**.
- Read the Member 1 status/decision log, Member 2/3 handoffs, data/weather/model reports, configs,
  ingestion/aggregation, shared features, HRRR extraction, training/tuning/evaluation, model bundle,
  Feast provider and deviation code, plus their tests.
- Did not merge branches into GitHub, push changes, retrain on the real data or reopen the final test set.

## What has been delivered

| Area | Present in the reviewed code | Acceptance qualification |
| --- | --- | --- |
| Production ingestion and hourly targets | S3/NASA ingestion, manifest generation, Pandera schemas, quarantine, UTC aggregation and explicit missingness | Full source datasets are not in this checkout; their reported counts/hashes have not been independently reproduced. |
| Forecast weather | HRRR TMP/cloud-cover extraction, older-run selection, fallback and missingness handling | Availability is assigned as run initialization plus two hours; it is an assumed latency policy, not per-file historical publication proof. |
| Features | Shared builder, 23 production/calendar/clear-sky features and 29-feature weather-enhanced set | Includes a derived cloud-adjusted irradiance proxy rather than directly ingested forecast DSWRF. Target-time retrieval and P95 calibration need attention. |
| Models and experiments | Persistence baselines, Ridge, histogram gradient boosting, chronological CV, model bundle, MLflow logging and saved reports | Actual model artifact and shared tracking access are absent. Reported scores have not been rerun here. |
| Feast | Definitions, Parquet source, SQLite materialization, historical/online provider and tests | Source data/registry absent; separate Feast suite not executed. Retrieval currently allows older target-specific rows. |
| Deviations | Frozen residual-bound config, synthetic injection benchmark and alert schema | Gap handling, calibration/model identity and automated CLI behavior need corrections. |
| Team integration | Detailed Member 2/3 handoffs and request/response samples | Useful contracts, but some advertised commands and clean-clone assumptions are not currently executable. |

### Alignment with the assigned idea

The implemented F1 design uses advance forecast temperature and cloud cover, time/solar geometry,
past production, and **derived irradiance**: pvlib clear-sky GHI adjusted by forecast cloud cover.
That is a reasonable prototype approach to weather-aware solar forecasting, provided the team explicitly
accepts and documents this interpretation of the irradiance input. It must not be described as direct HRRR
forecast DSWRF or measured irradiance. The feasibility report says direct DSWRF was obtainable through
GRIB2 but was not chosen for the full replay because of extraction overhead; this review did not independently
repeat those download experiments.

Historical replay remains appropriate. There is no live plant feed. NASA POWER remains supporting
retrospective data and must not be presented as target-hour observations available in advance.

### Reported performance, not independent reproduction

`evaluation.json` reports weather-enhanced HGB validation daylight MAE **31.4262 kWh** versus
**42.0891 kWh** for the selected persistence baseline, a **25.33%** improvement. Its final-test
weather-present daylight slice reports **32.6863 kWh** versus **43.4825 kWh**, a **24.83%** improvement.
The model report attributes a smaller **2.52% validation** and **3.84% final-test** improvement to adding
weather relative to HGB without weather. These are different comparisons: most improvement over persistence
must not be attributed solely to weather. Capacity-normalized MAE is not a classification accuracy percentage.

## Verification performed locally

The host exposes Python **3.12.5**, not the specified Python 3.11. Installed the repository's pinned core
requirements into the temporary `tmp/review_env` environment with system-site packages visible. Pip noted
conflicts with unrelated inherited host packages. No project dependency files or global packages were changed.
This verifies selected behavior, not a clean Python 3.11 container build or the separate Feast environment.

| Check | Observed result |
| --- | --- |
| `pytest -m "not network" -q --tb=short` | **48 passed, 1 failed, 14 errors, 3 skipped**. The failure/errors report absent datasets or model bundle. The complete suite is therefore not green on this checkout. |
| Ruff with project configuration, excluding local untracked `tools/` | **All checks passed.** The repository itself excludes `scripts/`; this is not a claim that those scripts were linted. |
| Independent synthetic edge-case probes | Confirmed NaN gate acceptance, nonconsecutive alert, old-target online feature acceptance, nonnumeric-power quarantine gap and an invalid documented CLI argument. |
| Actual model/dataset reproduction | Not possible with the files currently supplied in this checkout. |
| Final holdout evaluation | Not run or reopened. |

The earlier unrestricted `ruff check .` also found violations in the local untracked PDF-authoring helper
`tools/build_solarcast_plan.py`, which predates Member 1's work. Those are not attributed to Member 1.

Local reproduction commands (PowerShell):

```powershell
# Temporary review environment; do not treat Python 3.12 as the required production environment.
.\tmp\review_env\Scripts\python -m pytest -m "not network" -q --tb=short
.\tmp\review_env\Scripts\python -m ruff check . --exclude tools --output-format concise
.\tmp\review_env\Scripts\python tmp/member1_review_probes.py
```

The temporary probe script and logs are under `tmp/`, which is Git-ignored. Findings below record the inputs,
outcomes and source locations so the review remains understandable without those temporary files.

## Required corrections and handoff actions

### R1 - High: deliver the artifact packet before claiming clean-clone reproducibility

**Evidence:** `tests/test_contract_samples.py:28` asserts a model bundle exists; it does not. Feature tests
load `data/processed/hourly.parquet`, also absent. All seven artifact/config paths checked in the independent
probe were absent, including the model, processed weather and `.dvc/config`.

The Member 2 handoff's `dvc pull` step assumes storage that Member 2 has not configured yet. Keeping large
data/model artifacts out of Git is correct; omitting their transfer/bootstrap location prevents reproduction.

**Action - Member 1 -> Member 2:** supply an access-controlled artifact location/archive containing raw meter
and metadata, the processed hourly/quarter-hourly/weather files, Feast feature table, validation residuals,
the model bundle and exact checksums. Include the model's checksum, MLflow run ID/registered version and
tracking URI/access instructions (credentials through a separate secret channel). Member 2 then configures
DVC and the remote and verifies hashes. Do not perform multi-year HRRR extraction in every CI run.

**Acceptance:** from a new checkout, retrieve and verify the packet; run core, artifact-dependent and separate
Feast checks successfully in the intended Python 3.11 environments. Unit tests should use small deterministic
fixtures; mark real-artifact tests as integration tests with an explicit artifact-fetch prerequisite.

### R2 - High: make promotion fail closed and evaluate the actual candidate

**Reproduced:** `evaluate_candidate_gate()` returned `eligible=true` for candidate MAE and normalized MAE
both equal to `NaN`, with finite predictions. Comparisons such as `NaN > threshold` are false.
Source: `src/models/evaluate.py:126` and its comparisons around lines 151/159.

**Source inspection:** `src/models/train.py` writes the bundle at line 516 regardless of gate eligibility,
logs/registers a selected candidate regardless of acceptance, and returns CLI status 0 at line 616.
It uses a shared output path, so a rejected run can overwrite the local bundle. Logging rejected candidates
is useful; treating process success or that shared path as production approval is unsafe.

The advertised `evaluate --dry-run` refits models with hard-coded parameters (`src/models/evaluate.py`,
around lines 464-480), rather than loading and evaluating the just-produced bundle. Thus its completion
cannot validate a future candidate with different parameters. The training flow also writes
`final_test_opened=false` into fresh evaluation JSON while the separate historical audit remains true;
immutable experiment identities must distinguish these records.

**Action - Members 1 and 2:** reject nonfinite/empty metrics and predictions; validate the baseline denominator;
use run-specific candidate directories and a release manifest binding model hash, code/data/features and
evaluation; branch explicitly on `eligible` before promotion; evaluate the exact candidate being packaged.
Do not assume exit code 0 means a quality gate passed, or publish a rejected artifact as the serving champion.

**Acceptance:** failing metrics, NaN metrics and a changed/bad bundle cannot promote or overwrite the current
approved release. A rejected run may be recorded for audit, with a distinct candidate identity.

### R3 - High: target-specific Feast rows must match the requested prediction hour

**Reproduced with a mocked store response:** querying issue time `2023-06-15T16:00Z` returned an unchanged
feature row whose event time was `15:00Z`; the provider accepted its one-hour age.
Source: `src/features/feast_provider.py:100-204` (two-hour default staleness allowance).

This row contains target-hour solar elevation, calendar features, production lags and target-hour forecast
weather. It is not a generic observation that can safely be reused for the following hour. Merely returning a
staleness warning does not fix the prediction target. Offline retrieval also performs an as-of join without
checking the returned source feature hour; with daylight-only rows and a seven-day TTL it can substitute
the previous available hour/day for an absent request.

**Action - Members 1 and 3:** either require an exact feature issue/target timestamp or rebuild target-dependent
features for the requested hour from freshness-checked source observations and forecast vintages. Include
the original source row timestamp in offline retrieval results and reject inappropriate as-of substitutions.
Handle night requests explicitly. Define a process/service boundary for the isolated Feast and model environments.

**Acceptance:** exact-hour parity; missing target-hour rejection; no prior daylight feature row served as a night
forecast; earlier source observations only reused after target-specific transformations are recomputed.

### R4 - High: enforce temporal continuity in deviation alerts

**Reproduced:** two VALID daylight rows at `16:00Z` and `19:00Z`, each with actual 0 kWh versus forecast
500 kWh and lower bound -100 kWh, produced one alert reporting `consecutive_hours=2`.
There are two unobserved hours between those rows.

Source: `src/models/deviation.py:291-439`, particularly the same-direction increment around line 423.
The implementation sorts rows but does not require a one-hour difference before incrementing its counter.
This matters because the test-period frame is already filtered to VALID rows and can omit missing hours.

**Action - Member 1:** reset on gaps/duplicates as well as suppressed rows; emit an alert when the second
eligible consecutive label arrives, with issue/target/actual-availability times and an idempotent episode ID.
Define state across monitoring batches; the current API is a batch episode detector, not a deployed live alert service.

**Acceptance:** adjacent hourly breaches alert; separated/duplicate records do not satisfy persistence; missing
hours reset the episode. Keep detection timing distinct from the target interval's start time.

### R5 - High: repair runbook flags and protect the final test in automation

**Reproduced:** the handoff's deviation command includes `--reports-dir`, but the CLI accepts only
`--calibrate`, `--evaluate`, `--log-mlflow` and `--seed`. It exited 2 with
`unrecognized arguments: --reports-dir tmp/review_unused`. The advertised `--eval-config`, `--features`
and `--bundle` flags are also absent. See `docs/handoff_member2.md:161-170` and
`src/models/deviation.py:812-837`.

Even after removing that flag, the default deviation command evaluates the final test period through
`get_test_period_dataset()` without the safeguard in `evaluate.py`. Scheduling that command as written
would reuse the final test on each retraining run. Calibration also uses frozen current-champion parameters,
so recalibrating a different promoted model needs an explicit model/config contract.

**Action - Members 1 and 2:** align tested CLI flags with the runbook; separate calibration, rolling operational
evaluation and one-time benchmark commands. Use bounded past-only calibration/validation windows and the
candidate/champion's actual identity. Never schedule the default final-test benchmark as the retraining step.

**Acceptance:** copy/pasted runbook commands parse; orchestration does not access the final holdout or overwrite
its audit; a new model receives matching deviation calibration rather than the old hard-coded specification.

### R6 - Medium: close the raw-value validation gap

**Reproduced:** a raw power string `bad_sensor_value` was retained in the clean frame as NaN, with zero
quarantined records. Source: `src/data/validation.py:145-199`; the initial missingness mask checks the original
string rather than the numeric conversion, and the declared raw schema is not called by this function.

**Action - Member 1:** coerce once, classify invalid/nonfinite values from the coerced series, validate required
columns and timestamp parseability, then quarantine/reject with reasons. Add a regression case for invalid text.

### R7 - Medium: resolve the P95/holdout provenance limitation

`docs/data_feasibility.md:85` says P95 was computed from all seven years. `docs/model_card.md` explicitly
acknowledges it may include test-period data. The constant feeds `configs/features.yaml` and the feature builder.
The feasibility week selection also inspected late-2022/early-2023 production, which overlaps the reported
final test window, before the later one-time evaluation. An audit of evaluation invocations alone does not
prove those targets were never inspected during earlier data research.

The simple linear scaling factor cancels in some unclipped baseline ratios, but it also appears in nonlinear
clipped/thresholded features. The review does not establish the magnitude of any score effect; do not claim
that this necessarily explains or eliminates the reported advantage.

**Action - Member 1:** establish the constant's origin, fit learned calibration only within training folds, persist
it in the feature/model contract and disclose previous holdout inspection. Keep the existing final results as
recorded; do not silently tune and repeatedly reopen the same final set. Agree on future evaluation governance
before any new experiments that use those results.

### R8 - Medium: distinguish assumptions, model identities and retrospective evidence

- The feasibility report labels the two-hour weather latency **Assumed**; the status board calls it verified.
  Code constructs `weather_available_at_utc = init + lag`. Describe this as latency-policy replay unless original
  release evidence exists. Archive existence now does not establish original availability.
- `run_natural_suspect_zero_evaluation()` says it uses out-of-sample fold forecasts, but actually loads one bundle
  and predicts the full historical window (`src/models/deviation.py:698-755`). Earlier episodes therefore use
  a model trained on later dates. Either use chronological out-of-fold models or relabel that result as retrospective.
- Several handoff references use feature schema `1.1.0` as a model version; that will not uniquely identify retrained
  releases. Bind alerts, serving and rollback to a registry version/model hash as well as the schema version.
- Training rebuilds features directly rather than consuming Feast historical retrieval. The end-to-end workflow
  still needs to demonstrate how the feature store participates in the actual training/serving contract.
- README still describes HRRR as pending and older plans still use the previous approach. After the team reviews
  the proxy/latency choices, synchronize current scope/status without retroactively changing audit history.

## Work order and ownership

| Priority | Owner | Next task |
| --- | --- | --- |
| 1 | Member 1 -> Member 2 | Transfer immutable data/model packet and tracking references; verify checksums. |
| 2 | Member 1, reviewed by Member 2 | Fix R2/R4/R5 gate and alert/runbook issues, with focused regression tests. |
| 3 | Member 1 + Member 3 | Fix target-time retrieval and agree on the Feast/model process boundary. |
| 4 | Member 1 + team | Resolve P95 provenance, disclose latency assumptions and sign off on derived-irradiance wording. |
| 5 | Member 2 | Configure DVC/DagsHub storage and fixture versus artifact/Feast test jobs; reproduce the supplied release. |
| 6 | Member 2 | Build Airflow ingestion/validation -> Kubeflow feature/train/evaluate graph with explicit rejection branches. |
| 7 | Members 2 + 3 | Package approved immutable releases, add controlled deployment, monitoring, promotion and rollback. |

Full Member 2 implementation was not started during this review. The immediate useful starting point is the
artifact/versioning and CI foundation, not automatic deployment of the current shared bundle path.
