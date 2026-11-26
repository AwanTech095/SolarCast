# SolarCast Production Deviation Detection & Alerting Engine

## 1. Overview & Purpose
The SolarCast Production Deviation Detector compares measured hourly production with stored F1 forecasts.
The current project demonstrates historical replay. `DeviationDetector.process()` also supports ordered
incremental batches; a deployed live monitor, durable state and delivery system are not implemented here.

**26 November 2026 correction:** Earlier code counted nonadjacent or duplicate rows as consecutive.
The corrected implementation requires exactly one hour between eligible target intervals. All duplicate
target times within a batch are suppressed, and a gap resets pending persistence. The historical empirical
numbers below have not been rerun with this fix and do not establish the corrected detector's performance.

Operational timing conventions:
- **Issue Time ($H$ UTC)**: Start of the target hourly interval $[H, H+1\text{h})$ UTC.
- **Production Telemetry Cutoff**: $H - 15\text{ minutes}$ UTC (latest usable 15-minute generation reading is $[H-30\text{m}, H-15\text{m})$).
- **Residual Calculation**: Evaluated once the target hour closes and measured meter data arrives:
$$\text{residual} = \text{actual\_kwh} - \text{forecast\_kwh}$$

- **Negative Residual ($\text{actual} < \text{forecast}$)**: Underproduction relative to expectation (e.g., equipment curtailment, unanticipated cloud cover, tracker/inverter trips, soiling).
- **Positive Residual ($\text{actual} > \text{forecast}$)**: Overproduction relative to expectation (e.g., clear sky when cloud was forecast, model underestimation).

## 2. Calibration Protocol & Operating Bins
Thresholds are calibrated strictly on out-of-sample predictions across the 3 expanding validation folds (13,914 VALID daylight hours between 2019-08-03 and 2022-10-17 UTC). In-sample training data and the locked 20% final test partition were **never** accessed for threshold calibration.

### Stratification Decision & Clear-Sky Index Binning Rule
Per specification, clear-sky index ($k_t$) cross-stratification was evaluated against the 200-sample minimum threshold rule:
- `clear` cloud bin: sample count $\ge 1,542$ across all $k_t$ bins.
- `mixed` cloud bin: sample count $\ge 488$ across all $k_t$ bins.
- `overcast` cloud bin: sample count $\ge 535$ across all $k_t$ bins.
- `no_weather` bin: `no_weather x csi_low` contains only **85 samples** and `no_weather x csi_high` contains only **154 samples** (both $< 200$).

**Decision**: Because sub-bin sample counts fall below 200 in the missing-weather regime, clear-sky index bins were merged into the 4 primary forecast cloud-cover bins to ensure statistical stability.

### Frozen Calibrated Thresholds (`configs/deviation.yaml`)
Bounds are set to the 1st percentile (`bound_low`) and 99th percentile (`bound_high`) of validation residuals on VALID daylight hours:

| Operating Cloud Bin | Definition | Validation Samples ($N$) | Lower Bound $P_1$ (kWh) | Upper Bound $P_{99}$ (kWh) | Mean (kWh) | Std Dev (kWh) |
|---|---|---|---|---|---|---|
| **clear** | $\text{tcdc\_pct} < 20.0\%$ | 9,843 | **-140.66** | **+127.94** | +3.27 | 41.92 |
| **mixed** | $20.0\% \le \text{tcdc\_pct} \le 80.0\%$ | 1,664 | **-231.18** | **+226.24** | +5.54 | 78.08 |
| **overcast** | $\text{tcdc\_pct} > 80.0\%$ | 1,867 | **-203.70** | **+216.30** | +4.58 | 74.75 |
| **no_weather** | $\text{weather\_missing} == 1$ or NaN | 540 | **-155.80** | **+191.86** | +6.83 | 53.42 |

*Physical Interpretation*: Clear sky displays the tightest distribution ($\sigma = 41.92\text{ kWh}$) because PV output closely follows deterministic solar geometry. Mixed sky exhibits the widest bounds ($\sigma = 78.08\text{ kWh}$, $[-231.18, 226.24]$) due to NWP cloud timing/phase errors over localized collector fields.

## 3. Alerting Rule & State Machine
- **2-Hour Consecutive Breach Rule**: An alert event triggers only after **2 consecutive hourly intervals** outside calibrated bounds in the **same direction** (`consecutive_hours >= 2`).
- **Alert Trigger Timing**: Because hourly production energy over $[H, H+1\text{h})$ is integrated only after the hour has ended (with telemetry available at $H+1\text{h} + 15\text{m}$), **an alert is raised when the second anomalous hour has closed** and its telemetry has been ingested.
- **Incremental emission**: Keep one `DeviationDetector` instance per site/model/calibration context.
  `process(frame)` returns a new event at the second adjacent same-direction breach. Further hours of the
  same episode do not emit again. Returned snapshots are not mutated as the episode continues.
- **Batch compatibility**: `detect_deviations(frame)` returns retrospective full-episode summaries;
  each independent call starts fresh. `detector.summaries()` exposes summaries across incremental calls.
  Do not use independent batch-wrapper calls as a stateful live monitor.
- **Detection timing**: `confirmation_issue_time_utc` is the second target's issue time;
  `target_interval_start_utc` and `target_interval_end_utc` identify that confirming interval.
  Supply each row's real `actual_available_at_utc` to establish `triggered_at_utc` (the later availability
  of the first two labels). `detection_delay_hours` is that trigger minus the first anomalous target start.
  For 16:00 and 17:00 breaches with actuals available at 17:15 and 18:15, confirmation is at 18:15 and
  delay is 2.25 hours. The 1-hour difference between target starts is not operational detection latency.
  Without both availability timestamps, trigger time and delay are null; no SLA-based arrival is invented.
  `created_at_utc` is event-record creation time, distinct from historical label availability.
- **Window semantics**: Existing `window_start_utc`/`window_end_utc` remain first/last anomalous target
  interval STARTS for compatibility. The latter is neither an interval end nor a label arrival timestamp.
- **Identity**: `event_id` is a deterministic hash of site, model/version, episode start/direction and
  frozen bounds. It is independent of batch position and record creation time. Delivery consumers deduplicate
  on that ID. An intentionally truncated replay may lack the true episode start and cannot reproduce its ID.
- **Suppression Conditions**:
  - Nighttime hours ($\text{solar\_elevation\_deg} \le 0^\circ$).
  - Missing target measurements (`target_kwh` is NaN).
  - Missing model forecast.
  - Nonfinite numeric measurements, unknown daylight eligibility, duplicated or off-hour target timestamps.
  - Quality flags other than `VALID` (e.g., `MISSING_DATA`, `INCOMPLETE`, `SUSPECT_ZERO` during standard operation).
- **Reset Conditions**:
  - A suppressed hour (night or invalid).
  - A residual returning within normal bounds ($P_1 \le \text{res} \le P_{99}$).
  - A direction flip (e.g., negative breach immediately followed by positive breach).
  - A time gap of any length other than exactly one hour.
- **Batch ordering**: Inputs are normalized from explicit timezone-aware timestamps to UTC and sorted.
  Across `process()` calls, all new timestamps must be later than the last processed timestamp; overlapping
  or late batches reject before changing state. The service must buffer late labels and reconcile duplicates
  before committing a batch. It must persist/recover state or replay complete episodes after restart;
  this in-memory class does not provide durable checkpoints, corrections or exactly-once delivery.
- **Standardized Schema**: [configs/alert_event_schema.json](../configs/alert_event_schema.json) includes
  optional timing fields, including nullable availability/latency, alongside the existing alert fields.

```python
from src.models.deviation import DeviationDetector

detector = DeviationDetector()  # frozen configs/deviation.yaml, default site/model version
# first_batch and next_batch are already joined stored-forecast/actual rows in UTC.
new_events = detector.process(first_batch)
new_events += detector.process(next_batch)  # same instance, strictly later target hours
```

Core regression check (no calibration, test-period loading or benchmark invocation):

```powershell
.\tmp\review_env\Scripts\python -m pytest tests/test_deviation.py tests/test_deviation_continuity.py -q
```

## 4. Empirical Evaluation Results

Historical Member 1 results, preserved unchanged. These used the earlier row-based continuity logic and
target-start delay convention. Corrected rates/latencies are unmeasured; do not reopen the final test to
refresh them in automated runs. Use the synthetic regression tests to verify the code fix.

The evaluation protocol explicitly separates synthetic fault injections from natural historical cases:

### A. Synthetic Fault Injections (Final Test Partition Copy, Seed 42)
> [!NOTE]
> All injected-fault numbers and benchmark metrics in this subsection are **synthetic**: artificially simulated fault patterns (zero output, bad values, power drops) evaluated on a working copy of the held-out final test partition to measure detector sensitivity and latency. They do not represent physical hardware breakdowns on the operational plant.

20 non-overlapping episodes per fault category, spaced $\ge 12$ hours apart, injected on VALID daylight stretches with weather:

| Injection Type | Duration | Episodes Injected | Episodes Detected | Detection Rate (%) | Median Delay (h) | Max Delay (h) | Notes |
|---|---|---|---|---|---|---|---|
| **Zero Output** | 3 hours | 20 | 19 | **95.0%** | **1.0** | 1.0 | 1 missed episode occurred at low sun where forecast was below 140.66 kWh |
| **Bad Value** (-50 / 1250 kWh) | 2 hours | 20 | 18 | **90.0%** | **1.0** | 1.0 | High-sun breaches trigger immediately on hour 2 |
| **30% Power Drop** | 4 hours | 20 | 12 | **60.0%** | **1.5** | 3.0 | Conservative 1% bounds prevent false alarms on cloudy/low-power days |
| **Isolated 1-Hour Drop** | 1 hour | 20 | 0 | **0.0%** | N/A | N/A | **100% compliance** with persistence rule (0 false alerts) |

### B. Baseline False-Alert Rate (Unmodified Final Test Period)
- Total Valid Daylight Hours Evaluated: 4,601 hours.
- Total Unmodified Alert Events Triggered: 2 events.
- **False-Alert Rate**: **0.43 alert events per 1,000 valid daylight hours**.
- Details of the two alerts:
  1. `DEV-202304012000-001`: 2 consecutive hours of underproduction during a missing-weather interval (`mean_residual = -302.54 kWh`).
  2. `DEV-202305022000-002`: 2 consecutive hours of overproduction during overcast forecast (`mean_residual = +263.46 kWh`, localized clearing).

### C. Natural Real-Data Anomalies (`SUSPECT_ZERO` Evaluation)
For retrospective benchmarking, all 286 historical `SUSPECT_ZERO` hours across the 5.3-year evaluation window were unsuppressed and evaluated against out-of-sample model predictions:
- Total Real `SUSPECT_ZERO` Contiguous Episodes: 43 episodes.
- Episodes Detected by Deviation Detector: **6 episodes (13.95%)**.
- Median Detection Delay: **3.0 hours** (Max Delay: 4.0 hours).
- **Physical Reason for Non-Detection of Short Episodes**:
  - Many natural `SUSPECT_ZERO` episodes are 2 hours long occurring near sunrise/sunset ($5^\circ$ to $10^\circ$ elevation) where expected generation is only 15–35 kWh.
  - A drop from 25 kWh to 0 kWh produces a residual of $-25\text{ kWh}$, which does **not** breach the conservative $-140.66\text{ kWh}$ threshold.
  - In contrast, midday multi-hour zero episodes (e.g., 10-hour outage on 2019-02-22 with 500+ kWh expected generation) produce residuals of $-500\text{ kWh}$ and trigger alerts with high confidence.

## 5. Limitations & Operational Caveats
1. **Single-Site Parameterization**: Thresholds are calibrated specifically for PV System 2107 (893 kW DC ground mount, Arbuckle, CA). Other sites require separate calibration against their own capacity and residual distributions.
2. **Historical Validation Window**: Calibration was performed on 2019–2022 validation data. Long-term module degradation, inverter replacements, or seasonal soiling patterns may cause threshold drift.
3. **Alert is a Deviation Flag, Not a Diagnosis**: The detector signals that observed output statistically diverges from the advance forecast; it **cannot** determine the mechanical root cause (e.g., whether a drop is caused by an inverter trip, snow cover, curtailment, or unpredicted cloud cover). Root causes remain *unknown* without field telemetry.

## 6. How to Run
```powershell
# 1. Run deviation detector calibration, evaluation, and reporting
.\.venv\Scripts\python -m src.models.deviation --log-mlflow

# 2. Run deviation detector unit test suite
.\.venv\Scripts\pytest tests/test_deviation.py -v

# 3. Run full project test suite
.\.venv\Scripts\pytest -m "not network"

# 4. Check code style and formatting
.\.venv\Scripts\ruff check .
```
