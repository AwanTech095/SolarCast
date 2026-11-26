"""SolarCast Model Deviation Detection and Alerting Engine (Task M1.6 / Step 1 to Step 4).

Monitors forecast residuals (actual_kwh - forecast_kwh) produced by the champion model
(solarcast-f1 bundle) to flag sustained production anomalies or model deviations.
Strictly calibrated on out-of-sample validation folds with frozen thresholds.
"""

from __future__ import annotations

import copy
import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

DEFAULT_EVAL_CONFIG = Path("configs/evaluation.yaml")
DEFAULT_DEVIATION_CONFIG = Path("configs/deviation.yaml")
DEFAULT_ALERT_SCHEMA = Path("configs/alert_event_schema.json")
DEFAULT_HOURLY_PATH = Path("data/processed/hourly.parquet")
DEFAULT_FEATURES_PATH = Path("data/processed/feast/features_set_b.parquet")
DEFAULT_BUNDLE_PATH = Path("models/solarcast_f1_bundle.joblib")
DEFAULT_VALIDATION_RESIDUALS_PATH = Path("data/processed/validation_residuals.parquet")
DEFAULT_REPORTS_DIR = Path("reports")


def assign_operating_cloud_bin(
    tcdc_pct: float | None,
    weather_missing: int | float | None = 0,
) -> str:
    """Assign operating cloud cover bin from forecast total cloud cover.

    Bins:
    - 'clear': tcdc_pct < 20.0
    - 'mixed': 20.0 <= tcdc_pct <= 80.0
    - 'overcast': tcdc_pct > 80.0
    - 'no_weather': weather data missing or unavailable
    """
    if weather_missing == 1 or tcdc_pct is None or pd.isna(tcdc_pct):
        return "no_weather"
    val = float(tcdc_pct)
    if val < 20.0:
        return "clear"
    elif val <= 80.0:
        return "mixed"
    else:
        return "overcast"


def assign_clearsky_index_bin(csi: float | None) -> str:
    """Assign clear-sky index bin from latest available clear-sky index."""
    if csi is None or pd.isna(csi):
        return "unknown"
    val = float(csi)
    if val < 0.4:
        return "csi_low"
    elif val <= 0.8:
        return "csi_mid"
    else:
        return "csi_high"


def compute_validation_residuals(
    eval_config_path: Path | str = DEFAULT_EVAL_CONFIG,
    hourly_path: Path | str = DEFAULT_HOURLY_PATH,
    features_path: Path | str = DEFAULT_FEATURES_PATH,
    output_path: Path | str | None = None,
    *, bundle_path: Path | str | None = None,
) -> pd.DataFrame:
    """Compute dev-only OOF residuals from an explicit bundle's estimator recipe.

    No implicit champion parameters or output overwrite. These are cloned fold
    models, not predictions from the supplied bundle's already-fitted weights.
    """
    if bundle_path is None:
        raise ValueError("Calibration requires an explicit bundle_path/model identity")
    from src.models.deviation_workflow import compute_residuals

    return compute_residuals(eval_config_path, hourly_path, features_path, bundle_path, output_path)


def calibrate_deviation_bounds(
    val_residuals_df: pd.DataFrame,
    output_yaml_path: Path | str | None = None,
) -> dict[str, Any]:
    """Calibrate 1st and 99th percentile residual thresholds from validation data.

    Evaluated strictly on VALID daylight hours from the 3 expanding validation folds.
    Records clear-sky sub-bin counts; the frozen four-cloud-bin policy is retained.
    Requires model-bound development-fold provenance and never writes by default.

    Args:
        val_residuals_df: DataFrame of validation residuals from compute_validation_residuals.
        output_yaml_path: Destination path for frozen deviation configuration YAML.

    Returns:
        Calibrated bounds configuration dictionary.
    """
    provenance = val_residuals_df.attrs.get("calibration_provenance")
    if not provenance or provenance.get("method") != "cloned_bundle_expanding_dev_folds":
        raise ValueError("Calibration requires model-bound development OOF provenance")
    if (pd.to_datetime(val_residuals_df["timestamp"], utc=True)
            >= pd.Timestamp(provenance["protected_test_start_utc"])).any():
        raise RuntimeError("PROTECTED TEST SET: calibration residuals overlap final test")
    mask = (val_residuals_df["quality_flag"] == "VALID") & val_residuals_df["is_daylight"]
    df_valid = val_residuals_df[mask].copy()
    if df_valid.empty or not np.isfinite(df_valid["residual"].to_numpy(dtype=float)).all():
        raise ValueError("Calibration requires nonempty finite daylight residuals")
    ct = pd.crosstab(df_valid["forecast_cloud_bin"], df_valid["clearsky_index_bin"])
    has_sub_200 = bool((ct < 200).any().any())
    csi_note = (
        "Cloud-only calibration; clear-sky sub-bin counts: " + ct.to_json()
        + ". No clear-sky cross-stratification is applied."
    )

    bounds_dict: dict[str, dict[str, float]] = {}
    for cloud_bin, grp in df_valid.groupby("forecast_cloud_bin"):
        res = grp["residual"].values
        p1 = float(np.percentile(res, 1))
        p99 = float(np.percentile(res, 99))
        if not np.isfinite([p1, p99]).all() or round(p1, 2) >= round(p99, 2):
            raise ValueError(f"Insufficient residual variation to calibrate bin {cloud_bin}")
        bounds_dict[str(cloud_bin)] = {
            "bound_low": round(p1, 2),
            "bound_high": round(p99, 2),
            "sample_count": int(len(res)),
            "mean_residual_kwh": round(float(np.mean(res)), 2),
            "std_residual_kwh": round(float(np.std(res)), 2),
        }

    cfg: dict[str, Any] = {
        "schema_version": "1.0.0",
        "calibrated_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "provenance": provenance,
        "calibration_window": {
            "start_utc": df_valid["timestamp"].min().isoformat(),
            "end_utc": df_valid["timestamp"].max().isoformat(),
            "total_daylight_validation_samples": int(len(df_valid)),
        },
        "rule_parameters": {
            "consecutive_hours_threshold": 2,
            "evaluation_cadence": "hourly",
            "suppression_conditions": [
                "night (solar_elevation_deg <= 0)",
                "missing target (target_kwh is NaN)",
                "missing forecast",
                "quality_flag != 'VALID'",
            ],
            "reset_conditions": [
                "night",
                "missing or suppressed hour",
                "residual within calibrated bounds",
                "direction flip",
            ],
        },
        "stratification_decision": {
            "primary_stratification": "forecast_cloud_bin",
            "clearsky_index_bins_merged": has_sub_200,
            "justification": csi_note,
        },
        "bounds": bounds_dict,
    }

    if output_yaml_path is not None:
        out_p = Path(output_yaml_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with out_p.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(cfg, stream, sort_keys=False)
    return cfg


def load_deviation_config(config_path: Path | str = DEFAULT_DEVIATION_CONFIG) -> dict[str, Any]:
    """Load frozen deviation detector configuration from YAML."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Deviation config not found at {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _alert_timestamp(value: Any) -> pd.Timestamp:
    """Require an explicit timezone and normalize alert identities to UTC."""
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError("Alert timestamps require an explicit timezone/UTC offset")
    return timestamp.tz_convert("UTC")


class DeviationDetector:
    """Incremental two-hour breach detector for one site and frozen model version.

    Keep one instance across chronologically ordered, nonoverlapping batches. Each
    process call emits immutable snapshots only when the second eligible adjacent
    label confirms an episode. summaries() provides the retrospective full episodes.
    State is in memory; durable checkpoints, late-label buffering and delivery
    deduplication are responsibilities of the future monitoring service.
    """

    def __init__(
        self, deviation_config: dict[str, Any] | None = None,
        site_id: str = "2107", model_name: str = "solarcast-f1", model_version: str = "v4",
    ) -> None:
        """Bind detector state to calibrated bounds and one immutable model context."""
        config = load_deviation_config() if deviation_config is None else deviation_config
        self.bounds = copy.deepcopy(config["bounds"])
        self.schema_version = config.get("schema_version", "1.0.0")
        if config.get("rule_parameters", {}).get("consecutive_hours_threshold", 2) != 2:
            raise ValueError("This detector requires the frozen two-hour confirmation rule")
        for bound in self.bounds.values():
            low, high = bound["bound_low"], bound["bound_high"]
            if not np.isfinite([low, high]).all() or low >= high:
                raise ValueError("Deviation bounds must be finite and increasing")
        self.site_id = str(site_id)
        self.model_name = model_name
        self.model_version = model_version
        self._last_timestamp: pd.Timestamp | None = None
        self._active: dict[str, Any] | None = None
        self._episodes: list[dict[str, Any]] = []

    def process(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        """Process a batch and emit new confirmations, without waiting for episode closure.

        Duplicate target times within a batch suppress that whole hour. Overlaps
        with earlier calls reject before changing state; buffer/deduplicate upstream.
        Optional actual_available_at_utc establishes label-based trigger timing.
        Without it, timing fields are null rather than guessed from a reporting SLA.
        """
        if frame.empty:
            return []
        df = frame.copy()
        df["timestamp"] = pd.DatetimeIndex([_alert_timestamp(t) for t in df["timestamp"]])
        df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
        if self._last_timestamp is not None and df["timestamp"].min() <= self._last_timestamp:
            raise ValueError("Batch overlaps already processed hours; supply ordered new intervals")
        if "site_id" in df and not df["site_id"].astype(str).eq(self.site_id).all():
            raise ValueError("Detector cannot combine labels from different sites")
        # Preflight all timing fields before mutating state.
        if "issue_time_utc" in df:
            issues = pd.DatetimeIndex([_alert_timestamp(t) for t in df["issue_time_utc"]])
            if not (issues == pd.DatetimeIndex(df["timestamp"])).all():
                raise ValueError("Stored F1 issue times must match target interval starts")
        if "actual_available_at_utc" in df:
            available = [
                None if pd.isna(t) else _alert_timestamp(t)
                for t in df["actual_available_at_utc"]
            ]
            if any(t is not None and t < h + pd.Timedelta(hours=1)
                   for t, h in zip(available, df["timestamp"], strict=True)):
                raise ValueError("An hourly label cannot arrive before its target interval ends")
            df["actual_available_at_utc"] = available
        ambiguous = df["timestamp"].duplicated(keep=False)
        off_grid = df["timestamp"] != df["timestamp"].dt.floor("h")
        emitted = []
        for index, row in df.iterrows():
            timestamp = row["timestamp"]
            if (self._last_timestamp is not None
                    and timestamp - self._last_timestamp != pd.Timedelta(hours=1)):
                self._active = None
            self._last_timestamp = timestamp
            actual, forecast = row.get("actual", np.nan), row.get("forecast", np.nan)
            elevation = row.get("solar_elevation_deg", 10.0)
            daylight = row.get("is_daylight", True)
            valid_numbers = all(
                isinstance(value, int | float | np.number) and np.isfinite(value)
                for value in (actual, forecast, elevation)
            )
            suppressed = (
                ambiguous.iloc[index] or off_grid.iloc[index] or not valid_numbers
                or not isinstance(daylight, bool | np.bool_) or not bool(daylight) or elevation <= 0
                or str(row.get("quality_flag", "VALID")) != "VALID"
            )
            if suppressed:
                self._active = None
                continue
            residual = float(actual - forecast)
            if not np.isfinite(residual):
                self._active = None
                continue
            cloud_bin = row.get("forecast_cloud_bin")
            if cloud_bin is None or pd.isna(cloud_bin) or not cloud_bin:
                cloud_bin = assign_operating_cloud_bin(
                    row.get("tcdc_pct"), row.get("weather_missing")
                )
            if cloud_bin not in self.bounds:
                cloud_bin = "no_weather"
            low, high = self.bounds[cloud_bin]["bound_low"], self.bounds[cloud_bin]["bound_high"]
            direction = "negative" if residual < low else "positive" if residual > high else None
            if direction is None:
                self._active = None
                continue
            available_at = row.get("actual_available_at_utc")
            available_at = None if pd.isna(available_at) else available_at
            if self._active is None or direction != self._active["direction"]:
                self._active = {
                    "start": timestamp, "direction": direction, "bin": cloud_bin,
                    "count": 1, "residual_sum": residual, "available": available_at,
                    "event": None,
                }
                continue
            active = self._active
            active["count"] += 1
            active["residual_sum"] += residual
            if active["event"] is None:
                start = active["start"]
                identity = json.dumps([
                    self.site_id, self.model_name, self.model_version, start.isoformat(), direction,
                    self.bounds,
                ], sort_keys=True, allow_nan=False)
                trigger = (
                    max(active["available"], available_at)
                    if active["available"] is not None and available_at is not None else None
                )
                active["event"] = {
                    "schema_version": self.schema_version,
                    "event_id": "DEV-" + hashlib.sha256(identity.encode()).hexdigest()[:24],
                    "site_id": self.site_id, "alert_type": "residual_deviation",
                    "model_name": self.model_name, "model_version": self.model_version,
                    "window_start_utc": start.isoformat(), "direction": direction,
                    "bin": active["bin"],
                    "bound_low": self.bounds[active["bin"]]["bound_low"],
                    "bound_high": self.bounds[active["bin"]]["bound_high"],
                    "confirmation_issue_time_utc": timestamp.isoformat(),
                    "target_interval_start_utc": timestamp.isoformat(),
                    "target_interval_end_utc": (timestamp + pd.Timedelta(hours=1)).isoformat(),
                    "actual_available_at_utc": (
                        available_at.isoformat() if available_at is not None else None
                    ),
                    "triggered_at_utc": trigger.isoformat() if trigger is not None else None,
                    "detection_delay_hours": (
                        float((trigger - start) / pd.Timedelta(hours=1))
                        if trigger is not None else None
                    ),
                    "created_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
                }
                self._episodes.append(active["event"])
            event = active["event"]
            event.update({
                # Legacy window_end is the last target START, not its end/label arrival.
                "window_end_utc": timestamp.isoformat(), "consecutive_hours": active["count"],
                "mean_residual_kwh": round(active["residual_sum"] / active["count"], 2),
                "reason": f"{active['count']} consecutive hourly {direction} breaches; "
                          "confirmed at the second eligible label",
            })
            if active["count"] == 2:
                emitted.append(copy.deepcopy(event))
        return emitted

    def summaries(self) -> list[dict[str, Any]]:
        """Return retrospective episode summaries without mutating emitted events."""
        return copy.deepcopy(self._episodes)


def detect_deviations(
    frame: pd.DataFrame,
    deviation_config: dict[str, Any] | None = None,
    site_id: str = "2107",
    model_name: str = "solarcast-f1",
    model_version: str = "v4",
) -> list[dict[str, Any]]:
    """Summarize contiguous breach episodes in one complete retrospective batch.

    For incremental monitoring keep a DeviationDetector instance across calls and
    use process() to emit at confirmation. Independent calls here reset all state.
    Frozen thresholds and existing summary fields retain their meanings.
    """
    detector = DeviationDetector(deviation_config, site_id, model_name, model_version)
    detector.process(frame)
    return detector.summaries()


def get_test_period_dataset(
    eval_config_path: Path | str = DEFAULT_EVAL_CONFIG,
    hourly_path: Path | str = DEFAULT_HOURLY_PATH,
    features_path: Path | str = DEFAULT_FEATURES_PATH,
    bundle_path: Path | str = DEFAULT_BUNDLE_PATH,
) -> pd.DataFrame:
    """Retired historical benchmark entrypoint; never reopen final data implicitly."""
    raise RuntimeError(
        "PROTECTED TEST SET: this historical deviation benchmark reader is retired. "
        "Use explicit development calibration or bounded stored-record monitoring."
    )


def run_injected_evaluation(
    test_df: pd.DataFrame,
    bounds: dict[str, Any],
    seed: int = 42,
    n_episodes: int = 20,
    min_separation_hours: int = 12,
) -> dict[str, Any]:
    """Evaluate detector on synthetic fault injections on a copy of the test period.

    Injects at a fixed seed:
    (a) 30% power drop for 4 consecutive daylight hours
    (b) zero output for 3 consecutive daylight hours
    (c) bad value (negative or above 893 kWh) for 2 consecutive daylight hours
    (d) isolated single-hour 30% drop (should NOT alert)
    """
    rng = np.random.default_rng(seed)

    # Find candidate daylight stretches with VALID quality and weather present
    valid_daylight = (
        (test_df["quality_flag"] == "VALID")
        & (test_df["is_daylight"])
        & (test_df["solar_elevation_deg"] > 10.0)
        & (test_df["weather_missing"] == 0)
        & (test_df["target_kwh"] > 50.0)  # non-trivial generation
    )
    candidate_indices = np.where(valid_daylight.values)[0]

    # Helper to find non-overlapping anchor indices separated by min_separation_hours
    def select_anchors(episode_duration: int) -> list[int]:
        selected: list[int] = []
        shuffled = rng.permutation(candidate_indices)
        for idx in shuffled:
            # Check consecutive stretch validity
            end_idx = idx + episode_duration
            if end_idx > len(test_df):
                continue
            stretch_valid = valid_daylight.iloc[idx:end_idx].all()
            if not stretch_valid:
                continue
            # Check separation from previously selected anchors
            conflict = False
            for s in selected:
                if abs(idx - s) < (min_separation_hours + episode_duration):
                    conflict = True
                    break
            if not conflict:
                selected.append(int(idx))
                if len(selected) == n_episodes:
                    break
        selected.sort()
        return selected

    results: dict[str, Any] = {}

    # (a) 30% power drop for 4 consecutive hours
    anchors_a = select_anchors(4)
    df_a = test_df.copy()
    for idx in anchors_a:
        for offset in range(4):
            df_a.loc[idx + offset, "actual"] = df_a.loc[idx + offset, "actual"] * 0.70

    alerts_a = detect_deviations(df_a, bounds)
    detected_a: list[int] = []
    delays_a: list[float] = []
    for anc in anchors_a:
        t_start = df_a.loc[anc, "timestamp"]
        t_end = df_a.loc[anc + 3, "timestamp"]
        hit = [
            al
            for al in alerts_a
            if pd.to_datetime(al["window_start_utc"]) <= t_end
            and pd.to_datetime(al["window_end_utc"]) >= t_start
        ]
        if hit:
            detected_a.append(anc)
            # Delay in hours from anomaly start to alert trigger
            t_trig = pd.to_datetime(hit[0]["window_start_utc"]) + datetime.timedelta(hours=1)
            delays_a.append(max(1.0, (t_trig - t_start).total_seconds() / 3600.0))

    results["injected_30pct_drop_4h"] = {
        "episodes_injected": len(anchors_a),
        "episodes_detected": len(detected_a),
        "detection_rate_pct": round(len(detected_a) / len(anchors_a) * 100.0, 2),
        "median_delay_hours": round(float(np.median(delays_a)), 2) if delays_a else 0.0,
        "max_delay_hours": round(float(np.max(delays_a)), 2) if delays_a else 0.0,
    }

    # (b) Zero output for 3 consecutive hours
    anchors_b = select_anchors(3)
    df_b = test_df.copy()
    for idx in anchors_b:
        for offset in range(3):
            df_b.loc[idx + offset, "actual"] = 0.0

    alerts_b = detect_deviations(df_b, bounds)
    detected_b: list[int] = []
    delays_b: list[float] = []
    for anc in anchors_b:
        t_start = df_b.loc[anc, "timestamp"]
        t_end = df_b.loc[anc + 2, "timestamp"]
        hit = [
            al
            for al in alerts_b
            if pd.to_datetime(al["window_start_utc"]) <= t_end
            and pd.to_datetime(al["window_end_utc"]) >= t_start
        ]
        if hit:
            detected_b.append(anc)
            t_trig = pd.to_datetime(hit[0]["window_start_utc"]) + datetime.timedelta(hours=1)
            delays_b.append(max(1.0, (t_trig - t_start).total_seconds() / 3600.0))

    results["injected_zero_output_3h"] = {
        "episodes_injected": len(anchors_b),
        "episodes_detected": len(detected_b),
        "detection_rate_pct": round(len(detected_b) / len(anchors_b) * 100.0, 2),
        "median_delay_hours": round(float(np.median(delays_b)), 2) if delays_b else 0.0,
        "max_delay_hours": round(float(np.max(delays_b)), 2) if delays_b else 0.0,
    }

    # (c) Bad value (negative or above 893 kWh) for 2 consecutive hours
    anchors_c = select_anchors(2)
    df_c = test_df.copy()
    for i, idx in enumerate(anchors_c):
        val = -50.0 if i % 2 == 0 else 1250.0
        for offset in range(2):
            df_c.loc[idx + offset, "actual"] = val

    alerts_c = detect_deviations(df_c, bounds)
    detected_c: list[int] = []
    delays_c: list[float] = []
    for anc in anchors_c:
        t_start = df_c.loc[anc, "timestamp"]
        t_end = df_c.loc[anc + 1, "timestamp"]
        hit = [
            al
            for al in alerts_c
            if pd.to_datetime(al["window_start_utc"]) <= t_end
            and pd.to_datetime(al["window_end_utc"]) >= t_start
        ]
        if hit:
            detected_c.append(anc)
            t_trig = pd.to_datetime(hit[0]["window_start_utc"]) + datetime.timedelta(hours=1)
            delays_c.append(max(1.0, (t_trig - t_start).total_seconds() / 3600.0))

    results["injected_bad_value_2h"] = {
        "episodes_injected": len(anchors_c),
        "episodes_detected": len(detected_c),
        "detection_rate_pct": round(len(detected_c) / len(anchors_c) * 100.0, 2),
        "median_delay_hours": round(float(np.median(delays_c)), 2) if delays_c else 0.0,
        "max_delay_hours": round(float(np.max(delays_c)), 2) if delays_c else 0.0,
    }

    # (d) Isolated single-hour 30% drop (should NOT alert)
    anchors_d = select_anchors(1)
    df_d = test_df.copy()
    for idx in anchors_d:
        df_d.loc[idx, "actual"] = df_d.loc[idx, "actual"] * 0.70

    alerts_d = detect_deviations(df_d, bounds)
    falsely_alerted_d = 0
    for anc in anchors_d:
        t_anc = df_d.loc[anc, "timestamp"]
        hit = [
            al
            for al in alerts_d
            if pd.to_datetime(al["window_start_utc"])
            == pd.to_datetime(al["window_end_utc"])
            == t_anc
        ]
        if hit:
            falsely_alerted_d += 1

    results["injected_isolated_single_hour_drop"] = {
        "episodes_injected": len(anchors_d),
        "episodes_alerted": falsely_alerted_d,
        "compliance_rate_pct": round((1.0 - falsely_alerted_d / len(anchors_d)) * 100.0, 2),
    }

    # Unmodified baseline: False-alert rate on unmodified test period
    unmodified_alerts = detect_deviations(test_df, bounds)
    daylight_test_hours = int(((test_df["quality_flag"] == "VALID") & test_df["is_daylight"]).sum())
    fa_per_1000 = (
        round(len(unmodified_alerts) / daylight_test_hours * 1000.0, 2)
        if daylight_test_hours > 0
        else 0.0
    )

    results["unmodified_test_baseline"] = {
        "valid_daylight_hours": daylight_test_hours,
        "total_alert_events": len(unmodified_alerts),
        "false_alerts_per_1000_daylight_hours": fa_per_1000,
        "alert_events_summary": [
            {
                "event_id": a["event_id"],
                "start": a["window_start_utc"],
                "end": a["window_end_utc"],
                "direction": a["direction"],
                "bin": a["bin"],
                "hours": a["consecutive_hours"],
                "mean_residual": a["mean_residual_kwh"],
            }
            for a in unmodified_alerts
        ],
    }

    return results


def run_natural_suspect_zero_evaluation(
    eval_config_path: Path | str = DEFAULT_EVAL_CONFIG,
    hourly_path: Path | str = DEFAULT_HOURLY_PATH,
    features_path: Path | str = DEFAULT_FEATURES_PATH,
    bundle_path: Path | str = DEFAULT_BUNDLE_PATH,
    bounds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Retired historical benchmark entrypoint; never reopen final data implicitly."""
    raise RuntimeError(
        "PROTECTED TEST SET: this historical deviation benchmark reader is retired. "
        "Use explicit development calibration or bounded stored-record monitoring."
    )


def main() -> int:
    """Route explicit, isolated calibration or bounded monitoring commands."""
    from src.models.deviation_workflow import main as workflow_main

    return workflow_main()


if __name__ == "__main__":
    raise SystemExit(main())
