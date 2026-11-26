"""Explicit dev-only calibration and bounded stored-record monitoring workflows."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit

from src.models.bundle import load_bundle
from src.models.candidates import file_sha256
from src.models.deviation import (
    DeviationDetector,
    _alert_timestamp,
    assign_clearsky_index_bin,
    assign_operating_cloud_bin,
    calibrate_deviation_bounds,
    load_deviation_config,
)


def compute_residuals(
    eval_config_path: Path | str,
    hourly_path: Path | str,
    features_path: Path | str,
    bundle_path: Path | str,
    output_path: Path | str | None = None,
) -> pd.DataFrame:
    """Clone the supplied estimator recipe on three purged expanding dev folds.

    Final targets never enter fit/predict/calibration. Original fitted bundle weights
    are not modified. The source table must contain each daylight site/hour exactly.
    """
    config = yaml.safe_load(Path(eval_config_path).read_text(encoding="utf-8"))
    bundle = load_bundle(bundle_path)
    if file_sha256(Path(hourly_path)) != bundle.data_hashes.get("hourly_parquet"):
        raise ValueError("Hourly data hash does not match supplied model lineage")
    hourly, features = pd.read_parquet(hourly_path), pd.read_parquet(features_path)
    if bundle.feature_config is not None and features.attrs.get("feature_config") != (
        bundle.feature_config
    ):
        raise ValueError("Feature table must use the bundle's frozen feature configuration")
    site = str(config["site_id"])
    for frame, timestamp_column in ((hourly, "interval_start_utc"), (features, "event_timestamp")):
        frame[timestamp_column] = pd.DatetimeIndex(
            [_alert_timestamp(t) for t in frame[timestamp_column]]
        )
        if "site_id" not in frame or not frame["site_id"].astype(str).eq(site).all():
            raise ValueError("Calibration source rows must belong to the configured site")
        if frame[timestamp_column].duplicated().any():
            raise ValueError("Calibration site/hour keys must be unique")
    for column in (
        "production_cutoff_utc", "weather_available_at_utc", "source_event_time_max_utc"
    ):
        if column not in features:
            raise ValueError(f"Feature availability audit column missing: {column}")
        stamps = pd.to_datetime(features[column], utc=True)
        if column != "weather_available_at_utc" and stamps.isna().any():
            raise ValueError(f"Missing source timing evidence: {column}")
        if (stamps > features["event_timestamp"]).any():
            raise ValueError(f"Future source availability in calibration: {column}")
    window = config["evaluation_window"]
    valid = (
        hourly[
            hourly["interval_start_utc"].between(
                pd.to_datetime(window["start_utc"], utc=True),
                pd.to_datetime(window["end_utc"], utc=True),
            )
            & hourly["quality_flag"].eq("VALID")
        ]
        .sort_values("interval_start_utc")
        .reset_index(drop=True)
    )
    split = int(len(valid) * 0.8)
    if split < 8 or split >= len(valid):
        raise ValueError("Insufficient chronological development/final-test partitions")
    test_start = valid.iloc[split]["interval_start_utc"]
    test_end = pd.to_datetime(window["end_utc"], utc=True) + pd.Timedelta(hours=1)
    dev = valid.iloc[:split].copy()
    aux = ["tcdc_pct", "weather_missing", "clearsky_index_latest_avail"]
    columns = list(dict.fromkeys(["event_timestamp", *bundle.feature_names, *aux]))
    dev = dev.merge(
        features[columns],
        left_on="interval_start_utc",
        right_on="event_timestamp",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if ((dev["solar_elevation_deg"] > 0) & dev["_merge"].ne("both")).any():
        raise ValueError("Missing exact daylight feature hours in development source")
    delay = int(config["operational_timing"]["production_available_delay_minutes"])
    if delay < 0:
        raise ValueError("Production delay cannot be negative")
    records = []
    fold_windows = []
    for fold, (train_indices, val_indices) in enumerate(TimeSeriesSplit(3).split(dev), 1):
        training, validation = dev.iloc[train_indices], dev.iloc[val_indices]
        available = training["interval_start_utc"] + pd.Timedelta(hours=1, minutes=delay)
        training = training[
            (available <= validation["interval_start_utc"].min())
            & (training["solar_elevation_deg"] > 0)
        ]
        if training.empty:
            raise ValueError("No available daylight training labels in calibration fold")
        fold_windows.append(
            {
                "fold": fold,
                "training_end_utc": training["interval_start_utc"].max().isoformat(),
                "validation_start_utc": validation["interval_start_utc"].min().isoformat(),
                "validation_end_utc": validation["interval_start_utc"].max().isoformat(),
                "training_rows": len(training),
            }
        )
        estimator = clone(bundle.estimator)
        estimator.fit(training[bundle.feature_names].to_numpy(), training["target_kwh"].to_numpy())
        fold_bundle = replace(bundle, estimator=estimator)
        prediction = fold_bundle.predict(validation)
        part = validation.copy()
        part["fold"], part["forecast"] = fold, prediction
        part["timestamp"] = part["interval_start_utc"]
        part["actual"] = part["target_kwh"]
        part["residual"] = part["actual"] - part["forecast"]
        part["is_daylight"] = part["solar_elevation_deg"] > 0
        part["forecast_cloud_bin"] = [
            assign_operating_cloud_bin(c, m)
            for c, m in zip(part["tcdc_pct"], part["weather_missing"], strict=True)
        ]
        part["clearsky_index_bin"] = part["clearsky_index_latest_avail"].map(
            assign_clearsky_index_bin
        )
        records.append(part)
    result = pd.concat(records, ignore_index=True).drop(columns="_merge")
    result.attrs["calibration_provenance"] = {
        "method": "cloned_bundle_expanding_dev_folds",
        "site_id": site,
        "model_bundle_sha256": file_sha256(Path(bundle_path)),
        "model_name": bundle.model_name,
        "model_version": file_sha256(Path(bundle_path)),
        "source_mlflow_run_id": bundle.mlflow_run_id,
        "feature_set": bundle.feature_set,
        "feature_schema_version": bundle.feature_schema_version,
        "hourly_sha256": file_sha256(Path(hourly_path)),
        "features_sha256": file_sha256(Path(features_path)),
        "evaluation_config_sha256": file_sha256(Path(eval_config_path)),
        "protected_test_start_utc": test_start.isoformat(),
        "protected_test_end_utc": test_end.isoformat(),
        "production_delay_minutes": delay,
        "code_commit": _git_commit(),
        "folds": 3,
        "fold_windows": fold_windows,
    }
    if output_path is not None:
        with Path(output_path).open("xb") as stream:
            result.to_parquet(stream, index=False)
    return result


def _git_commit() -> str:
    """Reuse the tracked model pipeline's commit resolver."""
    from src.models.evaluate import get_git_commit

    return get_git_commit()


def monitor_records(
    records: pd.DataFrame,
    config: dict[str, Any],
    bundle_path: Path | str,
    start_utc: str,
    end_utc: str,
    as_of_utc: str,
) -> dict[str, Any]:
    """Evaluate a bounded stored-F1/actuals log without refitting or loading raw labels.

    Labels must have arrived at the supplied virtual clock. Historical protected
    final-test intervals are rejected, even if their audit has already been opened.
    """
    provenance = config.get("provenance", {})
    model_hash = file_sha256(Path(bundle_path))
    if provenance.get("model_bundle_sha256") != model_hash:
        raise ValueError("Thresholds lack matching model identity; use model-bound calibration")
    start, end, clock = map(_alert_timestamp, (start_utc, end_utc, as_of_utc))
    if start >= end or end > clock:
        raise ValueError("Monitoring requires start < end <= as-of UTC")
    test_start = _alert_timestamp(provenance["protected_test_start_utc"])
    test_end = _alert_timestamp(provenance["protected_test_end_utc"])
    if start < test_end and end > test_start:
        raise RuntimeError(
            "PROTECTED TEST SET: monitoring window overlaps the historical final test"
        )
    bundle = load_bundle(bundle_path)
    earliest_issue = _alert_timestamp(bundle.training_window["end_utc"]) + pd.Timedelta(
        hours=1, minutes=provenance["production_delay_minutes"]
    )
    if start < earliest_issue:
        raise ValueError(
            "Monitoring cannot replay this model before its training labels were available"
        )
    required = {
        "timestamp",
        "issue_time_utc",
        "actual_available_at_utc",
        "site_id",
        "model_bundle_sha256",
        "actual",
        "forecast",
        "quality_flag",
        "is_daylight",
        "solar_elevation_deg",
        "forecast_cloud_bin",
    }
    if not required.issubset(records):
        raise ValueError(f"Missing monitoring log columns: {sorted(required - set(records))}")
    frame = records.copy()
    frame["timestamp"] = pd.DatetimeIndex([_alert_timestamp(t) for t in frame["timestamp"]])
    frame = frame[(frame["timestamp"] >= start) & (frame["timestamp"] < end)].copy()
    available = pd.Series(
        [None if pd.isna(t) else _alert_timestamp(t) for t in frame["actual_available_at_utc"]],
        index=frame.index,
    )
    visible = available.notna() & (pd.to_datetime(available, utc=True) <= clock)
    excluded = int((~visible).sum())
    frame = frame[visible].copy()
    if not frame["model_bundle_sha256"].eq(model_hash).all():
        raise ValueError("Stored forecasts belong to a different model bundle")
    if not frame["site_id"].astype(str).eq(provenance["site_id"]).all():
        raise ValueError("Stored forecasts belong to a different site")
    detector = DeviationDetector(config, provenance["site_id"], bundle.model_name, model_hash)
    events = detector.process(frame)
    return {
        "mode": "bounded_operational_records",
        "code_commit": _git_commit(),
        "model_bundle_sha256": model_hash,
        "start_utc": start.isoformat(),
        "end_utc": end.isoformat(),
        "as_of_utc": clock.isoformat(),
        "visible_label_rows": len(frame),
        "status": "evaluated" if len(frame) else "no_visible_labels",
        "excluded_unavailable_labels": excluded,
        "events": events,
        "final_test_opened": False,
    }


def main() -> int:
    """Run one explicit operation; no operation defaults to calibration or final test."""
    parser = argparse.ArgumentParser(
        description="Model-bound deviation calibration / bounded monitoring"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--calibrate", action="store_true", help="Three dev-only OOF calibration folds"
    )
    mode.add_argument(
        "--evaluate", action="store_true", help="Bounded stored forecasts and arrived labels"
    )
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--eval-config", default="configs/evaluation.yaml")
    parser.add_argument("--hourly", default="data/processed/hourly.parquet")
    parser.add_argument("--features", default="data/processed/feast/features_set_b.parquet")
    parser.add_argument("--deviation-config")
    parser.add_argument("--records")
    parser.add_argument("--start-utc")
    parser.add_argument("--end-utc")
    parser.add_argument("--as-of-utc")
    parser.add_argument("--reports-dir", default="reports/deviation_runs")
    parser.add_argument("--log-mlflow", action="store_true")
    args = parser.parse_args()
    if args.calibrate and any(
        (args.records, args.deviation_config, args.start_utc, args.end_utc, args.as_of_utc)
    ):
        parser.error("Monitoring records/window flags require --evaluate, not --calibrate")
    if args.evaluate and not all(
        (args.records, args.deviation_config, args.start_utc, args.end_utc, args.as_of_utc)
    ):
        parser.error(
            "--evaluate requires --records, --deviation-config, --start-utc, --end-utc, --as-of-utc"
        )
    if args.calibrate:
        residuals = compute_residuals(args.eval_config, args.hourly, args.features, args.bundle)
        summary = calibrate_deviation_bounds(residuals)
    else:
        frame = pd.read_parquet(args.records)
        summary = monitor_records(
            frame,
            load_deviation_config(args.deviation_config),
            args.bundle,
            args.start_utc,
            args.end_utc,
            args.as_of_utc,
        )
    output = Path(args.reports_dir) / uuid.uuid4().hex
    output.mkdir(parents=True, exist_ok=False)
    if args.calibrate:
        residuals.to_parquet(output / "validation_residuals.parquet", index=False)
        with (output / "deviation.yaml").open("x", encoding="utf-8") as stream:
            yaml.safe_dump(summary, stream, sort_keys=False)
    summary["source_files"] = {
        "bundle": file_sha256(Path(args.bundle)),
        **(
            {
                "records": file_sha256(Path(args.records)),
                "thresholds": file_sha256(Path(args.deviation_config)),
            }
            if args.evaluate
            else {}
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    if args.log_mlflow:
        import mlflow

        from src.models.lineage import log_lineage

        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
        mlflow.set_experiment("solarcast-deviation-workflows")
        with mlflow.start_run() as run:
            log_lineage(
                run,
                extra_tags={
                    "bundle_sha256": summary["source_files"]["bundle"],
                    "final_test_opened": "false",
                    "workflow_mode": "dev_calibration" if args.calibrate else "bounded_records",
                    "feature_schema_version": load_bundle(args.bundle).feature_schema_version,
                    "calibration_config_sha256": file_sha256(
                        output / "deviation.yaml" if args.calibrate else Path(args.deviation_config)
                    ),
                },
            )
            mlflow.log_artifacts(str(output))
    print(f"Isolated deviation outputs: {output}")
    return 0
