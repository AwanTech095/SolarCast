"""End-to-end synthetic calibration/monitoring commands and leakage regressions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.dummy import DummyRegressor

from src.models import deviation_workflow as workflow
from src.models.bundle import ModelBundle, save_bundle
from src.models.candidates import file_sha256
from src.models.deviation import calibrate_deviation_bounds


@pytest.fixture
def inputs(tmp_path: Path, monkeypatch) -> dict:
    """Public-shape synthetic data, including extreme protected test labels."""
    monkeypatch.chdir(tmp_path)
    times = pd.date_range("2022-01-01", periods=100, freq="h", tz="UTC")
    target = 100 + np.sin(np.arange(100)) * 10
    target[80:] = 888  # Must never enter fit, validation residuals or threshold calibration.
    hourly = pd.DataFrame(
        {
            "site_id": "2107",
            "interval_start_utc": times,
            "target_kwh": target,
            "quality_flag": "VALID",
            "solar_elevation_deg": 30.0,
            "is_daylight": True,
        }
    )
    hourly.to_parquet("hourly.parquet", index=False)
    pd.DataFrame(
        {
            "site_id": "2107",
            "event_timestamp": times,
            "solar_elevation_target_deg": 30.0,
            "tcdc_pct": 0.0,
            "weather_missing": 0,
            "clearsky_index_latest_avail": 0.5,
            "production_cutoff_utc": times - pd.Timedelta(minutes=15),
            "weather_available_at_utc": times - pd.Timedelta(hours=1),
            "source_event_time_max_utc": times - pd.Timedelta(minutes=15),
        }
    ).to_parquet("features.parquet", index=False)
    Path("evaluation.yaml").write_text(
        yaml.safe_dump(
            {
                "site_id": "2107",
                "evaluation_window": {"start_utc": str(times[0]), "end_utc": str(times[-1])},
                "operational_timing": {"production_available_delay_minutes": 15},
            }
        )
    )
    estimator = DummyRegressor().fit([[1], [2]], [1, 2])
    bundle = ModelBundle(
        estimator=estimator,
        feature_names=["solar_elevation_target_deg"],
        training_window={"start_utc": str(times[0]), "end_utc": str(times[1])},
        data_hashes={"hourly_parquet": file_sha256(Path("hourly.parquet"))},
        mlflow_run_id="synthetic-calibration-run",
    )
    save_bundle(bundle, "bundle.joblib")
    return {"times": times, "bundle": bundle}


def residuals() -> pd.DataFrame:
    """Compute calibration inputs entirely from temporary files."""
    return workflow.compute_residuals(
        "evaluation.yaml", "hourly.parquet", "features.parquet", "bundle.joblib"
    )


def test_calibration_excludes_holdout_and_purges_label_boundaries(inputs, monkeypatch) -> None:
    """All three fit/validation folds are earlier than protected test targets."""
    original = DummyRegressor.fit
    fits = []

    def checked_fit(self, x, y, **kwargs):
        assert max(y) < 150, "Held-out extreme labels leaked into fit"
        fits.append(len(y))
        return original(self, x, y, **kwargs)

    monkeypatch.setattr(DummyRegressor, "fit", checked_fit)
    before = Path("bundle.joblib").read_bytes()
    result = residuals()
    assert result["timestamp"].max() < inputs["times"][80]
    assert result["actual"].max() < 150
    assert len(fits) == 3
    for fold in result.attrs["calibration_provenance"]["fold_windows"]:
        assert pd.Timestamp(fold["training_end_utc"]) + pd.Timedelta(hours=1, minutes=15) <= (
            pd.Timestamp(fold["validation_start_utc"])
        )
    config = calibrate_deviation_bounds(result)
    assert config["provenance"]["model_bundle_sha256"] == file_sha256(Path("bundle.joblib"))
    assert Path("bundle.joblib").read_bytes() == before
    assert not Path("configs/deviation.yaml").exists()


def test_calibration_refuses_changed_data_lineage(inputs) -> None:
    """A model's configured data identity must match the actual source snapshot."""
    with Path("hourly.parquet").open("ab") as stream:
        stream.write(b"modified")
    with pytest.raises(ValueError, match="hash"):
        residuals()


def test_calibration_refuses_duplicate_feature_hours(inputs) -> None:
    frame = pd.read_parquet("features.parquet")
    pd.concat([frame, frame.iloc[:1]]).to_parquet("features.parquet")
    with pytest.raises(ValueError, match="unique"):
        residuals()


def test_calibration_refuses_future_feature_availability(inputs) -> None:
    frame = pd.read_parquet("features.parquet")
    frame["weather_available_at_utc"] = frame["event_timestamp"] + pd.Timedelta(hours=1)
    frame.to_parquet("features.parquet", index=False)
    with pytest.raises(ValueError, match="Future source"):
        residuals()


def test_calibration_requires_oof_identity(inputs) -> None:
    frame = residuals()
    frame.attrs.clear()
    with pytest.raises(ValueError, match="provenance"):
        calibrate_deviation_bounds(frame)


def test_calibration_cannot_reuse_test_rows_with_preserved_attrs(inputs) -> None:
    frame = residuals()
    frame.loc[0, "timestamp"] = inputs["times"][80]
    with pytest.raises(RuntimeError, match="PROTECTED"):
        calibrate_deviation_bounds(frame)


def monitoring_frame() -> pd.DataFrame:
    """Actual stored predictions and explicit label arrivals, earlier than holdout."""
    times = pd.date_range("2022-01-01T10:00:00Z", periods=3, freq="h")
    return pd.DataFrame(
        {
            "timestamp": times,
            "issue_time_utc": times,
            "actual_available_at_utc": times + pd.Timedelta(hours=1, minutes=15),
            "site_id": "2107",
            "model_bundle_sha256": file_sha256(Path("bundle.joblib")),
            "actual": 0.0,
            "forecast": 500.0,
            "quality_flag": "VALID",
            "is_daylight": True,
            "solar_elevation_deg": 30.0,
            "forecast_cloud_bin": "clear",
        }
    )


def monitor(frame, config, **overrides) -> dict:
    options = {
        "start_utc": "2022-01-01T10:00:00Z",
        "end_utc": "2022-01-01T13:00:00Z",
        "as_of_utc": "2022-01-01T13:00:00Z",
    }
    options.update(overrides)
    return workflow.monitor_records(frame, config, "bundle.joblib", **options)


def test_monitoring_uses_only_arrived_labels_without_fit(inputs, monkeypatch) -> None:
    config = calibrate_deviation_bounds(residuals())
    monkeypatch.setattr(DummyRegressor, "fit", lambda *a, **k: pytest.fail("Monitoring refit"))
    result = monitor(monitoring_frame(), config)
    assert result["visible_label_rows"] == 2
    assert result["excluded_unavailable_labels"] == 1
    assert len(result["events"]) == 1
    assert result["events"][0]["triggered_at_utc"] == "2022-01-01T12:15:00+00:00"


def test_missing_labels_are_reported_without_inventing_accuracy(inputs) -> None:
    config = calibrate_deviation_bounds(residuals())
    frame = monitoring_frame()
    frame["actual_available_at_utc"] = pd.NaT
    result = monitor(frame, config)
    assert result["visible_label_rows"] == 0
    assert result["status"] == "no_visible_labels"
    assert result["excluded_unavailable_labels"] == 3
    assert result["events"] == []


def test_old_unbound_thresholds_cannot_be_used_silently(inputs) -> None:
    with pytest.raises(ValueError, match="matching model identity"):
        monitor(monitoring_frame(), {"bounds": {}})


def test_calibration_cannot_overwrite_approved_config(inputs) -> None:
    frame = residuals()
    approved = Path("approved.yaml")
    approved.write_bytes(b"existing frozen thresholds")
    with pytest.raises(FileExistsError):
        calibrate_deviation_bounds(frame, approved)
    assert approved.read_bytes() == b"existing frozen thresholds"


def test_incomplete_monitoring_command_fails_before_io(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["deviation", "--evaluate", "--bundle", "absent.joblib"])
    monkeypatch.setattr(workflow.pd, "read_parquet", lambda *a, **k: pytest.fail("Unexpected I/O"))
    with pytest.raises(SystemExit) as error:
        workflow.main()
    assert error.value.code == 2


def test_monitoring_window_cannot_be_silently_ignored_in_calibration(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["deviation", "--calibrate", "--bundle", "absent.joblib",
                                     "--start-utc", "2022-01-01T00:00:00Z"])
    with pytest.raises(SystemExit) as error:
        workflow.main()
    assert error.value.code == 2


@pytest.mark.parametrize("field,value", [("site_id", "wrong"), ("model_bundle_sha256", "wrong")])
def test_monitoring_checks_site_and_model(inputs, field: str, value: str) -> None:
    config = calibrate_deviation_bounds(residuals())
    frame = monitoring_frame()
    frame[field] = value
    with pytest.raises(ValueError, match="different"):
        monitor(frame, config)


def test_monitoring_refuses_protected_calendar_even_when_opened(inputs) -> None:
    config = calibrate_deviation_bounds(residuals())
    with pytest.raises(RuntimeError, match="PROTECTED"):
        monitor(
            monitoring_frame(),
            config,
            start_utc=str(inputs["times"][80]),
            end_utc=str(inputs["times"][81]),
            as_of_utc=str(inputs["times"][82]),
        )


def test_evaluation_does_not_calibrate_and_preserves_inputs(inputs, monkeypatch) -> None:
    """Run the actual CLI paths and outputs against the synthetic data packet."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "deviation",
            "--calibrate",
            "--bundle",
            "bundle.joblib",
            "--eval-config",
            "evaluation.yaml",
            "--hourly",
            "hourly.parquet",
            "--features",
            "features.parquet",
            "--reports-dir",
            "runs",
        ],
    )
    assert workflow.main() == 0
    run = next(Path("runs").iterdir())
    protected = {
        path: path.read_bytes() for path in [Path("bundle.joblib"), run / "deviation.yaml"]
    }
    monitoring_frame().to_parquet("records.parquet", index=False)
    monkeypatch.setattr(workflow, "compute_residuals", lambda *a, **k: pytest.fail("Recalibration"))
    monkeypatch.setattr(
        "sys.argv",
        [
            "deviation",
            "--evaluate",
            "--bundle",
            "bundle.joblib",
            "--deviation-config",
            str(run / "deviation.yaml"),
            "--records",
            "records.parquet",
            "--reports-dir",
            "runs",
            "--start-utc",
            "2022-01-01T10:00:00Z",
            "--end-utc",
            "2022-01-01T13:00:00Z",
            "--as-of-utc",
            "2022-01-01T13:00:00Z",
        ],
    )
    assert workflow.main() == 0
    assert len(list(Path("runs").iterdir())) == 2
    for path, content in protected.items():
        assert path.read_bytes() == content
    summaries = [json.loads(p.read_text()) for p in Path("runs").glob("*/summary.json")]
    assert any(s.get("mode") == "bounded_operational_records" for s in summaries)
