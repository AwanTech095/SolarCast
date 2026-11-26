"""Regression checks for immutable candidates and exact saved-model evaluation."""

from __future__ import annotations

import contextlib
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

from src.models import train
from src.models.bundle import ModelBundle


def make_bundle(constant: float = 40.0) -> ModelBundle:
    """Fit a small synthetic model before its validation period."""
    model = DummyRegressor(strategy="constant", constant=constant)
    model.fit([[1.0], [2.0]], [40.0, 40.0])
    return ModelBundle(
        estimator=model, feature_names=["solar_elevation_target_deg"],
        training_window={
            "start_utc": "2022-01-01T00:00:00Z", "end_utc": "2022-01-01T01:00:00Z",
        },
        data_hashes={"synthetic_fixture": "fixture-only"}, mlflow_run_id="test-run",
    )


def validation_rows() -> pd.DataFrame:
    """Provide later development labels and a weaker baseline."""
    return pd.DataFrame({
        "issue_time_utc": pd.date_range("2022-01-01T03:00:00Z", periods=3, freq="h"),
        "solar_elevation_target_deg": [30.0, 40.0, 50.0],
        "target_kwh": [40.0, 40.0, 40.0],
        "baseline_prediction_kwh": [50.0, 50.0, 50.0],
        "evaluation_daylight": [True, True, True],
    })


def write_candidate(tmp_path: Path, **overrides: Any) -> Path:
    """Use only temporary synthetic artifacts."""
    from src.models.candidates import write_candidate_artifact

    options = {
        "bundle": make_bundle(), "validation_frame": validation_rows(),
        "cross_validation_gate": {"eligible": True, "failure_reasons": []},
        "candidate_root": tmp_path / "models" / "candidates",
        "protected_test_start_utc": "2022-01-02T00:00:00Z",
        "code_version": "synthetic-test-commit", "production_delay_minutes": 15,
    }
    options.update(overrides)
    return write_candidate_artifact(**options)


def test_rejected_candidate_never_changes_approved_model(tmp_path: Path) -> None:
    """Rejected outputs never replace the serving model."""
    from src.models.candidates import evaluate_candidate_artifact

    approved = tmp_path / "models" / "solarcast_f1_bundle.joblib"
    approved.parent.mkdir()
    approved.write_bytes(b"existing approved artifact")
    manifest_path = write_candidate(tmp_path, bundle=make_bundle(constant=0.0))
    assert evaluate_candidate_artifact(manifest_path)["eligible"] is False
    assert approved.read_bytes() == b"existing approved artifact"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "rejected"
    assert manifest["deployment_approved"] is False


def test_loaded_artifact_is_evaluated_without_refitting(tmp_path: Path, monkeypatch) -> None:
    """Predict with the serialized model, never a newly fitted substitute."""
    from src.models.candidates import evaluate_candidate_artifact

    manifest_path = write_candidate(tmp_path)
    monkeypatch.setattr(DummyRegressor, "fit", lambda *a, **k: pytest.fail("Unexpected refit"))
    result = evaluate_candidate_artifact(manifest_path)
    assert result["eligible"] is True
    assert result["candidate_metrics"]["daylight_mae_kwh"] == 0.0
    manifest = json.loads(manifest_path.read_text())
    bundle_path = manifest_path.parent / manifest["bundle_file"]
    assert result["bundle_sha256"] == hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    assert manifest["deployment_approved"] is False


@pytest.mark.parametrize("filename", ["bundle.joblib", "validation.parquet", "evaluation.json"])
def test_modified_artifact_is_rejected(tmp_path: Path, filename: str) -> None:
    """A previous passing record cannot validate changed files."""
    from src.models.candidates import CandidateArtifactError, evaluate_candidate_artifact

    manifest_path = write_candidate(tmp_path)
    artifact = manifest_path.parent / filename
    artifact.write_bytes(artifact.read_bytes() + b"changed")
    with pytest.raises(CandidateArtifactError, match="[Hh]ash"):
        evaluate_candidate_artifact(manifest_path)


def test_missing_artifact_is_rejected(tmp_path: Path) -> None:
    """Reject an incomplete candidate."""
    from src.models.candidates import CandidateArtifactError, evaluate_candidate_artifact

    manifest_path = write_candidate(tmp_path)
    (manifest_path.parent / "validation.parquet").unlink()
    with pytest.raises(CandidateArtifactError, match="Missing"):
        evaluate_candidate_artifact(manifest_path)


def test_cv_rejection_cannot_be_overridden(tmp_path: Path) -> None:
    """A good artifact check cannot override a failed cross-validation gate."""
    from src.models.candidates import evaluate_candidate_artifact

    manifest_path = write_candidate(
        tmp_path, cross_validation_gate={"eligible": False, "failure_reasons": ["CV failed"]}
    )
    assert evaluate_candidate_artifact(manifest_path)["eligible"] is False


def test_training_label_overlap_is_rejected(tmp_path: Path) -> None:
    """Training targets and their delivery delay must end before validation begins."""
    from src.models.candidates import CandidateArtifactError

    bundle = make_bundle()
    bundle.training_window["end_utc"] = "2022-01-01T02:00:00Z"
    with pytest.raises(CandidateArtifactError, match="training|Training"):
        write_candidate(tmp_path, bundle=bundle)


def test_final_holdout_rows_are_rejected(tmp_path: Path) -> None:
    """Candidate automation cannot use final test rows."""
    from src.models.candidates import CandidateArtifactError

    with pytest.raises(CandidateArtifactError, match="holdout|test"):
        write_candidate(tmp_path, protected_test_start_utc="2022-01-01T04:00:00Z")


def test_each_candidate_has_a_separate_directory(tmp_path: Path) -> None:
    """Repeated runs cannot overwrite earlier candidates."""
    first = write_candidate(tmp_path)
    original = first.read_bytes()
    second = write_candidate(tmp_path)
    assert first.parent != second.parent
    assert first.read_bytes() == original


@pytest.fixture
def pipeline_fixture(tmp_path: Path, monkeypatch) -> dict[str, Any]:
    """Exercise training control flow with synthetic data and mocked external services."""
    schema = Path("configs/feature_schema.json").read_bytes()
    monkeypatch.chdir(tmp_path)
    Path("configs").mkdir()
    Path("configs/feature_schema.json").write_bytes(schema)
    times = pd.date_range("2022-01-01", periods=128, freq="h", tz="UTC")
    pd.DataFrame({
        "interval_start_utc": times, "target_kwh": np.arange(128) * 2.0 + 10,
        "quality_flag": "VALID", "is_daylight": True,
    }).to_parquet("hourly.parquet")
    pd.DataFrame({"unused": [1]}).to_parquet("qh.parquet")
    Path("eval.yaml").write_text(yaml.safe_dump({
        "evaluation_window": {"start_utc": str(times[0]), "end_utc": str(times[-1])},
        "capacity_dc_kw": 893.0,
        "operational_timing": {"production_available_delay_minutes": 15},
    }))
    Path("params.yaml").write_text("random_seed: 42\n")

    def features(issue_times, feature_set="A", **kwargs):
        index = pd.DatetimeIndex(issue_times)
        cols = (
            train.get_feature_columns_a() if feature_set == "A" else train.get_feature_columns_b()
        )
        frame = pd.DataFrame(0.0, columns=cols, index=index)
        frame["power_kw_lag_0m"] = (index - times[0]).total_seconds() / 3600
        frame["solar_elevation_target_deg"] = 30.0
        if feature_set == "B":
            frame["weather_missing"] = 0
        return frame

    monkeypatch.setattr(train, "build_features", features)
    monkeypatch.setattr(train, "tune_models", lambda *a, **k: {
        "ridge": {"best_params": {}}, "hist_gradient_boosting": {"best_params": {}}
    })
    monkeypatch.setattr(train, "build_ridge_pipeline", lambda *a, **k: LinearRegression())
    monkeypatch.setattr(train, "build_hgb_model", lambda *a, **k: LinearRegression())
    monkeypatch.setattr(train, "ClearSky15mPersistence", lambda **k: SimpleNamespace(
        predict=lambda frame: frame["power_kw_lag_0m"].to_numpy() * 2 + 30
    ))
    monkeypatch.setattr(train, "generate_evaluation_plots", lambda **k: {})
    monkeypatch.setattr(train, "log_lineage", lambda *a, **k: {})
    for name in [
        "set_tracking_uri", "set_experiment", "log_params", "log_metrics", "set_tags",
        "log_artifact",
    ]:
        monkeypatch.setattr(train.mlflow, name, MagicMock())

    @contextlib.contextmanager
    def run(**kwargs):
        yield SimpleNamespace(info=SimpleNamespace(run_id="synthetic-run"))

    monkeypatch.setattr(train.mlflow, "start_run", run)
    registry = MagicMock()
    monkeypatch.setattr(train.mlflow.sklearn, "log_model", registry)
    protected = {
        Path("models/solarcast_f1_bundle.joblib"): b"approved model",
        Path("evaluation.json"): b"historical evaluation",
        Path("reports/evaluation.json"): b"historical evaluation report",
        Path("reports/final_test_audit.json"): b"historical test audit",
    }
    for path, content in protected.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return {"protected": protected, "registry": registry}


@pytest.mark.parametrize("passing", [True, False])
def test_training_preserves_champion_and_historical_reports(
    pipeline_fixture, monkeypatch, passing: bool,
) -> None:
    """Both gate outcomes must preserve the existing serving artifacts."""
    if not passing:
        monkeypatch.setattr(train, "build_hgb_model", lambda *a, **k: DummyRegressor(
            strategy="constant", constant=0.0
        ))
    result = train.run_training_pipeline(
        eval_config_path="eval.yaml", params_config_path="params.yaml",
        hourly_path="hourly.parquet", quarter_hourly_path="qh.parquet",
        weather_path="absent-weather.parquet", tracking_uri="unused-test-uri",
    )
    for path, original in pipeline_fixture["protected"].items():
        assert path.read_bytes() == original
    assert result["gate_evaluation"]["eligible"] is passing
    assert Path(result["bundle_path"]).parent != Path("models")
    from src.models.candidates import evaluate_candidate_artifact

    assert evaluate_candidate_artifact(result["candidate_manifest_path"])["eligible"] is passing
    assert pipeline_fixture["registry"].call_count == (1 if passing else 0)


@pytest.mark.parametrize("passing,expected", [(True, 0), (False, 2)])
def test_training_cli_exit_code(monkeypatch, passing: bool, expected: int) -> None:
    """Orchestration must see a nonzero exit for a rejected candidate."""
    monkeypatch.setattr("sys.argv", ["train"])
    monkeypatch.setattr(train, "run_training_pipeline", lambda **k: {
        "gate_evaluation": {"eligible": passing}
    })
    assert train.main() == expected


@pytest.mark.parametrize("passing,expected", [(True, 0), (False, 2)])
def test_evaluation_cli_checks_saved_artifact(
    tmp_path: Path, monkeypatch, passing: bool, expected: int,
) -> None:
    """The automation CLI must not enter the legacy refitting benchmark."""
    from src.models import evaluate

    manifest = write_candidate(tmp_path, bundle=make_bundle(40.0 if passing else 0.0))
    monkeypatch.setattr("sys.argv", ["evaluate", "--candidate-manifest", str(manifest)])
    monkeypatch.setattr(evaluate, "run_final_test_evaluation", lambda **k: pytest.fail(
        "Candidate gate invoked a research refit or final test"
    ))
    monkeypatch.setattr(DummyRegressor, "fit", lambda *a, **k: pytest.fail("Unexpected refit"))
    assert evaluate.main() == expected


def test_registration_failure_is_not_reported_as_success(pipeline_fixture) -> None:
    """A registry outage must fail the command while preserving approved artifacts."""
    pipeline_fixture["registry"].side_effect = RuntimeError("synthetic registry unavailable")
    with pytest.raises(RuntimeError, match="synthetic registry unavailable"):
        train.run_training_pipeline(
            eval_config_path="eval.yaml", params_config_path="params.yaml",
            hourly_path="hourly.parquet", quarter_hourly_path="qh.parquet",
            weather_path="absent-weather.parquet", tracking_uri="unused-test-uri",
        )
    for path, original in pipeline_fixture["protected"].items():
        assert path.read_bytes() == original
