"""Isolated candidate artifacts with hash-bound, chronological evaluation evidence.

Creating or checking a candidate never modifies the approved serving model.
Eligibility is evidence for a later approval step, not deployment authorization.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from src.models.bundle import ModelBundle, load_bundle, save_bundle
from src.models.evaluate import compute_evaluation_metrics, evaluate_candidate_gate


class CandidateArtifactError(ValueError):
    """Candidate evidence is missing, changed or temporally inappropriate."""


def file_sha256(path: Path) -> str:
    """Hash exact file bytes for artifact identity."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _utc(value: Any) -> pd.Timestamp:
    """Require explicit timezone information instead of assuming local time."""
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise CandidateArtifactError("Evidence timestamps must be timezone-aware")
    return stamp.tz_convert("UTC")


def _validate_frame(
    bundle: ModelBundle, frame: pd.DataFrame, manifest: dict[str, Any],
) -> pd.DataFrame:
    """Check the validation window is after training and before the final holdout."""
    required = {
        "issue_time_utc", "target_kwh", "baseline_prediction_kwh", "evaluation_daylight",
        *bundle.feature_names,
    }
    missing = sorted(required - set(frame.columns))
    if missing or frame.empty:
        raise CandidateArtifactError(f"Missing validation evidence: {missing}")
    result = frame.copy()
    times = pd.DatetimeIndex([_utc(t) for t in result["issue_time_utc"]])
    if not times.is_monotonic_increasing or times.has_duplicates:
        raise CandidateArtifactError("Validation times must be increasing and unique")
    result["issue_time_utc"] = times
    train_end = _utc(bundle.training_window["end_utc"])
    delay = manifest["production_delay_minutes"]
    if not isinstance(delay, int) or delay < 0:
        raise CandidateArtifactError("Invalid training label availability delay")
    last_training_label_available = train_end + pd.Timedelta(hours=1, minutes=delay)
    if last_training_label_available > times.min():
        raise CandidateArtifactError("Validation overlaps training label availability")
    test_start = _utc(manifest["protected_test_start_utc"])
    if times.max() >= test_start:
        raise CandidateArtifactError("Candidate validation would access the final test holdout")
    mask = result["evaluation_daylight"]
    if mask.isna().any() or not pd.api.types.is_bool_dtype(mask):
        raise CandidateArtifactError("Validation daylight mask must contain booleans")
    return result


def _evaluate(
    bundle: ModelBundle, frame: pd.DataFrame, manifest: dict[str, Any],
) -> dict[str, Any]:
    """Score a loaded candidate without fitting or changing an estimator."""
    frame = _validate_frame(bundle, frame, manifest)
    target = frame["target_kwh"].to_numpy(dtype=float)
    mask = frame["evaluation_daylight"].to_numpy(dtype=bool)
    predictions = bundle.predict(frame[bundle.feature_names])
    capacity = bundle.capacity_limit_kwh
    candidate_metrics = compute_evaluation_metrics(target, predictions, mask, capacity)
    baseline_metrics = compute_evaluation_metrics(
        target, frame["baseline_prediction_kwh"].to_numpy(dtype=float), mask, capacity
    )
    policy = manifest["gate_policy"]
    gate = evaluate_candidate_gate(
        candidate_metrics, baseline_metrics, predictions, capacity_limit_kwh=capacity,
        required_improvement_pct=policy["required_improvement_pct"],
        max_capacity_norm_mae=policy["max_capacity_norm_mae"],
    )
    cv_gate = manifest["cross_validation_gate"]
    if cv_gate.get("eligible") is not True:
        gate["failure_reasons"].append("Cross-validation gate rejected this candidate")
        gate["eligible"] = False
    gate.update({
        "candidate_id": manifest["candidate_id"],
        "bundle_sha256": manifest["bundle_sha256"],
        "validation_sha256": manifest["validation_sha256"],
        "validation_scope": "development_fold_3",
        "validation_window": {
            "start_utc": frame["issue_time_utc"].min().isoformat(),
            "end_utc": frame["issue_time_utc"].max().isoformat(),
            "rows": len(frame),
        },
        "cross_validation_gate": cv_gate,
    })
    return gate


def write_candidate_artifact(
    bundle: ModelBundle,
    validation_frame: pd.DataFrame,
    cross_validation_gate: dict[str, Any],
    candidate_root: Path | str,
    protected_test_start_utc: str,
    code_version: str,
    production_delay_minutes: int = 15,
    config_hashes: dict[str, str] | None = None,
    required_improvement_pct: float = 5.0,
    max_capacity_norm_mae: float = 0.20,
) -> Path:
    """Save, reload and score a uniquely named candidate; never promote it.

    Validation is a fixed development snapshot, not final-test data. The manifest
    binds the actual model, validation inputs, evaluation, code and configuration.
    It is written last and is the completion marker for the candidate directory.
    """
    candidate_id = uuid.uuid4().hex
    manifest: dict[str, Any] = {
        "manifest_version": "1.0",
        "candidate_id": candidate_id,
        "created_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "code_version": code_version,
        "config_hashes": config_hashes or {},
        "data_hashes": bundle.data_hashes,
        "feature_schema_version": bundle.feature_schema_version,
        "feature_set": bundle.feature_set,
        "feature_names": bundle.feature_names,
        "training_window": bundle.training_window,
        "mlflow_run_id": bundle.mlflow_run_id,
        "protected_test_start_utc": _utc(protected_test_start_utc).isoformat(),
        "production_delay_minutes": production_delay_minutes,
        "deployment_approved": False,
        "cross_validation_gate": cross_validation_gate,
        "gate_policy": {
            "required_improvement_pct": required_improvement_pct,
            "max_capacity_norm_mae": max_capacity_norm_mae,
        },
        "bundle_file": "bundle.joblib",
        "validation_file": "validation.parquet",
        "evaluation_file": "evaluation.json",
    }
    frame = _validate_frame(bundle, validation_frame, manifest)
    root = Path(candidate_root) / candidate_id
    root.mkdir(parents=True, exist_ok=False)
    model_path = save_bundle(bundle, root / manifest["bundle_file"])
    validation_path = root / manifest["validation_file"]
    frame.to_parquet(validation_path, index=False)
    manifest["bundle_sha256"] = file_sha256(model_path)
    manifest["validation_sha256"] = file_sha256(validation_path)

    # Deliberately reload the bytes that will be distributed, not the in-memory model.
    gate = _evaluate(load_bundle(model_path), pd.read_parquet(validation_path), manifest)
    evaluation_path = root / manifest["evaluation_file"]
    evaluation_path.write_text(json.dumps(gate, indent=2, allow_nan=False), encoding="utf-8")
    manifest["evaluation_sha256"] = file_sha256(evaluation_path)
    manifest["status"] = "candidate" if gate["eligible"] else "rejected"
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
    )
    return manifest_path


def evaluate_candidate_artifact(manifest_path: Path | str) -> dict[str, Any]:
    """Recheck hashes and score the exact saved candidate without any refit.

    This is the automation gate. It does not call the legacy research benchmark,
    alter historical results, approve a deployment or access the final holdout.
    """
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("manifest_version") != "1.0"
        or manifest.get("deployment_approved") is not False
    ):
        raise CandidateArtifactError("Not an unapproved candidate manifest")
    artifacts = {}
    for kind in ("bundle", "validation", "evaluation"):
        artifact = path.parent / manifest[f"{kind}_file"]
        if artifact.resolve().parent != path.parent.resolve():
            raise CandidateArtifactError("Artifact path escapes candidate directory")
        if not artifact.is_file():
            raise CandidateArtifactError(f"Missing candidate artifact: {kind}")
        if file_sha256(artifact) != manifest[f"{kind}_sha256"]:
            raise CandidateArtifactError(f"Hash mismatch for candidate artifact: {kind}")
        artifacts[kind] = artifact
    bundle = load_bundle(artifacts["bundle"])
    if (
        bundle.feature_names != manifest["feature_names"]
        or bundle.feature_schema_version != manifest["feature_schema_version"]
        or bundle.feature_set != manifest["feature_set"]
        or bundle.mlflow_run_id != manifest["mlflow_run_id"]
        or bundle.training_window != manifest["training_window"]
        or bundle.data_hashes != manifest["data_hashes"]
    ):
        raise CandidateArtifactError("Model metadata does not match candidate manifest")
    result = _evaluate(bundle, pd.read_parquet(artifacts["validation"]), manifest)
    recorded = json.loads(artifacts["evaluation"].read_text(encoding="utf-8"))
    expected_status = "candidate" if result["eligible"] else "rejected"
    if result != recorded or manifest["status"] != expected_status:
        raise CandidateArtifactError("Recomputed evaluation does not match saved evidence")
    return result
