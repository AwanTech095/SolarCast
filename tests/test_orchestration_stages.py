"""Exercise the container stage boundaries without a Kubernetes cluster."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from orchestration.stages import (
    INPUTS,
    TARGETS,
    TRACKING_URI,
    restore_snapshot,
    train_candidate,
    verify_candidate,
)


def test_restore_verifies_before_exporting(tmp_path, monkeypatch):
    """Only the three trainer inputs are exported from the verified packet."""
    monkeypatch.chdir(tmp_path)
    source = Path("data/processed")
    source.mkdir(parents=True)
    for name in INPUTS:
        (source / name).write_bytes(name.encode())
    output = tmp_path / "snapshot"
    with (
        patch("orchestration.stages.subprocess.run") as pull,
        patch("orchestration.stages.verify_snapshot", side_effect=ValueError("bad receipt")),
        pytest.raises(ValueError, match="bad receipt"),
    ):
        restore_snapshot(output)
    pull.assert_called_once_with(["dvc", "pull", "-r", "team", *TARGETS], check=True)
    assert not output.exists()
    with (
        patch("orchestration.stages.subprocess.run"),
        patch("orchestration.stages.verify_snapshot"),
    ):
        restore_snapshot(output)
    assert sorted(p.name for p in output.iterdir()) == sorted(INPUTS)


def test_train_exports_exact_manifest_without_registering(tmp_path, monkeypatch):
    """An orchestration candidate stays unregistered regardless of its gate result."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", TRACKING_URI)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for name in INPUTS:
        (inputs / name).touch()
    output = tmp_path / "candidate"
    manifest = output / "models/candidates/one/manifest.json"
    with patch("src.models.train.run_training_pipeline", return_value={
        "candidate_manifest_path": str(manifest),
    }) as train:
        train_candidate(inputs, output)
    assert train.call_args.kwargs["register_candidate"] is False
    assert train.call_args.kwargs["hourly_path"] == inputs / INPUTS[0]
    assert (output / "candidate_path.txt").read_text() == "models/candidates/one/manifest.json"


@pytest.mark.parametrize("eligible", [True, False])
def test_gate_outputs_both_decisions(tmp_path, eligible):
    """Policy rejection is recorded as false instead of raising an infrastructure error."""
    manifest = tmp_path / "manifest.json"
    (tmp_path / "candidate_path.txt").write_text("manifest.json", encoding="utf-8")
    gate = tmp_path / "gate.json"
    decision = tmp_path / "eligibility"
    with patch("src.models.candidates.evaluate_candidate_artifact", return_value={
        "eligible": eligible,
    }) as evaluate:
        verify_candidate(tmp_path, gate, decision)
    evaluate.assert_called_once_with(manifest.resolve())
    assert json.loads(gate.read_text()) == {"eligible": eligible}
    assert json.loads(decision.read_text()) is eligible


def test_manifest_cannot_escape_artifact_directory(tmp_path):
    """Reject a manifest pointer outside the candidate artifact."""
    (tmp_path / "candidate_path.txt").write_text("../outside.json", encoding="utf-8")
    with pytest.raises(ValueError):
        verify_candidate(tmp_path, tmp_path / "gate", tmp_path / "decision")
