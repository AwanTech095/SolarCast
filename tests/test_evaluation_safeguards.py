"""Regression tests for implicit holdout access and immutable audit claims."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.models import deviation, evaluate


@pytest.mark.parametrize(
    "entry", [deviation.get_test_period_dataset, deviation.run_natural_suspect_zero_evaluation]
)
def test_retired_deviation_readers_fail_before_io(monkeypatch, entry) -> None:
    """Calling historical convenience readers must not silently access final labels."""
    monkeypatch.setattr(deviation.pd, "read_parquet", lambda *a, **k: pytest.fail("Holdout I/O"))
    with pytest.raises(RuntimeError, match="retired|PROTECTED|historical"):
        entry()


def test_deviation_no_action_does_no_work(monkeypatch) -> None:
    """A bare CLI command must never calibrate or benchmark implicitly."""
    monkeypatch.setattr("sys.argv", ["deviation"])
    monkeypatch.setattr(
        deviation,
        "compute_validation_residuals",
        lambda *a, **k: pytest.fail("Implicit calibration"),
    )
    with pytest.raises(SystemExit) as error:
        deviation.main()
    assert error.value.code == 2


def test_opened_holdout_cannot_be_forced_again(tmp_path: Path) -> None:
    """A valid reason does not make an already-used test set new again."""
    marker = tmp_path / "audit.json"
    marker.write_bytes(b'{"final_test_opened": true}')
    before = marker.read_bytes()
    with pytest.raises(RuntimeError, match="already been opened"):
        evaluate.check_final_test_safeguard(True, "repeat existing test for CI", marker)
    assert marker.read_bytes() == before


def test_first_open_requires_explicit_force_and_reason(tmp_path: Path, monkeypatch) -> None:
    """A fresh filesystem is not implicit authorization to open test data."""
    for key in ("CI", "GITHUB_ACTIONS", "AIRFLOW_CTX_DAG_ID", "KFP_RUN_ID"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(RuntimeError, match="force|authorization"):
        evaluate.check_final_test_safeguard(log_path=tmp_path / "audit.json")


def test_audit_cannot_overwrite_existing_evidence(tmp_path: Path) -> None:
    """Exclusive audit creation protects evidence even if preflight races."""
    marker = tmp_path / "audit.json"
    marker.write_bytes(b"old historical evidence")
    with pytest.raises((RuntimeError, FileExistsError)):
        evaluate.log_final_test_audit("repeat test must never overwrite", marker)
    assert marker.read_bytes() == b"old historical evidence"


def test_final_runner_checks_guard_before_any_data_read(monkeypatch) -> None:
    """Denied final-test calls must fail before touching configs, targets or features."""
    monkeypatch.setattr(
        evaluate,
        "check_final_test_safeguard",
        lambda **k: (_ for _ in ()).throw(RuntimeError("synthetic guard denied")),
    )
    monkeypatch.setattr(
        evaluate.pd, "read_parquet", lambda *a, **k: pytest.fail("Read before guard")
    )
    with pytest.raises(RuntimeError, match="synthetic guard denied"):
        evaluate.run_final_test_evaluation(
            "final_test", True, "test guard ordering", eval_config_path="does-not-exist.yaml"
        )


def test_ci_cannot_open_even_unused_holdout(tmp_path: Path, monkeypatch) -> None:
    """Scheduler contexts cannot acquire a first-use audit claim."""
    monkeypatch.setenv("CI", "true")
    with pytest.raises(RuntimeError, match="automation|CI"):
        evaluate.check_final_test_safeguard(True, "synthetic first use", tmp_path / "audit.json")


def test_packet_instructions_do_not_reopen_or_regenerate_historical_evidence() -> None:
    """An artifact handoff must not instruct the recipient to rerun the final test."""
    tree = ast.parse(Path("scripts/make_artifact_packet.py").read_text(encoding="utf-8"))
    mapping = next(ast.literal_eval(node.value) for node in tree.body
                   if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "PIPELINE_COMMANDS"
                           for t in node.targets))
    assert mapping["reports/final_test_audit.json"] is None
    assert mapping["models/solarcast_f1_bundle.joblib"] is None
    assert not any("--final-test" in command for command in mapping.values() if command)


def test_changing_working_directory_does_not_lose_historical_audit(tmp_path, monkeypatch) -> None:
    """Explicit absolute dataset paths cannot bypass a cwd-relative audit marker."""
    assert evaluate.FINAL_TEST_LOG_PATH.is_absolute()
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="already been opened"):
        evaluate.run_final_test_evaluation("final_test", True, "attempt from another directory")
