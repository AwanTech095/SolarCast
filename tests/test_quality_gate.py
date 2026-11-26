"""Regression tests ensuring invalid evaluation evidence cannot pass the gate."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from src.models.evaluate import evaluate_candidate_gate


@pytest.fixture
def gate_inputs() -> dict[str, Any]:
    """Provide a passing candidate, positive baseline and valid predictions."""
    return {
        "candidate_metrics": {
            "daylight_mae_kwh": 40.0,
            "capacity_norm_daylight_mae": 40.0 / 893.0,
            "daylight_rmse_kwh": 60.0,
        },
        "baseline_metrics": {"daylight_mae_kwh": 50.0},
        "y_pred": np.array([0.0, 40.0, 100.0]),
    }


def assert_rejected(result: dict[str, Any]) -> None:
    """Require explicit rejection, a reason and portable strict-JSON output."""
    assert result["eligible"] is False
    assert result["failure_reasons"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("metric", ["daylight_mae_kwh", "capacity_norm_daylight_mae"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, -1.0, None, "invalid"])
def test_rejects_invalid_candidate_metrics(
    gate_inputs: dict[str, Any], metric: str, value: Any,
) -> None:
    """Unknown or invalid candidate errors must never count as acceptable errors."""
    gate_inputs["candidate_metrics"][metric] = value
    assert_rejected(evaluate_candidate_gate(**gate_inputs))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 0.0, -1.0, None, "invalid"])
def test_rejects_invalid_baseline(gate_inputs: dict[str, Any], value: Any) -> None:
    """A relative-improvement gate requires a finite, strictly positive baseline."""
    gate_inputs["baseline_metrics"]["daylight_mae_kwh"] = value
    assert_rejected(evaluate_candidate_gate(**gate_inputs))


@pytest.mark.parametrize(
    ("group", "metric"),
    [
        ("candidate_metrics", "daylight_mae_kwh"),
        ("candidate_metrics", "capacity_norm_daylight_mae"),
        ("baseline_metrics", "daylight_mae_kwh"),
    ],
)
def test_rejects_missing_required_metrics(
    gate_inputs: dict[str, Any], group: str, metric: str,
) -> None:
    """Missing metrics yield an auditable rejection instead of an accidental pass."""
    del gate_inputs[group][metric]
    assert_rejected(evaluate_candidate_gate(**gate_inputs))


def test_rejects_nonfinite_optional_metric(gate_inputs: dict[str, Any]) -> None:
    """An invalid supplied RMSE cannot hide behind otherwise passing MAE values."""
    gate_inputs["candidate_metrics"]["daylight_rmse_kwh"] = np.nan
    assert_rejected(evaluate_candidate_gate(**gate_inputs))


@pytest.mark.parametrize(
    "predictions",
    [[], 40.0, [[40.0]], [np.nan], [np.inf], [-5.0], [1000.0], ["invalid"]],
)
def test_rejects_invalid_prediction_evidence(
    gate_inputs: dict[str, Any], predictions: Any,
) -> None:
    """Require nonempty one-dimensional, finite predictions within physical limits."""
    gate_inputs["y_pred"] = predictions
    assert_rejected(evaluate_candidate_gate(**gate_inputs))


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("capacity_limit_kwh", np.nan),
        ("capacity_limit_kwh", np.inf),
        ("capacity_limit_kwh", 0.0),
        ("capacity_limit_kwh", -1.0),
        ("required_improvement_pct", np.nan),
        ("required_improvement_pct", np.inf),
        ("required_improvement_pct", -1.0),
        ("required_improvement_pct", 101.0),
        ("max_capacity_norm_mae", np.nan),
        ("max_capacity_norm_mae", np.inf),
        ("max_capacity_norm_mae", -1.0),
    ],
)
def test_invalid_gate_configuration_fails_explicitly(
    gate_inputs: dict[str, Any], parameter: str, value: float,
) -> None:
    """Invalid policy settings must fail the job rather than disable a gate check."""
    with pytest.raises(ValueError, match=parameter):
        evaluate_candidate_gate(**gate_inputs, **{parameter: value})


def test_valid_candidate_and_threshold_boundary_still_pass(gate_inputs: dict[str, Any]) -> None:
    """Preserve valid behavior, including exactly the required five-percent improvement."""
    gate_inputs["candidate_metrics"]["daylight_mae_kwh"] = 47.5
    gate_inputs["candidate_metrics"]["capacity_norm_daylight_mae"] = 47.5 / 893.0
    result = evaluate_candidate_gate(**gate_inputs)
    assert result["eligible"] is True
    assert result["failure_reasons"] == []
    assert result["achieved_improvement_pct"] == 5.0
    assert result["thresholds"]["max_daylight_mae_kwh"] == 47.5
    json.dumps(result, allow_nan=False)


def test_insufficient_improvement_still_rejected(gate_inputs: dict[str, Any]) -> None:
    """Reject just below the improvement threshold without rounding it into eligibility."""
    gate_inputs["candidate_metrics"]["daylight_mae_kwh"] = 47.5001
    result = evaluate_candidate_gate(**gate_inputs)
    assert_rejected(result)
    assert any("Insufficient daylight MAE improvement" in r for r in result["failure_reasons"])
