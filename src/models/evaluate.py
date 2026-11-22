"""Evaluation metrics, candidate gate verification, and final test safeguards for SolarCast.

Provides:
- `compute_evaluation_metrics`: Daylight MAE, RMSE, capacity-normalized MAE, all-hour MAE/RMSE.
- `compute_monthly_daylight_mae`: Per-month daylight MAE breakdown.
- `evaluate_candidate_gate`: Verifies the 5% improvement threshold, capacity bound,
  and finite non-negative constraints.
- `write_evaluation_json`: Persists gate decisions and metric comparisons.
- `check_final_test_safeguard`: Enforces strict test-set protection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_CAPACITY_KWH = 893.0
DEFAULT_EVAL_JSON_PATH = Path("reports/evaluation.json")
FINAL_TEST_LOG_PATH = Path("reports/final_test_audit.json")


def compute_evaluation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    is_daylight: np.ndarray,
    capacity_denominator_kwh: float = DEFAULT_CAPACITY_KWH,
) -> dict[str, float]:
    """Compute primary daylight MAE, all-hour MAE, RMSE, and normalized MAE.

    Args:
        y_true: Ground truth target values (kWh).
        y_pred: Predicted values (kWh).
        is_daylight: Boolean array indicating daylight hours (elevation > 0).
        capacity_denominator_kwh: Capacity scaling factor in kWh (default 893.0).

    Returns:
        Dictionary of calculated metric values.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    is_daylight = np.asarray(is_daylight, dtype=bool)

    # Daylight metrics
    y_true_day = y_true[is_daylight]
    y_pred_day = y_pred[is_daylight]

    if len(y_true_day) > 0:
        daylight_mae = float(np.mean(np.abs(y_true_day - y_pred_day)))
        daylight_rmse = float(np.sqrt(np.mean((y_true_day - y_pred_day) ** 2)))
        norm_daylight_mae = float(daylight_mae / capacity_denominator_kwh)
    else:
        daylight_mae = np.nan
        daylight_rmse = np.nan
        norm_daylight_mae = np.nan

    # All-hour metrics
    all_hour_mae = float(np.mean(np.abs(y_true - y_pred)))
    all_hour_rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    return {
        "daylight_mae_kwh": daylight_mae,
        "daylight_rmse_kwh": daylight_rmse,
        "capacity_norm_daylight_mae": norm_daylight_mae,
        "all_hour_mae_kwh": all_hour_mae,
        "all_hour_rmse_kwh": all_hour_rmse,
    }


def compute_monthly_daylight_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: pd.DatetimeIndex,
    is_daylight: np.ndarray,
) -> pd.DataFrame:
    """Compute daylight MAE broken down by calendar month (YYYY-MM)."""
    df = pd.DataFrame({
        "month": pd.DatetimeIndex(timestamps).strftime("%Y-%m"),
        "y_true": np.asarray(y_true, dtype=float),
        "y_pred": np.asarray(y_pred, dtype=float),
        "is_daylight": np.asarray(is_daylight, dtype=bool),
    })
    day_df = df[df["is_daylight"]].copy()
    day_df["abs_err"] = (day_df["y_true"] - day_df["y_pred"]).abs()

    monthly = day_df.groupby("month")["abs_err"].agg(
        daylight_hours="count",
        daylight_mae="mean",
    ).reset_index()
    return monthly


def evaluate_candidate_gate(
    candidate_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    y_pred: np.ndarray,
    capacity_limit_kwh: float = DEFAULT_CAPACITY_KWH,
    required_improvement_pct: float = 5.0,
    max_capacity_norm_mae: float = 0.20,
) -> dict[str, Any]:
    """Evaluate whether candidate model satisfies the acceptance gate.

    Rules from configs/evaluation.yaml:
    1. At least 5% reduction in Daylight MAE relative to baseline champion.
    2. Capacity-normalized daylight MAE <= 20.0% (0.20).
    3. Predictions must be strictly finite, non-negative, and <= plant DC capacity.
    """
    baseline_mae = float(baseline_metrics["daylight_mae_kwh"])
    cand_mae = float(candidate_metrics["daylight_mae_kwh"])
    cand_norm_mae = float(candidate_metrics["capacity_norm_daylight_mae"])

    max_allowed_mae = baseline_mae * (1.0 - (required_improvement_pct / 100.0))
    rel_improvement = ((baseline_mae - cand_mae) / baseline_mae) * 100.0

    failure_reasons: list[str] = []

    # Check 1: 5% Daylight MAE improvement
    if cand_mae > max_allowed_mae:
        failure_reasons.append(
            f"Insufficient daylight MAE improvement: achieved {rel_improvement:.2f}% "
            f"(target >= {required_improvement_pct:.1f}%; "
            f"candidate {cand_mae:.2f} kWh > max allowed {max_allowed_mae:.2f} kWh)"
        )

    # Check 2: Capacity normalized bound
    if cand_norm_mae > max_capacity_norm_mae:
        failure_reasons.append(
            f"Capacity-normalized daylight MAE exceeded: {cand_norm_mae * 100:.2f}% "
            f"> max allowed {max_capacity_norm_mae * 100:.1f}%"
        )

    # Check 3: Finite and physical output bounds
    preds = np.asarray(y_pred, dtype=float)
    if not np.all(np.isfinite(preds)):
        failure_reasons.append("Predictions contain NaN or infinite values")
    if np.any(preds < -1e-5):
        min_val = float(np.min(preds))
        failure_reasons.append(f"Predictions contain negative values (min: {min_val:.4f} kWh)")
    if np.any(preds > (capacity_limit_kwh + 1e-3)):
        max_val = float(np.max(preds))
        failure_reasons.append(
            f"Predictions exceed capacity limit {capacity_limit_kwh:.1f} kWh "
            f"(max: {max_val:.2f} kWh)"
        )

    eligible = len(failure_reasons) == 0

    return {
        "eligible": eligible,
        "required_improvement_pct": required_improvement_pct,
        "achieved_improvement_pct": float(np.round(rel_improvement, 2)),
        "thresholds": {
            "max_daylight_mae_kwh": float(np.round(max_allowed_mae, 2)),
            "max_capacity_norm_daylight_mae": max_capacity_norm_mae,
            "capacity_limit_kwh": capacity_limit_kwh,
        },
        "baseline_metrics": {k: float(np.round(v, 4)) for k, v in baseline_metrics.items()},
        "candidate_metrics": {k: float(np.round(v, 4)) for k, v in candidate_metrics.items()},
        "failure_reasons": failure_reasons,
        "final_test_opened": False,
    }


def write_evaluation_json(
    gate_result: dict[str, Any],
    output_path: Path | str = DEFAULT_EVAL_JSON_PATH,
) -> Path:
    """Write gate evaluation results to evaluation.json."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(gate_result, f, indent=2)
    return path


def check_final_test_safeguard(
    force: bool = False,
    reason: str | None = None,
    log_path: Path | str = FINAL_TEST_LOG_PATH,
) -> None:
    """Enforce safeguard preventing unauthorized opening of the final test set.

    Raises RuntimeError if the final test set was already opened unless --force
    is explicitly supplied with a non-empty documented reason.
    """
    path = Path(log_path)
    if path.exists():
        if not force:
            raise RuntimeError(
                "PROTECTED TEST SET VIOLATION: Final 20% test partition has already been opened! "
                f"See {path}. Re-evaluating requires passing --force and providing a reason."
            )
        if not reason or len(reason.strip()) < 10:
            raise ValueError(
                "FORCED OPENING REJECTED: A meaningful reason (>= 10 chars) must be logged "
                "when using --force to recompute the final test set."
            )
