"""Deterministic continuity, timing and incremental alert contract regressions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from jsonschema import Draft202012Validator

from src.models import deviation

START = pd.Timestamp("2023-06-01T16:00:00Z")
CONFIG = {"bounds": {"clear": {"bound_low": -100.0, "bound_high": 100.0}}}


def rows(hours: list[float], **columns: Any) -> pd.DataFrame:
    """Synthetic target-hour labels paired with stored forecasts."""
    frame = pd.DataFrame({
        "timestamp": [START + pd.Timedelta(hours=h) for h in hours],
        "actual": 0.0, "forecast": 500.0, "is_daylight": True,
        "solar_elevation_deg": 30.0, "quality_flag": "VALID", "forecast_cloud_bin": "clear",
    })
    for key, value in columns.items():
        frame[key] = value
    return frame


@pytest.mark.parametrize("hours", [[0, 3], [0, 0], [0, 0, 1], [0, 0.5], [0, 1.000001]])
def test_nonconsecutive_or_ambiguous_hours_cannot_alert(hours: list[float]) -> None:
    """Gaps, duplicates and subhourly rows must not manufacture persistence."""
    assert deviation.detect_deviations(rows(hours), CONFIG) == []


def test_gap_splits_two_real_episodes() -> None:
    """Each side of a gap needs its own adjacent pair."""
    alerts = deviation.detect_deviations(rows([0, 1, 5, 6]), CONFIG)
    assert [a["consecutive_hours"] for a in alerts] == [2, 2]


def test_duplicate_hour_resets_both_sides() -> None:
    """Conflicting duplicate labels are suppressed rather than arbitrarily deduplicated."""
    assert deviation.detect_deviations(rows([0, 1, 1, 2]), CONFIG) == []


@pytest.mark.parametrize("column,value", [
    ("actual", float("inf")), ("forecast", float("inf")),
    ("solar_elevation_deg", float("nan")), ("is_daylight", None),
])
def test_invalid_measurements_break_persistence(column: str, value: Any) -> None:
    """Nonfinite or unknown eligibility values cannot count as breach hours."""
    frame = rows([0, 1, 2])
    frame[column] = frame[column].astype(object)
    frame.loc[1, column] = value
    assert deviation.detect_deviations(frame, CONFIG) == []


def test_event_id_is_stable_with_unrelated_earlier_episode() -> None:
    """A replay's batch position must not alter the identity of the same episode."""
    alone = deviation.detect_deviations(rows([5, 6]), CONFIG)[0]
    together = deviation.detect_deviations(rows([0, 1, 5, 6]), CONFIG)[-1]
    assert alone["event_id"] == together["event_id"]


def test_second_label_timing_is_not_target_start() -> None:
    """Use real label availability, not the second target interval's start, for trigger time."""
    frame = rows([0, 1, 2])
    frame["actual_available_at_utc"] = frame["timestamp"] + pd.Timedelta(hours=1, minutes=15)
    event = deviation.detect_deviations(frame, CONFIG)[0]
    assert pd.Timestamp(event["confirmation_issue_time_utc"]) == START + pd.Timedelta(hours=1)
    assert pd.Timestamp(event["triggered_at_utc"]) == START + pd.Timedelta(hours=2, minutes=15)
    assert pd.Timestamp(event["target_interval_end_utc"]) == START + pd.Timedelta(hours=2)
    assert event["detection_delay_hours"] == 2.25


def test_missing_availability_is_not_invented() -> None:
    """Old batch inputs cannot establish a real label-delivery time."""
    event = deviation.detect_deviations(rows([0, 1]), CONFIG)[0]
    assert event["triggered_at_utc"] is None
    assert event["actual_available_at_utc"] is None
    assert event["detection_delay_hours"] is None


def test_incremental_emission_and_one_event_per_episode() -> None:
    """Retain state across calls and emit once at the second breach, without waiting for closure."""
    detector = deviation.DeviationDetector(CONFIG)
    assert detector.process(rows([0])) == []
    event = detector.process(rows([1]))
    assert len(event) == 1
    assert event[0]["consecutive_hours"] == 2
    assert detector.process(rows([2, 3])) == []
    assert event[0]["consecutive_hours"] == 2  # emitted snapshot stays immutable


def test_incremental_gap_resets() -> None:
    """An omitted hour between batches also resets the pending breach."""
    detector = deviation.DeviationDetector(CONFIG)
    assert detector.process(rows([0])) == []
    assert detector.process(rows([3])) == []
    assert len(detector.process(rows([4]))) == 1


def test_overlapping_batch_is_rejected_without_corrupting_state() -> None:
    """A late replay cannot count a previously processed interval twice."""
    detector = deviation.DeviationDetector(CONFIG)
    detector.process(rows([0]))
    with pytest.raises(ValueError, match="overlap|ordered|processed"):
        detector.process(rows([0, 1]))
    assert len(detector.process(rows([1]))) == 1


def test_timestamp_requires_explicit_timezone() -> None:
    """No silent local-time conversion in an alert's UTC identity."""
    frame = rows([0, 1])
    frame["timestamp"] = frame["timestamp"].dt.tz_localize(None)
    with pytest.raises(ValueError, match="timezone|UTC"):
        deviation.detect_deviations(frame, CONFIG)


def test_alert_validates_against_full_shared_schema() -> None:
    """New optional timing fields remain valid alongside the other alert types."""
    schema = json.loads(Path("configs/alert_event_schema.json").read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema, format_checker=Draft202012Validator.FORMAT_CHECKER)
    frame = rows([0, 1])
    for with_availability in (False, True):
        if with_availability:
            frame["actual_available_at_utc"] = (
                frame["timestamp"] + pd.Timedelta(hours=1, minutes=15)
            )
        validator.validate(deviation.detect_deviations(frame, CONFIG)[0])


def test_explicit_availability_before_interval_end_is_rejected() -> None:
    """Delivery timestamps cannot claim knowledge before the hourly target completes."""
    frame = rows([0, 1])
    frame["actual_available_at_utc"] = frame["timestamp"]
    with pytest.raises(ValueError, match="interval ends"):
        deviation.detect_deviations(frame, CONFIG)


def test_direction_change_requires_a_new_pair() -> None:
    """Opposite breaches must not combine into a single confirmed episode."""
    assert deviation.detect_deviations(rows([0, 1], actual=[0.0, 1000.0]), CONFIG) == []


def test_event_identity_includes_model_version() -> None:
    """Two releases must not accidentally share an alert identity."""
    first = deviation.detect_deviations(rows([0, 1]), CONFIG, model_version="v4")[0]
    second = deviation.detect_deviations(rows([0, 1]), CONFIG, model_version="v5")[0]
    assert first["event_id"] != second["event_id"]
