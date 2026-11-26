"""Regression coverage for malformed meter inputs."""

import pandas as pd
import pytest

from src.data.validation import validate_and_quarantine_raw_generation


def test_quarantine_keeps_invalid_text_and_recovers_valid_duplicate(tmp_path):
    path = tmp_path / "quarantine.parquet"
    frame = pd.DataFrame({
        "time": ["2023-06-01 12:00:00", "2023-06-01T12:00:00", "bad date",
                 "2023-06-01T12:15:00Z", "2023-06-01 12:00:00", None],
        "power": ["bad_sensor_value", "20", "30", "40", "50", "60"],
    })
    clean, rejected = validate_and_quarantine_raw_generation(frame, "time", "power", 893, path)
    assert clean["power"].tolist() == [20.0]
    assert clean["time"].tolist() == ["2023-06-01 12:00:00"]
    assert rejected["reason"].tolist() == [
        "non_finite", "invalid_timestamp", "invalid_timestamp",
        "duplicate_timestamp", "invalid_timestamp",
    ]
    assert pd.read_parquet(path)["raw_power"].iloc[0] == "bad_sensor_value"


def test_missing_meter_column_is_reported():
    with pytest.raises(ValueError, match="Missing required raw meter columns.*power"):
        validate_and_quarantine_raw_generation(pd.DataFrame({"time": []}), "time", "power")
