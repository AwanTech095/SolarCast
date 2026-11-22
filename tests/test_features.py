"""Unit tests for feature engineering, cutoff integrity, and leakage prevention.

Verifies:
1. No feature uses information at or after the cutoff (perturbation test).
2. Continuity and graceful behavior across DST fall-back transition.
3. Behavior when production history is missing or invalid (missing hour test).
4. Edge cases during the first week of the historical archive (pre-history test).
5. Exact adherence to configs/feature_schema.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.preprocessing import build_features

SCHEMA_PATH = Path("configs/feature_schema.json")
HOURLY_PATH = Path("data/processed/hourly.parquet")
QH_PATH = Path("data/processed/quarter_hourly.parquet")


@pytest.fixture(scope="module")
def hourly_df() -> pd.DataFrame:
    """Load cached hourly dataset."""
    return pd.read_parquet(HOURLY_PATH)


@pytest.fixture(scope="module")
def quarter_hourly_df() -> pd.DataFrame:
    """Load cached quarter-hourly dataset."""
    return pd.read_parquet(QH_PATH)


@pytest.fixture(scope="module")
def schema_features() -> list[str]:
    """Load ordered feature names from JSON schema."""
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return [feat["name"] for feat in data["features"]]


class TestFeatureIntegrityAndLeakage:
    """Tests proving strict past-only information boundaries and zero future leakage."""

    def test_schema_conformance(
        self,
        hourly_df: pd.DataFrame,
        quarter_hourly_df: pd.DataFrame,
        schema_features: list[str],
    ) -> pd.DataFrame:
        """Verify build_features produces columns matching schema exactly in order and count."""
        test_times = pd.date_range("2022-06-13 12:00:00", periods=3, freq="h", tz="UTC")
        feats = build_features(test_times, hourly_df=hourly_df, quarter_hourly_df=quarter_hourly_df)

        assert len(feats.columns) == 23
        assert list(feats.columns) == schema_features
        assert feats.index.name == "issue_time_utc"
        assert len(feats) == 3

    def test_no_leakage_after_cutoff_via_perturbation(
        self,
        hourly_df: pd.DataFrame,
        quarter_hourly_df: pd.DataFrame,
    ) -> None:
        """Prove that perturbing all data after the cutoff does not change features.

        For issue time H = 2022-06-13 14:00:00 UTC (daytime generation peak):
        - Cutoff = H - 15m = 2022-06-13 13:45:00 UTC.
        - Any data at or after 13:45:00 UTC is strictly future information.
        - We perturb all values in hourly_df and quarter_hourly_df at or after 13:45:00 UTC.
        - The resulting features at H must remain bit-for-bit identical.
        """
        issue_time = pd.Timestamp("2022-06-13 14:00:00", tz="UTC")
        cutoff_time = issue_time - pd.Timedelta(minutes=15)  # 13:45:00 UTC

        # Compute baseline unperturbed features
        base_feats = build_features(
            [issue_time], hourly_df=hourly_df, quarter_hourly_df=quarter_hourly_df
        )

        # Create perturbed copies
        h_perturbed = hourly_df.copy()
        qh_perturbed = quarter_hourly_df.copy()

        # Perturb all hourly rows with interval_start_utc >= cutoff_time - 1h (i.e. >= 13:00)
        # Note: hour 13:00-14:00 ends at 14:00 > 13:45, so it must not be used at all!
        h_mask = h_perturbed["interval_start_utc"] >= (cutoff_time - pd.Timedelta(hours=1))
        h_perturbed.loc[h_mask, "target_kwh"] = 99999.0
        h_perturbed.loc[h_mask, "quality_flag"] = "PERTURBED_FUTURE"

        # Perturb all 15-minute rows with interval_start_utc >= cutoff_time - 15m (ends at 13:45)
        # The interval ending at 13:45 starts at 13:30.
        # So intervals starting at 13:45, 14:00, etc., are strictly after cutoff.
        qh_mask = qh_perturbed["interval_start_utc"] >= cutoff_time
        qh_perturbed.loc[qh_mask, "power_kw"] = 88888.0
        qh_perturbed.loc[qh_mask, "quality_flag"] = "PERTURBED_FUTURE"

        # Compute features with perturbed future data
        perturbed_feats = build_features(
            [issue_time], hourly_df=h_perturbed, quarter_hourly_df=qh_perturbed
        )

        # Baseline and perturbed features must be identical
        pd.testing.assert_frame_equal(base_feats, perturbed_feats)

    def test_dst_fallback_day(
        self,
        hourly_df: pd.DataFrame,
        quarter_hourly_df: pd.DataFrame,
    ) -> None:
        """Verify feature extraction across autumn Daylight Saving Time fall-back day.

        On 2022-11-06, clocks fall back from PDT to PST at 02:00 local time (09:00 UTC).
        """
        fallback_times = pd.date_range(
            "2022-11-06 00:00:00", "2022-11-06 23:00:00", freq="h", tz="UTC"
        )
        feats = build_features(
            fallback_times, hourly_df=hourly_df, quarter_hourly_df=quarter_hourly_df
        )

        assert len(feats) == 24
        # Verify no NaN in calendar, solar, or age features
        assert not feats["hour_sin"].isna().any()
        assert not feats["solar_elevation_target_deg"].isna().any()
        assert not feats["production_age_minutes"].isna().any()

    def test_missing_hour_handling(
        self,
        hourly_df: pd.DataFrame,
        quarter_hourly_df: pd.DataFrame,
    ) -> None:
        """Verify feature extraction when preceding hourly production data is missing."""
        # Find an issue time where the latest available hour (H - 2h) is MISSING_DATA
        missing_hourly_starts = hourly_df[
            hourly_df["quality_flag"] == "MISSING_DATA"
        ]["interval_start_utc"]
        assert len(missing_hourly_starts) > 0

        # Choose the first missing hour and set issue_time = missing_hour + 2h
        missing_h = missing_hourly_starts.iloc[0]
        issue_time = missing_h + pd.Timedelta(hours=2)

        feats = build_features(
            [issue_time], hourly_df=hourly_df, quarter_hourly_df=quarter_hourly_df
        )

        # latest_hour_missing must be flagged as 1
        assert feats.iloc[0]["latest_hour_missing"] == 1
        assert np.isnan(feats.iloc[0]["energy_kwh_latest_avail_hour"])
        # Clear-sky index must gracefully fall back to 1.0 (no division by zero / nan)
        assert feats.iloc[0]["clearsky_index_latest_avail"] == 1.0

    def test_first_week_prehistory(
        self,
        hourly_df: pd.DataFrame,
        quarter_hourly_df: pd.DataFrame,
    ) -> None:
        """Verify feature extraction during the first week where H - 168h is unavailable."""
        # Hour 2017-01-01 09:00:00 is the first VALID hour in the dataset.
        # Choose issue time 24 hours later: 2017-01-02 09:00:00 UTC.
        issue_time = pd.Timestamp("2017-01-02 09:00:00", tz="UTC")

        feats = build_features(
            [issue_time], hourly_df=hourly_df, quarter_hourly_df=quarter_hourly_df
        )

        assert len(feats) == 1
        # prev_week_missing must be 1 because H - 168h is prior to dataset start
        assert feats.iloc[0]["prev_week_missing"] == 1
        assert np.isnan(feats.iloc[0]["energy_kwh_prev_week_same_hour"])
        # prev_day_missing is 0 (2017-01-01 09:00:00 is VALID)
        assert feats.iloc[0]["prev_day_missing"] == 0
        assert not np.isnan(feats.iloc[0]["energy_kwh_prev_day_same_hour"])
