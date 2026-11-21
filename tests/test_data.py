"""Unit tests for raw generation data validation, quarantine, and hourly aggregation."""

import numpy as np
import pandas as pd
import pytest

from src.data.aggregation import aggregate_to_hourly, localize_and_convert_to_utc
from src.data.validation import validate_and_quarantine_raw_generation

_POWER_COL = "meter_revenue_grade_ac_output_meter_149578"


class TestDataValidationAndQuarantine:
    """Tests for Pandera validation and quarantine logic."""

    def test_above_893_kw_quarantined(self) -> None:
        """Verify that power readings above 893 kW are quarantined, not silently clipped."""
        df_raw = pd.DataFrame({
            "measured_on": ["2018-02-18 12:15:00", "2018-02-18 12:30:00", "2018-02-18 12:45:00"],
            _POWER_COL: [500.0, 1783.68, 520.0],
        })
        clean_df, quarantine_df = validate_and_quarantine_raw_generation(
            df_raw, power_col=_POWER_COL, max_capacity_kw=893.0
        )

        assert len(quarantine_df) == 1
        assert quarantine_df.iloc[0]["reason"] == "above_893_kw"
        assert quarantine_df.iloc[0]["power_kw"] == pytest.approx(1783.68)
        assert len(clean_df) == 2
        assert (clean_df[_POWER_COL] <= 893.0).all()
        # Verify no silent clipping
        assert 893.0 not in clean_df[_POWER_COL].values

    def test_negative_power_quarantined(self) -> None:
        """Verify that negative power values are quarantined."""
        df_raw = pd.DataFrame({
            "measured_on": ["2020-01-01 01:00:00", "2020-01-01 01:15:00"],
            _POWER_COL: [-10.5, 0.0],
        })
        clean_df, quarantine_df = validate_and_quarantine_raw_generation(
            df_raw, power_col=_POWER_COL
        )

        assert len(quarantine_df) == 1
        assert quarantine_df.iloc[0]["reason"] == "negative_power"
        assert quarantine_df.iloc[0]["power_kw"] == pytest.approx(-10.5)
        assert len(clean_df) == 1

    def test_non_finite_power_quarantined(self) -> None:
        """Verify that NaN and infinite values are quarantined."""
        df_raw = pd.DataFrame({
            "measured_on": [
                "2020-01-01 01:00:00",
                "2020-01-01 01:15:00",
                "2020-01-01 01:30:00",
            ],
            _POWER_COL: [100.0, np.nan, np.inf],
        })
        clean_df, quarantine_df = validate_and_quarantine_raw_generation(
            df_raw, power_col=_POWER_COL
        )

        assert len(quarantine_df) == 2
        assert (quarantine_df["reason"] == "non_finite").all()
        assert len(clean_df) == 1

    def test_duplicate_timestamps_quarantined(self) -> None:
        """Verify that duplicate timestamps are detected and quarantined."""
        df_raw = pd.DataFrame({
            "measured_on": [
                "2020-01-01 12:00:00",
                "2020-01-01 12:15:00",
                "2020-01-01 12:00:00",  # Duplicate of first row
            ],
            _POWER_COL: [250.0, 260.0, 255.0],
        })
        clean_df, quarantine_df = validate_and_quarantine_raw_generation(
            df_raw, power_col=_POWER_COL
        )

        assert len(quarantine_df) == 1
        assert quarantine_df.iloc[0]["reason"] == "duplicate_timestamp"
        assert quarantine_df.iloc[0]["measured_on"] == "2020-01-01 12:00:00"
        assert len(clean_df) == 2


class TestTimezoneAndDST:
    """Tests for timezone localization and Daylight Saving Time edge cases."""

    def test_spring_dst_gap_preserves_continuous_utc(self) -> None:
        """Verify that the 75-minute local clock jump in spring maps to continuous UTC."""
        # On 2022-03-13, clocks jump from 01:59:59 PST to 03:00:00 PDT.
        # Intervals 02:00..02:45 do not exist in local civil time.
        df_local = pd.DataFrame({
            "measured_on": ["2022-03-13 01:45:00", "2022-03-13 03:00:00"],
            _POWER_COL: [0.0, 0.0],
        })
        df_utc = localize_and_convert_to_utc(
            df_local, timestamp_col="measured_on", timezone="America/Los_Angeles"
        )

        # 01:45 PST is 09:45 UTC; 03:00 PDT is 10:00 UTC.
        # Difference in UTC must be exactly 15 minutes!
        t1 = df_utc.iloc[0]["interval_start_utc"]
        t2 = df_utc.iloc[1]["interval_start_utc"]
        assert (t2 - t1) == pd.Timedelta(minutes=15)
        assert str(t1) == "2022-03-13 09:45:00+00:00"
        assert str(t2) == "2022-03-13 10:00:00+00:00"

    def test_fall_dst_ambiguous_assigns_first_occurrence(self) -> None:
        """Verify that ambiguous fall-back times assign to first occurrence (PDT)."""
        # On 2022-11-06, 01:00..01:45 occurs once in the meter file.
        df_local = pd.DataFrame({
            "measured_on": [
                "2022-11-06 00:45:00",
                "2022-11-06 01:00:00",
                "2022-11-06 01:15:00",
                "2022-11-06 01:30:00",
                "2022-11-06 01:45:00",
                "2022-11-06 02:00:00",
            ],
            _POWER_COL: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        })
        df_utc = localize_and_convert_to_utc(
            df_local, timestamp_col="measured_on", timezone="America/Los_Angeles"
        )

        # 01:00 PDT is 08:00 UTC (first occurrence / daylight time)
        # 02:00 is PST (standard time), which is 10:00 UTC.
        # Thus 09:00 UTC is unrecorded and missing.
        assert str(df_utc.iloc[1]["interval_start_utc"]) == "2022-11-06 08:00:00+00:00"
        assert str(df_utc.iloc[4]["interval_start_utc"]) == "2022-11-06 08:45:00+00:00"
        assert str(df_utc.iloc[5]["interval_start_utc"]) == "2022-11-06 10:00:00+00:00"


class TestHourlyAggregation:
    """Tests for hourly energy aggregation and arithmetic."""

    def test_kw_to_kwh_arithmetic(self) -> None:
        """Verify that constant 400 kW for four 15-min intervals yields exactly 400 kWh."""
        # 4 intervals of 400 kW in hour 2023-06-21 18:00 UTC
        timestamps = pd.date_range("2023-06-21 18:00:00", periods=4, freq="15min", tz="UTC")
        clean_df = pd.DataFrame({
            "interval_start_utc": timestamps,
            "interval_end_utc": timestamps + pd.Timedelta(minutes=15),
            _POWER_COL: [400.0, 400.0, 400.0, 400.0],
        })

        hourly_df = aggregate_to_hourly(
            clean_df,
            power_col=_POWER_COL,
            min_intervals=4,
            start_time_utc=pd.Timestamp("2023-06-21 18:00:00", tz="UTC"),
            end_time_utc=pd.Timestamp("2023-06-21 18:00:00", tz="UTC"),
        )

        assert len(hourly_df) == 1
        assert hourly_df.iloc[0]["target_kwh"] == pytest.approx(400.0)
        assert hourly_df.iloc[0]["quality_flag"] == "VALID"
        assert hourly_df.iloc[0]["capacity_kw"] == 893.0
        assert hourly_df.iloc[0]["capacity_ac_observed_kw"] == 708.48

    def test_incomplete_hour_produces_nan_target(self) -> None:
        """Verify that an hour with fewer than min_intervals (e.g. 3) yields NaN."""
        # Only 3 intervals present in hour 2023-06-21 18:00 UTC
        timestamps = pd.date_range("2023-06-21 18:00:00", periods=3, freq="15min", tz="UTC")
        clean_df = pd.DataFrame({
            "interval_start_utc": timestamps,
            "interval_end_utc": timestamps + pd.Timedelta(minutes=15),
            _POWER_COL: [300.0, 300.0, 300.0],
        })

        hourly_df = aggregate_to_hourly(
            clean_df,
            power_col=_POWER_COL,
            min_intervals=4,
            start_time_utc=pd.Timestamp("2023-06-21 18:00:00", tz="UTC"),
            end_time_utc=pd.Timestamp("2023-06-21 18:00:00", tz="UTC"),
        )

        assert len(hourly_df) == 1
        assert pd.isna(hourly_df.iloc[0]["target_kwh"])
        assert hourly_df.iloc[0]["quality_flag"] == "INCOMPLETE_INTERVALS"

    def test_completely_missing_hour_produces_nan_target(self) -> None:
        """Verify that completely unrecorded hours produce NaN with MISSING_DATA flag."""
        # Provide intervals for 18:00 and 20:00; 19:00 is missing entirely
        t_18 = pd.date_range("2023-06-21 18:00:00", periods=4, freq="15min", tz="UTC")
        t_20 = pd.date_range("2023-06-21 20:00:00", periods=4, freq="15min", tz="UTC")
        all_t = t_18.append(t_20)
        clean_df = pd.DataFrame({
            "interval_start_utc": all_t,
            "interval_end_utc": all_t + pd.Timedelta(minutes=15),
            _POWER_COL: [200.0] * 8,
        })

        hourly_df = aggregate_to_hourly(
            clean_df,
            power_col=_POWER_COL,
            min_intervals=4,
            start_time_utc=pd.Timestamp("2023-06-21 18:00:00", tz="UTC"),
            end_time_utc=pd.Timestamp("2023-06-21 20:00:00", tz="UTC"),
        )

        assert len(hourly_df) == 3
        # Row 0: 18:00 (VALID)
        assert hourly_df.iloc[0]["quality_flag"] == "VALID"
        assert hourly_df.iloc[0]["target_kwh"] == pytest.approx(200.0)
        # Row 1: 19:00 (MISSING_DATA)
        assert hourly_df.iloc[1]["quality_flag"] == "MISSING_DATA"
        assert pd.isna(hourly_df.iloc[1]["target_kwh"])
        # Row 2: 20:00 (VALID)
        assert hourly_df.iloc[2]["quality_flag"] == "VALID"
        assert hourly_df.iloc[2]["target_kwh"] == pytest.approx(200.0)

    def test_solar_elevation_and_daylight_flag(self) -> None:
        """Verify solar elevation calculation and daylight boolean flag."""
        # Test summer midday (high daylight) and midnight (night)
        t_midday = pd.Timestamp("2023-06-21 20:00:00", tz="UTC")  # ~13:00 PDT (solar noon)
        t_midnight = pd.Timestamp("2023-06-21 07:00:00", tz="UTC")  # ~00:00 PDT

        timestamps = [
            t_midday,
            t_midday + pd.Timedelta(minutes=15),
            t_midday + pd.Timedelta(minutes=30),
            t_midday + pd.Timedelta(minutes=45),
            t_midnight,
            t_midnight + pd.Timedelta(minutes=15),
            t_midnight + pd.Timedelta(minutes=30),
            t_midnight + pd.Timedelta(minutes=45),
        ]
        clean_df = pd.DataFrame({
            "interval_start_utc": timestamps,
            "interval_end_utc": [t + pd.Timedelta(minutes=15) for t in timestamps],
            _POWER_COL: [500.0] * 4 + [0.0] * 4,
        })

        hourly_df = aggregate_to_hourly(
            clean_df,
            power_col=_POWER_COL,
            min_intervals=4,
            start_time_utc=t_midnight,
            end_time_utc=t_midday,
        )

        midday_row = hourly_df[hourly_df["interval_start_utc"] == t_midday].iloc[0]
        midnight_row = hourly_df[hourly_df["interval_start_utc"] == t_midnight].iloc[0]

        assert midday_row["solar_elevation_deg"] > 50.0  # Summer solar noon is ~74 deg
        assert bool(midday_row["is_daylight"]) is True
        assert bool(midday_row["high_sun"]) is True
        assert midnight_row["solar_elevation_deg"] < 0.0
        assert bool(midnight_row["is_daylight"]) is False
        assert bool(midnight_row["high_sun"]) is False

    def test_sunrise_hour_midpoint_elevation(self) -> None:
        """Verify that sunrise hour is classified as daylight using midpoint elevation."""
        # On 2023-03-21 at 14:00 UTC:
        # Elevation at start (14:00) is -2.89 deg (below horizon)
        # Elevation at midpoint (14:30) is +2.95 deg (daylight, but high_sun is False)
        t_sunrise = pd.Timestamp("2023-03-21 14:00:00", tz="UTC")
        timestamps = [
            t_sunrise,
            t_sunrise + pd.Timedelta(minutes=15),
            t_sunrise + pd.Timedelta(minutes=30),
            t_sunrise + pd.Timedelta(minutes=45),
        ]
        clean_df = pd.DataFrame({
            "interval_start_utc": timestamps,
            "interval_end_utc": [t + pd.Timedelta(minutes=15) for t in timestamps],
            _POWER_COL: [50.0, 100.0, 150.0, 200.0],
        })

        hourly_df = aggregate_to_hourly(
            clean_df,
            power_col=_POWER_COL,
            min_intervals=4,
            start_time_utc=t_sunrise,
            end_time_utc=t_sunrise,
        )

        row = hourly_df.iloc[0]
        assert row["solar_elevation_deg"] == pytest.approx(2.95, abs=0.2)
        assert bool(row["is_daylight"]) is True
        assert bool(row["high_sun"]) is False
