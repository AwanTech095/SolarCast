"""Unit tests for raw generation data validation, quarantine, and hourly aggregation."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.data.aggregation import (
    aggregate_to_hourly,
    aggregate_to_quarter_hourly,
    localize_and_convert_to_utc,
)
from src.data.hrrr import (
    HRRR_FEATURE_COLUMNS,
    HRRRExtractorConfig,
    extract_hour_weather,
    find_eligible_run,
    get_thread_fs,
    load_coverage_scan_set,
    merge_parts,
)
from src.data.validation import (
    quarter_hourly_schema,
    validate_and_quarantine_raw_generation,
)

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

    def test_suspect_zero_flag_on_consecutive_high_sun_zeros(self) -> None:
        """Verify that quality_flag=SUSPECT_ZERO is assigned only to runs of 2+ high-sun zeros."""
        # 3 consecutive midday hours on 2023-06-21 (solar noon ~20:00 UTC, high_sun > 5 deg)
        # Hour 18:00 (zero), Hour 19:00 (zero) -> run of 2, both SUSPECT_ZERO, target=0 unchanged
        # Hour 20:00 (positive generation) -> VALID
        t_base = pd.Timestamp("2023-06-21 18:00:00", tz="UTC")
        t_18 = pd.date_range(t_base, periods=4, freq="15min", tz="UTC")
        t_19 = pd.date_range(t_base + pd.Timedelta(hours=1), periods=4, freq="15min", tz="UTC")
        t_20 = pd.date_range(t_base + pd.Timedelta(hours=2), periods=4, freq="15min", tz="UTC")

        # Also add an isolated single high-sun zero hour on the next day at 18:00
        t_next = pd.Timestamp("2023-06-22 18:00:00", tz="UTC")
        t_next_18 = pd.date_range(t_next, periods=4, freq="15min", tz="UTC")
        t_next_19 = pd.date_range(t_next + pd.Timedelta(hours=1), periods=4, freq="15min", tz="UTC")

        all_t = t_18.append(t_19).append(t_20).append(t_next_18).append(t_next_19)
        powers = [0.0] * 4 + [0.0] * 4 + [300.0] * 4 + [0.0] * 4 + [250.0] * 4

        clean_df = pd.DataFrame({
            "interval_start_utc": all_t,
            "interval_end_utc": all_t + pd.Timedelta(minutes=15),
            _POWER_COL: powers,
        })

        hourly_df = aggregate_to_hourly(
            clean_df,
            power_col=_POWER_COL,
            min_intervals=4,
            start_time_utc=t_base,
            end_time_utc=t_next + pd.Timedelta(hours=1),
        )

        h18 = hourly_df[hourly_df["interval_start_utc"] == t_base].iloc[0]
        h19 = hourly_df[hourly_df["interval_start_utc"] == t_base + pd.Timedelta(hours=1)].iloc[0]
        h20 = hourly_df[hourly_df["interval_start_utc"] == t_base + pd.Timedelta(hours=2)].iloc[0]
        h_iso = hourly_df[hourly_df["interval_start_utc"] == t_next].iloc[0]

        # Consecutive run of 2: both flagged SUSPECT_ZERO, targets remain exactly 0.0
        assert h18["quality_flag"] == "SUSPECT_ZERO"
        assert h18["target_kwh"] == 0.0
        assert h19["quality_flag"] == "SUSPECT_ZERO"
        assert h19["target_kwh"] == 0.0

        # Non-zero generation: VALID
        assert h20["quality_flag"] == "VALID"
        assert h20["target_kwh"] == pytest.approx(300.0)

        # Isolated single zero hour: remains VALID (not in a run of 2+)
        assert h_iso["quality_flag"] == "VALID"
        assert h_iso["target_kwh"] == 0.0


class TestQuarterHourlyDataset:
    """Tests for continuous 15-minute dataset aggregation and schema."""

    def test_aggregate_to_quarter_hourly_synthetic(self) -> None:
        """Verify continuous 15-minute grid generation and quality flags."""
        t_base = pd.Timestamp("2022-06-13 12:00:00", tz="UTC")
        times = pd.date_range(t_base, periods=4, freq="15min", tz="UTC")
        clean_df = pd.DataFrame({
            "interval_start_utc": [times[0], times[1], times[3]],  # interval 2 missing
            "interval_end_utc": [times[0] + pd.Timedelta(minutes=15),
                                 times[1] + pd.Timedelta(minutes=15),
                                 times[3] + pd.Timedelta(minutes=15)],
            _POWER_COL: [100.0, 200.0, 400.0],
        })

        qh_df = aggregate_to_quarter_hourly(
            clean_df,
            power_col=_POWER_COL,
            start_time_utc=t_base,
            end_time_utc=t_base + pd.Timedelta(minutes=45),
        )

        assert len(qh_df) == 4
        # Validate schema
        validated = quarter_hourly_schema.validate(qh_df)
        assert len(validated) == 4

        # Interval 0: valid
        assert qh_df.iloc[0]["quality_flag"] == "VALID"
        assert qh_df.iloc[0]["power_kw"] == pytest.approx(100.0)

        # Interval 2: missing
        assert qh_df.iloc[2]["quality_flag"] == "MISSING_DATA"
        assert np.isnan(qh_df.iloc[2]["power_kw"])

    def test_quarter_hourly_parquet_on_disk(self) -> None:
        """Verify the generated quarter_hourly.parquet file on disk."""
        qh_path = "data/processed/quarter_hourly.parquet"
        df = pd.read_parquet(qh_path)

        assert len(df) == 240384
        validated = quarter_hourly_schema.validate(df)
        assert len(validated) == 240384

        # Check frequency is exactly 15 minutes
        diffs = df["interval_start_utc"].diff().dropna()
        assert (diffs == pd.Timedelta(minutes=15)).all()

        # Check quality flags distribution
        flags = df["quality_flag"].value_counts().to_dict()
        assert flags["VALID"] > 230000
        assert flags["SUSPECT_ZERO"] == 1792  # Exactly 448 hours * 4
        assert flags["MISSING_DATA"] == 323

    def test_hourly_kwh_equals_sum_of_quarter_hourly_kw(self) -> None:
        """Verify that hourly kWh equals sum(15-min kW * 0.25) for complete hours."""
        qh_path = "data/processed/quarter_hourly.parquet"
        h_path = "data/processed/hourly.parquet"
        if not Path(qh_path).exists() or not Path(h_path).exists():
            pytest.skip("Processed datasets not found")

        df_qh = pd.read_parquet(qh_path)
        df_h = pd.read_parquet(h_path)

        df_qh["hour_start"] = df_qh["interval_start_utc"].dt.floor("h")
        valid_qh = df_qh[df_qh["quality_flag"] == "VALID"]
        qh_complete = valid_qh.groupby("hour_start").filter(
            lambda g: len(g) == 4 and not g["power_kw"].isna().any()
        )
        # Sample complete hours across the dataset
        sample_hours = qh_complete["hour_start"].drop_duplicates().iloc[:500]
        qh_sample = qh_complete[qh_complete["hour_start"].isin(sample_hours)]
        qh_sum = qh_sample.groupby("hour_start")["power_kw"].sum() * 0.25

        df_h_valid = df_h.set_index("interval_start_utc")
        aligned_h = df_h_valid.loc[qh_sum.index, "target_kwh"]

        assert np.allclose(aligned_h.values, qh_sum.values, atol=1e-5)


class TestHRRRExtraction:
    """Unit and regression tests for NOAA HRRR forecast weather extraction pipeline."""

    def test_find_eligible_run_nominal(self) -> None:
        """Verify nominal run selection when H - 2h is available in scan set."""
        h = pd.Timestamp("2022-06-13 14:00:00", tz="UTC")
        nominal_init = pd.Timestamp("2022-06-13 12:00:00", tz="UTC")
        mock_scan = {nominal_init}

        fs = get_thread_fs()
        init_time, offset = find_eligible_run(
            issue_time=h,
            fs=fs,
            valid_scan_set=mock_scan,
            lag_hours=2,
            max_fallback_hours=6,
        )

        assert init_time == nominal_init
        assert offset == 0

    def test_find_eligible_run_fallback(self) -> None:
        """Verify fallback to earlier run when nominal H - 2h is missing."""
        h = pd.Timestamp("2022-06-13 14:00:00", tz="UTC")
        fallback_init = pd.Timestamp("2022-06-13 11:00:00", tz="UTC")  # 3 hours prior (offset 1)
        mock_scan = {fallback_init}

        fs = get_thread_fs()
        init_time, offset = find_eligible_run(
            issue_time=h,
            fs=fs,
            valid_scan_set=mock_scan,
            lag_hours=2,
            max_fallback_hours=6,
        )

        assert init_time == fallback_init
        assert offset == 1

    def test_find_eligible_run_max_fallback_exceeded(self) -> None:
        """Verify None returned when all runs within max_fallback_hours are missing."""
        h = pd.Timestamp("2022-06-13 14:00:00", tz="UTC")
        too_old_init = pd.Timestamp("2022-06-13 04:00:00", tz="UTC")  # 10h prior (exceeds 6+2)
        mock_scan = {too_old_init}

        fs = get_thread_fs()
        init_time, offset = find_eligible_run(
            issue_time=h,
            fs=fs,
            valid_scan_set=mock_scan,
            lag_hours=2,
            max_fallback_hours=6,
        )

        assert init_time is None
        assert offset == -1

    def test_extract_hour_weather_unit_conversion_mocked(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify temperature unit conversion (Kelvin to Celsius) and lead averaging."""
        h = pd.Timestamp("2022-06-13 14:00:00", tz="UTC")
        init_time = pd.Timestamp("2022-06-13 12:00:00", tz="UTC")
        mock_scan = {init_time}

        # Mock array returning 300.15 K (F02) and 298.15 K (F03) -> mean 299.15 K = 26.0 C
        # For cloud cover: 10.0% (F02) and 20.0% (F03) -> mean 15.0%
        class MockZarrArray:
            def __getitem__(self, key: tuple[int, int, int]) -> float:
                lead_idx = key[0]
                if lead_idx == 1:  # F02 (lead_start 2, index 1)
                    return 300.15
                elif lead_idx == 2:  # F03 (lead_end 3, index 2)
                    return 298.15
                return 0.0

        monkeypatch.setattr(
            "src.data.hrrr.get_cached_zarr_array", lambda var_path, fs: MockZarrArray()
        )

        cfg = HRRRExtractorConfig()
        fs = get_thread_fs()
        res = extract_hour_weather(h, cfg, fs, mock_scan)

        assert res["extraction_status"] == "SUCCESS:tmp,tcdc"
        assert res["fallback_used"] is False
        assert res["run_age_hours"] == 2.0
        assert res["tmp_c"] == pytest.approx(26.0)
        # Check cloud variables also computed or NaN based on requested variables
        assert res["tcdc_pct"] == pytest.approx(299.15)
        assert np.isnan(res["lcdc_pct"])
        assert np.isnan(res["mcdc_pct"])
        assert np.isnan(res["hcdc_pct"])
        # Verify schema columns
        for col in HRRR_FEATURE_COLUMNS:
            assert col in res

    def test_variable_subset_selection(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify variable subset selection and unrequested columns staying NaN."""
        h = pd.Timestamp("2022-06-13 14:00:00", tz="UTC")
        init_time = pd.Timestamp("2022-06-13 12:00:00", tz="UTC")
        mock_scan = {init_time}

        class MockZarrArray:
            def __getitem__(self, key: tuple[int, int, int]) -> float:
                return 300.15

        monkeypatch.setattr(
            "src.data.hrrr.get_cached_zarr_array", lambda var_path, fs: MockZarrArray()
        )

        # Test single variable tmp
        cfg_tmp = HRRRExtractorConfig(variables=["tmp"])
        fs = get_thread_fs()
        res_tmp = extract_hour_weather(h, cfg_tmp, fs, mock_scan)
        assert res_tmp["extraction_status"] == "SUCCESS:tmp"
        assert not np.isnan(res_tmp["tmp_c"])
        assert np.isnan(res_tmp["tcdc_pct"])
        assert np.isnan(res_tmp["lcdc_pct"])
        assert np.isnan(res_tmp["mcdc_pct"])
        assert np.isnan(res_tmp["hcdc_pct"])

        # Test all 5 variables
        cfg_all = HRRRExtractorConfig(variables=["tmp", "tcdc", "lcdc", "mcdc", "hcdc"])
        res_all = extract_hour_weather(h, cfg_all, fs, mock_scan)
        assert res_all["extraction_status"] == "SUCCESS:tmp,tcdc,lcdc,mcdc,hcdc"
        assert not np.isnan(res_all["tmp_c"])
        assert not np.isnan(res_all["tcdc_pct"])
        assert not np.isnan(res_all["lcdc_pct"])
        assert not np.isnan(res_all["mcdc_pct"])
        assert not np.isnan(res_all["hcdc_pct"])

    def test_merge_parts(self, tmp_path: Path) -> None:
        """Verify merging of multiple monthly parquet parts."""
        part1 = tmp_path / "hrrr_features_2023_10.parquet"
        part2 = tmp_path / "hrrr_features_2023_11.parquet"

        df1 = pd.DataFrame({col: [0.0] for col in HRRR_FEATURE_COLUMNS})
        df1["site_id"] = "2107"
        df1["target_start_utc"] = [pd.Timestamp("2023-10-15 12:00:00", tz="UTC")]
        df1["extraction_status"] = "SUCCESS"
        df1["fallback_used"] = False
        df1.to_parquet(part1)

        df2 = pd.DataFrame({col: [0.0] for col in HRRR_FEATURE_COLUMNS})
        df2["site_id"] = "2107"
        df2["target_start_utc"] = [pd.Timestamp("2023-11-01 12:00:00", tz="UTC")]
        df2["extraction_status"] = "SUCCESS"
        df2["fallback_used"] = False
        df2.to_parquet(part2)

        merged_file = tmp_path / "merged.parquet"
        merged_df = merge_parts(parts_dir=tmp_path, output_merged_path=merged_file)

        t1 = pd.Timestamp("2023-10-15 12:00:00", tz="UTC")
        t2 = pd.Timestamp("2023-11-01 12:00:00", tz="UTC")
        assert len(merged_df) == 2
        assert merged_df["target_start_utc"].iloc[0] == t1
        assert merged_df["target_start_utc"].iloc[1] == t2

    def test_cli_argument_parsing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify CLI argument parsing in scripts/hrrr_extract.py."""
        import scripts.hrrr_extract as he

        test_args = [
            "scripts/hrrr_extract.py",
            "--start", "2022-10-17",
            "--end", "2023-11-09",
            "--variables", "tmp,tcdc",
            "--threads", "8",
            "--parts-dir", "data/test_parts",
        ]
        monkeypatch.setattr("sys.argv", test_args)

        # Intercept run_hrrr_pipeline so we don't do real I/O
        captured_kwargs: dict[str, Any] = {}

        def mock_pipeline(**kwargs: Any) -> dict[str, Any]:
            captured_kwargs.update(kwargs)
            return {
                "total_daylight_hours": 0,
                "total_months": 0,
                "processed_months": [],
                "skipped_months": [],
                "merged": False,
            }

        monkeypatch.setattr(he, "run_hrrr_pipeline", mock_pipeline)
        ret = he.main()

        assert ret == 0
        cfg = captured_kwargs["config"]
        assert cfg.start_date == "2022-10-17"
        assert cfg.end_date == "2023-11-09"
        assert cfg.variables == ["tmp", "tcdc"]
        assert cfg.num_threads == 8
        assert captured_kwargs["parts_dir"] == Path("data/test_parts")

    def test_regression_june_week_sample(self) -> None:
        """Regression test against data/processed/june_week_extracted.parquet."""
        june_path = Path("data/processed/june_week_extracted.parquet")
        if not june_path.exists():
            pytest.skip("june_week_extracted.parquet not found")

        df_june = pd.read_parquet(june_path)
        df_june_day = df_june[df_june["is_daylight"]].copy().reset_index(drop=True)

        cfg = HRRRExtractorConfig.from_yaml()
        fs = get_thread_fs()
        scan_set = load_coverage_scan_set()

        # Check first daylight hour
        row = df_june_day.iloc[0]
        t = pd.to_datetime(row["interval_start_utc"], utc=True)
        res = extract_hour_weather(t, cfg, fs, scan_set)

        assert res["extraction_status"].startswith(("SUCCESS", "FALLBACK_SUCCESS"))
        assert res["tmp_c"] == pytest.approx(row["forecast_tmp_c"], abs=0.1)
        assert res["tcdc_pct"] == pytest.approx(row["forecast_tcdc_pct"], abs=0.1)

