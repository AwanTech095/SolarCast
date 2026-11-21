"""Hourly dataset aggregation and solar position calculation for SolarCast."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib
import yaml

from src.data.validation import hourly_schema, validate_and_quarantine_raw_generation


def localize_and_convert_to_utc(
    df: pd.DataFrame,
    timestamp_col: str = "measured_on",
    timezone: str = "America/Los_Angeles",
    interval_minutes: int = 15,
) -> pd.DataFrame:
    """Convert naive local civil timestamps to UTC.

    Handles Daylight Saving Time (DST) transitions:
    - Ambiguous timestamps (autumn fall-back): assigns them to the first occurrence
      (Daylight Time) using ambiguous=True. The second occurrence is treated as missing.
    - Nonexistent timestamps (spring forward): skips or shifts forward.

    Labels are LEFT-aligned: label t covers [t, t + interval_minutes).
    Canonical keys: site_id + interval_start_utc; also stores interval_end_utc.

    Args:
        df: DataFrame containing the local timestamp column.
        timestamp_col: Name of the local timestamp column.
        timezone: Local IANA timezone name (default 'America/Los_Angeles').
        interval_minutes: Length of measurement interval in minutes (default 15).

    Returns:
        DataFrame with added interval_start_utc and interval_end_utc columns.
    """
    df_out = df.copy()

    # Parse local naive timestamps
    local_dt = pd.to_datetime(df_out[timestamp_col])

    # Localize to local timezone:
    # ambiguous=True chooses the first occurrence (daylight saving time, e.g. PDT)
    # nonexistent='shift_forward' handles missing hour if present
    localized = local_dt.dt.tz_localize(
        timezone, ambiguous=True, nonexistent="shift_forward"
    )

    # Convert to UTC
    start_utc = localized.dt.tz_convert("UTC")
    df_out["interval_start_utc"] = start_utc
    df_out["interval_end_utc"] = start_utc + pd.Timedelta(minutes=interval_minutes)

    return df_out


def aggregate_to_hourly(
    clean_df: pd.DataFrame,
    power_col: str = "meter_revenue_grade_ac_output_meter_149578",
    min_intervals: int = 4,
    site_id: str = "2107",
    capacity_dc_kw: float = 893.0,
    capacity_ac_observed_kw: float = 708.48,
    latitude: float = 38.996306,
    longitude: float = -122.134111,
    start_time_utc: pd.Timestamp | None = None,
    end_time_utc: pd.Timestamp | None = None,
    eval_solar_at_midpoint: bool = False,
    hourly_output_path: Path | str | None = None,
) -> pd.DataFrame:
    """Aggregate 15-minute average power (kW) to hourly energy (kWh).

    Formula:
        Hourly Energy (kWh) = sum(kW) * 0.25 h over the 4 intervals of each UTC hour.
        Requires all intervals (min_intervals, default 4). If fewer intervals are
        present, target_kwh is set to NaN with an explanatory quality_flag.

    Generates a continuous UTC hourly grid spanning start_time_utc to end_time_utc.
    If not specified, spans from min to max hourly interval present in the data.

    Calculates solar elevation using pvlib for solar elevation and daylight classification.

    Args:
        clean_df: Validated 15-minute DataFrame with interval_start_utc and power_col.
        power_col: Column name containing 15-minute average power in kW.
        min_intervals: Required valid intervals per hour (default 4).
        site_id: Identifier of the PV site (default '2107').
        capacity_dc_kw: DC nameplate capacity in kW (default 893.0).
        capacity_ac_observed_kw: Maximum observed operational AC power in kW (default 708.48).
        latitude: Plant latitude in decimal degrees.
        longitude: Plant longitude in decimal degrees.
        start_time_utc: Optional starting UTC hour for grid.
        end_time_utc: Optional ending UTC hour for grid.
        eval_solar_at_midpoint: If True, evaluates solar position at hour midpoint (+30m).
                                If False, evaluates at interval_start_utc.
        hourly_output_path: Optional path to save hourly.parquet.

    Returns:
        Aggregated hourly DataFrame conforming to hourly_schema.
    """
    df_work = clean_df.copy()

    # Floor timestamps to the UTC hour
    df_work["hour_start_utc"] = df_work["interval_start_utc"].dt.floor("1h")

    # Group by UTC hour
    hourly_agg = df_work.groupby("hour_start_utc").agg(
        n_intervals=(power_col, "count"),
        sum_kw=(power_col, "sum"),
    )

    # Determine full continuous hourly grid
    if start_time_utc is None:
        start_time_utc = df_work["interval_start_utc"].min().floor("1h")
    if end_time_utc is None:
        end_time_utc = df_work["interval_start_utc"].max().floor("1h")

    full_grid = pd.date_range(start_time_utc, end_time_utc, freq="1h", tz="UTC")
    grid_df = pd.DataFrame({"interval_start_utc": full_grid})

    # Merge aggregated intervals onto the full continuous grid
    merged = grid_df.merge(hourly_agg, left_on="interval_start_utc", right_index=True, how="left")
    merged["n_intervals"] = merged["n_intervals"].fillna(0).astype(int)

    # Calculate target_kwh: require min_intervals (4), otherwise NaN
    merged["target_kwh"] = np.where(
        merged["n_intervals"] >= min_intervals,
        merged["sum_kw"] * 0.25,
        np.nan,
    )

    # Assign quality_flag
    def assign_quality_flag(n: int) -> str:
        if n >= min_intervals:
            return "VALID"
        if n == 0:
            return "MISSING_DATA"
        return "INCOMPLETE_INTERVALS"

    merged["quality_flag"] = merged["n_intervals"].apply(assign_quality_flag)

    # Interval metadata
    merged["site_id"] = str(site_id)
    merged["interval_end_utc"] = merged["interval_start_utc"] + pd.Timedelta(hours=1)
    merged["capacity_kw"] = float(capacity_dc_kw)
    merged["capacity_ac_observed_kw"] = float(capacity_ac_observed_kw)
    merged["latitude"] = float(latitude)
    merged["longitude"] = float(longitude)

    # Compute solar position via pvlib at interval midpoint (interval_start_utc + 30m)
    midpoint_times = merged["interval_start_utc"] + pd.Timedelta(minutes=30)
    sol_pos = pvlib.solarposition.get_solarposition(midpoint_times, latitude, longitude)
    merged["solar_elevation_deg"] = sol_pos["elevation"].values
    merged["is_daylight"] = merged["solar_elevation_deg"] > 0.0
    merged["high_sun"] = merged["solar_elevation_deg"] > 5.0

    # Flag SUSPECT_ZERO: high_sun (>5 deg) and target_kwh == 0 sitting inside a run of 2+ such hours
    is_candidate = merged["high_sun"] & (merged["target_kwh"] == 0.0)
    is_consec_prev = (
        is_candidate
        & is_candidate.shift(1, fill_value=False)
        & (merged["interval_start_utc"].diff() == pd.Timedelta(hours=1))
    )
    is_consec_next = (
        is_candidate
        & is_candidate.shift(-1, fill_value=False)
        & (merged["interval_start_utc"].diff(-1) == -pd.Timedelta(hours=1))
    )
    is_suspect_zero = is_consec_prev | is_consec_next
    merged.loc[is_suspect_zero, "quality_flag"] = "SUSPECT_ZERO"

    # Retain exact canonical schema columns
    final_cols = [
        "site_id",
        "interval_start_utc",
        "interval_end_utc",
        "target_kwh",
        "capacity_kw",
        "capacity_ac_observed_kw",
        "latitude",
        "longitude",
        "solar_elevation_deg",
        "is_daylight",
        "high_sun",
        "quality_flag",
    ]
    hourly_df = merged[final_cols].copy()

    # Validate against Pandera schema
    hourly_df = hourly_schema.validate(hourly_df)

    if hourly_output_path:
        out_path = Path(hourly_output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        hourly_df.to_parquet(out_path, index=False)

    return hourly_df


def run_data_pipeline(
    config_path: Path | str = "configs/data.yaml",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute end-to-end validation, quarantine, and hourly aggregation."""
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    paths = config.get("paths", {})
    raw_meter_csv = Path(paths.get("meter_csv", "data/raw/2107_meter_15m_data.csv"))
    processed_dir = Path(paths.get("processed_dir", "data/processed"))
    quarantine_parquet = processed_dir / "quarantine.parquet"
    hourly_parquet = processed_dir / "hourly.parquet"

    power_col = "meter_revenue_grade_ac_output_meter_149578"
    capacity_dc_kw = float(config.get("capacity_dc_kw", 893.0))
    capacity_ac_observed_kw = float(config.get("capacity_ac_max_observed_kw", 708.48))
    site_id = str(config.get("site_id", "2107"))
    lat = float(config.get("latitude", 38.996306))
    lon = float(config.get("longitude", -122.134111))
    timezone = str(config.get("timezone", "America/Los_Angeles"))
    min_intervals = int(config.get("min_intervals", 4))

    print(f"Reading raw meter data from {raw_meter_csv}...")
    df_raw = pd.read_csv(raw_meter_csv)

    print("Step 1: Validating and quarantining raw generation...")
    clean_df, quarantine_df = validate_and_quarantine_raw_generation(
        df_raw,
        timestamp_col="measured_on",
        power_col=power_col,
        max_capacity_kw=capacity_dc_kw,
        quarantine_output_path=quarantine_parquet,
    )
    print(f"  Valid records: {len(clean_df):,}")
    print(f"  Quarantined records: {len(quarantine_df):,} (saved to {quarantine_parquet})")

    print("Converting local labels to UTC...")
    clean_df_utc = localize_and_convert_to_utc(
        clean_df,
        timestamp_col="measured_on",
        timezone=timezone,
        interval_minutes=15,
    )

    print("Step 2: Aggregating to hourly energy (kWh)...")
    hourly_df = aggregate_to_hourly(
        clean_df_utc,
        power_col=power_col,
        min_intervals=min_intervals,
        site_id=site_id,
        capacity_dc_kw=capacity_dc_kw,
        capacity_ac_observed_kw=capacity_ac_observed_kw,
        latitude=lat,
        longitude=lon,
        hourly_output_path=hourly_parquet,
    )
    print(f"  Aggregated hourly rows: {len(hourly_df):,} (saved to {hourly_parquet})")

    return clean_df_utc, quarantine_df, hourly_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SolarCast hourly aggregation pipeline")
    parser.add_argument("--config", default="configs/data.yaml", help="Path to data config YAML")
    args = parser.parse_args()
    run_data_pipeline(args.config)
