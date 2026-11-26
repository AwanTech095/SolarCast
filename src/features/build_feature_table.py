"""Build feature table for Feast feature store (Task Step 2).

Generates `data/processed/feast/features_set_b.parquet` using the SHARED
`build_features` function from `src.features.preprocessing` for all daylight
issue hours across 2018-07-12 to 2023-11-09.

Includes Feast entity key (`site_id`), timestamp columns (`event_timestamp`,
`created_timestamp`), the 29 model features matching `configs/feature_schema.json`,
and explicit audit columns:
- `production_cutoff_utc`: H - 15 minutes UTC
- `weather_available_at_utc`: Ingest release timestamp of the HRRR run (or NaT)
- `source_event_time_max_utc`: max(production_cutoff_utc, weather_available_at_utc)

Enforces the non-leakage invariant: every audit timestamp <= event_timestamp.
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features.preprocessing import (
    DEFAULT_HOURLY_PATH,
    DEFAULT_QH_PATH,
    DEFAULT_WEATHER_PATH,
    build_features,
    get_feature_names,
    load_feature_config,
)

DEFAULT_OUTPUT_PATH = Path("data/processed/feast/features_set_b.parquet")
DEFAULT_START_UTC = "2018-07-12 00:00:00+00:00"
DEFAULT_END_UTC = "2023-11-09 23:00:00+00:00"


def generate_feature_table(
    hourly_path: Path | str = DEFAULT_HOURLY_PATH,
    quarter_hourly_path: Path | str = DEFAULT_QH_PATH,
    weather_path: Path | str = DEFAULT_WEATHER_PATH,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    start_utc: str = DEFAULT_START_UTC,
    end_utc: str = DEFAULT_END_UTC,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Generate Feast feature table for PV site 2107 across all daylight hours.

    Args:
        hourly_path: Path to hourly.parquet.
        quarter_hourly_path: Path to quarter_hourly.parquet.
        weather_path: Path to hrrr_forecast_features.parquet.
        output_path: Destination parquet file path.
        start_utc: Start timestamp for issue times.
        end_utc: End timestamp for issue times.
        config: Optional feature configuration.

    Returns:
        Constructed feature table DataFrame.
    """
    if config is None:
        config = load_feature_config()

    site_id = str(config.get("site_id", "2107"))
    op_timing = config.get("operational_timing", {})
    delay_min = int(op_timing.get("production_available_delay_minutes", 15))

    # 1. Load hourly dataset to identify valid daylight issue times
    h_df = pd.read_parquet(hourly_path)
    if "interval_start_utc" in h_df.columns:
        h_df["interval_start_utc"] = pd.to_datetime(h_df["interval_start_utc"], utc=True)
        h_df.set_index("interval_start_utc", inplace=True)
    elif h_df.index.tz is None:
        h_df.index = h_df.index.tz_localize("UTC")
    else:
        h_df.index = h_df.index.tz_convert("UTC")

    t_start = pd.to_datetime(start_utc, utc=True)
    t_end = pd.to_datetime(end_utc, utc=True)

    daylight_mask = (
        (h_df.index >= t_start)
        & (h_df.index <= t_end)
        & (h_df["is_daylight"] == True)  # noqa: E712
    )
    issue_times = h_df.index[daylight_mask]

    print(
        f"Building Feature Set B table for {len(issue_times)} daylight issue hours "
        f"({start_utc} to {end_utc})..."
    )

    # 2. Extract features using the SHARED pipeline
    qh_df = pd.read_parquet(quarter_hourly_path)
    w_df = pd.read_parquet(weather_path) if Path(weather_path).exists() else None

    features_df = build_features(
        issue_times_utc=issue_times,
        hourly_df=h_df.reset_index(),
        quarter_hourly_df=qh_df,
        weather_df=w_df,
        config=config,
        feature_set="B",
    )

    # 3. Align weather availability metadata for audit
    if w_df is not None and "issue_time_utc" in w_df.columns:
        w_work = w_df.copy()
        w_work["issue_time_utc"] = pd.to_datetime(w_work["issue_time_utc"], utc=True)
        w_work.set_index("issue_time_utc", inplace=True)
        w_avail_series = pd.to_datetime(w_work["weather_available_at_utc"], utc=True).reindex(
            issue_times
        )
    else:
        w_avail_series = pd.Series(pd.NaT, index=issue_times)

    # 4. Construct audit columns
    production_cutoff = issue_times - pd.Timedelta(minutes=delay_min)
    w_avail_dt = pd.to_datetime(w_avail_series, utc=True)

    # source_event_time_max_utc: max of production_cutoff and weather_available
    # If weather is missing, max is production_cutoff
    valid_w = w_avail_dt.notna().values
    source_max_vals = np.where(
        valid_w,
        np.maximum(production_cutoff.values, w_avail_dt.values),
        production_cutoff.values,
    )
    source_max = pd.to_datetime(source_max_vals, utc=True)

    # 5. Create Feast-ready DataFrame
    created_ts = datetime.datetime.now(datetime.UTC)

    table_df = pd.DataFrame(index=issue_times)
    table_df["site_id"] = site_id
    table_df["event_timestamp"] = issue_times
    table_df["created_timestamp"] = created_ts

    # Add 29 model features in exact schema order
    feat_names = get_feature_names("B")
    for name in feat_names:
        table_df[name] = features_df[name].values

    # Add audit columns (preserve datetime64[ns, UTC])
    table_df["production_cutoff_utc"] = pd.to_datetime(production_cutoff, utc=True)
    table_df["weather_available_at_utc"] = pd.to_datetime(w_avail_series, utc=True)
    table_df["source_event_time_max_utc"] = source_max

    # 6. Validate non-leakage invariant
    assert (
        table_df["production_cutoff_utc"] <= table_df["event_timestamp"]
    ).all(), "VIOLATION: production_cutoff_utc > event_timestamp!"

    valid_weather_mask = table_df["weather_available_at_utc"].notna()
    assert (
        table_df.loc[valid_weather_mask, "weather_available_at_utc"]
        <= table_df.loc[valid_weather_mask, "event_timestamp"]
    ).all(), "VIOLATION: weather_available_at_utc > event_timestamp!"

    assert (
        table_df["source_event_time_max_utc"] <= table_df["event_timestamp"]
    ).all(), "VIOLATION: source_event_time_max_utc > event_timestamp!"

    # 7. Write to parquet
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    table_df.attrs["feature_config"] = config
    table_df.to_parquet(out, index=False)
    print(
        f"Successfully generated Feast feature table ({len(table_df)} rows, "
        f"{len(table_df.columns)} columns) -> {out}"
    )

    return table_df


def main() -> int:
    """CLI entrypoint for building Feast feature table."""
    parser = argparse.ArgumentParser(description="Build Feast Feature Table (Feature Set B)")
    parser.add_argument("--hourly", default=str(DEFAULT_HOURLY_PATH))
    parser.add_argument("--quarter-hourly", default=str(DEFAULT_QH_PATH))
    parser.add_argument("--weather", default=str(DEFAULT_WEATHER_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--start", default=DEFAULT_START_UTC)
    parser.add_argument("--end", default=DEFAULT_END_UTC)
    parser.add_argument("--feature-config", type=Path, help="Frozen bundle feature configuration")
    args = parser.parse_args()

    generate_feature_table(
        hourly_path=args.hourly,
        quarter_hourly_path=args.quarter_hourly,
        weather_path=args.weather,
        output_path=args.output,
        start_utc=args.start,
        end_utc=args.end,
        config=load_feature_config(args.feature_config) if args.feature_config else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
