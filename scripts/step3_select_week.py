"""Find optimal mixed-weather Monday-Sunday week from 2022-10-01 to 2023-03-31."""

import pandas as pd
import numpy as np
import pvlib

# Load hourly data
df = pd.read_parquet("data/processed/hourly.parquet")
df["interval_start_utc"] = pd.to_datetime(df["interval_start_utc"], utc=True)
df["interval_end_utc"] = pd.to_datetime(df["interval_end_utc"], utc=True)

# Filter 2022-10-01 to 2023-03-31
start_bound = pd.Timestamp("2022-10-01 00:00:00", tz="UTC")
end_bound = pd.Timestamp("2023-03-31 23:59:59", tz="UTC")

# Calculate pvlib clear-sky GHI at midpoint for all rows
site_lat = 38.996306
site_lon = -122.134111
location = pvlib.location.Location(latitude=site_lat, longitude=site_lon, altitude=10.0)

# Midpoint times for all rows
mid_times = pd.DatetimeIndex(df["interval_start_utc"] + pd.Timedelta(minutes=30))
cs = location.get_clearsky(mid_times)
df["clearsky_ghi"] = cs["ghi"].values

# Definition of clear-sky index proxy:
# target_kwh / clearsky_ghi, normalised by 95th percentile over all VALID high_sun hours
valid_high_sun = df[(df["quality_flag"] == "VALID") & (df["high_sun"]) & (df["clearsky_ghi"] > 10.0)].copy()
raw_ratio_all = valid_high_sun["target_kwh"] / valid_high_sun["clearsky_ghi"]
p95 = np.percentile(raw_ratio_all.dropna(), 95)
print(f"95th percentile of (target_kwh / clearsky_ghi) over all VALID high_sun hours: {p95:.6f}")

df["kt_proxy"] = np.nan
mask = (df["quality_flag"] == "VALID") & (df["high_sun"]) & (df["clearsky_ghi"] > 10.0)
df.loc[mask, "kt_proxy"] = (df.loc[mask, "target_kwh"] / df.loc[mask, "clearsky_ghi"]) / p95

# Find all Monday-Sunday 7-day windows in [2022-10-01, 2023-03-31]
# A Monday starts at 00:00 UTC and ends Sunday 23:59:59 UTC (168 hours)
# Filter df to the range
sub_df = df[(df["interval_start_utc"] >= start_bound) & (df["interval_start_utc"] <= end_bound)].copy()

# Mondays in range
mondays = pd.date_range("2022-10-03", "2023-03-27", freq="W-MON", tz="UTC")

candidates = []
for mon in mondays:
    sun_end = mon + pd.Timedelta(days=7)
    week_slice = sub_df[(sub_df["interval_start_utc"] >= mon) & (sub_df["interval_start_utc"] < sun_end)]
    if len(week_slice) != 168:
        continue
    # Check if ALL rows are VALID
    all_valid = (week_slice["quality_flag"] == "VALID").all()
    # High sun hours in this week
    high_sun_slice = week_slice[week_slice["high_sun"]]
    kt_vals = high_sun_slice["kt_proxy"].dropna()
    
    kt_std = kt_vals.std() if len(kt_vals) > 0 else 0.0
    kt_range = (kt_vals.max() - kt_vals.min()) if len(kt_vals) > 0 else 0.0
    kt_iqr = (np.percentile(kt_vals, 75) - np.percentile(kt_vals, 25)) if len(kt_vals) > 0 else 0.0
    
    candidates.append({
        "monday": mon.strftime("%Y-%m-%d"),
        "sunday": (sun_end - pd.Timedelta(hours=1)).strftime("%Y-%m-%d"),
        "all_valid": all_valid,
        "n_high_sun": len(high_sun_slice),
        "kt_std": kt_std,
        "kt_range": kt_range,
        "kt_iqr": kt_iqr,
        "kt_min": kt_vals.min() if len(kt_vals) > 0 else 0,
        "kt_max": kt_vals.max() if len(kt_vals) > 0 else 0,
        "kt_mean": kt_vals.mean() if len(kt_vals) > 0 else 0,
    })

cand_df = pd.DataFrame(candidates)
valid_cands = cand_df[cand_df["all_valid"]].sort_values("kt_std", ascending=False)
print("\nTop 10 Monday-Sunday weeks where all rows are VALID, ranked by kt_proxy std:")
print(valid_cands.head(10).to_string(index=False))
