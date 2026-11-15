import json
import pandas as pd
import numpy as np
import pvlib
from pathlib import Path

csv_path = Path("data/raw/2107_meter_15m_data.csv")
meta_path = Path("data/raw/metadata/2107_system_metadata.json")

with open(meta_path, "r", encoding="utf-8") as f:
    meta = json.load(f)

# Read CSV header and first 5 rows raw
with open(csv_path, "r", encoding="utf-8") as f:
    lines = [f.readline().strip() for _ in range(6)]

print("=== a) Timestamp format ===")
print("Header:", lines[0])
print("First 3 rows:", lines[1:4])
col_ts, col_val = lines[0].split(",")
has_tz = any("+" in l.split(",")[0] or "-" in l.split(",")[0][10:] or "Z" in l.split(",")[0] for l in lines[1:10])
print(f"Timestamp column name: {col_ts}")
print(f"Power column name: {col_val}")
print(f"Raw format example: '{lines[1].split(',')[0]}'")
print(f"Carries timezone offset string: {has_tz}")

# Load full data
df = pd.read_csv(csv_path)
print(f"\nTotal rows in CSV: {len(df):,}")

# Parse datetime
df['dt'] = pd.to_datetime(df[col_ts])
df['date_str'] = df['dt'].dt.strftime('%Y-%m-%d')

print("\n=== b) Daylight Saving Test ===")
# Test 2022-03-13 (spring forward) and 2022-11-06 (fall back)
# Also check other years: 2019, 2020, 2021
for yr, spring_date, fall_date in [
    (2019, '2019-03-10', '2019-11-03'),
    (2020, '2020-03-08', '2020-11-01'),
    (2021, '2021-03-14', '2021-11-07'),
    (2022, '2022-03-13', '2022-11-06')
]:
    df_sp = df[df['date_str'] == spring_date]
    df_fa = df[df['date_str'] == fall_date]
    print(f"\nYear {yr}:")
    print(f"  Spring {spring_date}: {len(df_sp)} rows")
    if len(df_sp) != 96:
        # find gap
        sp_diffs = df_sp['dt'].diff().dropna()
        gaps = sp_diffs[sp_diffs > pd.Timedelta(minutes=15)]
        for idx in gaps.index:
            print(f"    Gap: {df.loc[idx-1, 'dt']} -> {df.loc[idx, 'dt']} ({gaps[idx]})")
    print(f"  Fall   {fall_date}: {len(df_fa)} rows")
    # check duplicates in fall date
    dups = df_fa[df_fa['dt'].duplicated(keep=False)]
    if not dups.empty:
        print(f"    Duplicates in fall: {dups['dt'].tolist()}")
    else:
        # Check timestamps between 01:00 and 03:00 on fall date
        t_fa = df_fa[df_fa['dt'].dt.hour.between(0, 3)]
        print(f"    Fall 00-03h timestamps ({len(t_fa)} rows): {t_fa['dt'].dt.strftime('%H:%M').tolist()}")

print("\n=== c) Solar-Noon Test ===")
lat = 38.996306
lon = -122.134111
site = pvlib.location.Location(lat, lon, tz='UTC')

# Test several clear summer and winter days:
# Summer days: 2021-06-20, 2021-06-21, 2022-06-21, 2022-06-22
# Winter days: 2020-12-21, 2021-12-21, 2022-12-21
test_dates = ['2021-06-21', '2022-06-21', '2021-12-21', '2022-12-21']

for d in test_dates:
    day_df = df[df['date_str'] == d].copy()
    if day_df.empty:
        continue
    # Find peak power timestamp
    max_val = day_df[col_val].max()
    peak_row = day_df[day_df[col_val] == max_val].iloc[0]
    peak_ts = peak_row['dt']
    
    # Calculate solar noon in UTC for that day
    # Times in UTC for full day:
    times_utc = pd.date_range(f"{d} 00:00", f"{d} 23:59", freq='1min', tz='UTC')
    solpos = site.get_solarposition(times_utc)
    solar_noon_utc = solpos['apparent_elevation'].idxmax()
    
    # Hypothesis 1: timestamps in CSV are Local Time with DST (PST/PDT, America/Los_Angeles)
    # If peak_ts is local time, what is that in UTC?
    peak_ts_hyp1_utc = pd.Timestamp(peak_ts).tz_localize('America/Los_Angeles').tz_convert('UTC')
    err_hyp1_min = (peak_ts_hyp1_utc - solar_noon_utc).total_seconds() / 60.0
    
    # Hypothesis 2: timestamps in CSV are Fixed Standard Time (PST, UTC-8)
    peak_ts_hyp2_utc = pd.Timestamp(peak_ts).tz_localize('Etc/GMT+8').tz_convert('UTC')
    err_hyp2_min = (peak_ts_hyp2_utc - solar_noon_utc).total_seconds() / 60.0
    
    # What time is solar noon when converted to America/Los_Angeles?
    sn_local = solar_noon_utc.tz_convert('America/Los_Angeles')
    sn_pst = solar_noon_utc.tz_convert('Etc/GMT+8')
    
    print(f"Date: {d} | Peak AC power = {max_val:.1f} kW at {peak_ts.strftime('%H:%M:%S')}")
    print(f"  Solar noon UTC: {solar_noon_utc.strftime('%H:%M:%S %Z')}")
    print(f"  Solar noon in Local (with DST): {sn_local.strftime('%H:%M:%S %Z')} -> Hyp 1 error: {err_hyp1_min:+.1f} min")
    print(f"  Solar noon in Fixed PST (UTC-8): {sn_pst.strftime('%H:%M:%S %Z')} -> Hyp 2 error: {err_hyp2_min:+.1f} min")

print("\n=== d) AC Power Statistics ===")
vals = df[col_val]
max_p = vals.max()
p999 = vals.quantile(0.999)
p99 = vals.quantile(0.99)
p50 = vals.median()
exact_zeros = (vals == 0).sum()
pct_zeros = (vals == 0).mean() * 100
negatives = (vals < 0).sum()
nulls = vals.isnull().sum()

print(f"Total rows: {len(vals):,}")
print(f"Max: {max_p:.2f} kW")
print(f"99.9th percentile: {p999:.2f} kW")
print(f"99.0th percentile: {p99:.2f} kW")
print(f"Median: {p50:.2f} kW")
print(f"Exact zeros: {exact_zeros:,} ({pct_zeros:.2f}%)")
print(f"Negative values: {negatives}")
print(f"Null values: {nulls}")

# Check night values (using pvlib solar elevation for all rows)
# To be fast, sample 1000 night timestamps across winter and summer
print("\nChecking nighttime generation...")
# Localize to America/Los_Angeles and get solar elevation
times_la = df['dt'].dt.tz_localize('America/Los_Angeles', ambiguous='NaT', nonexistent='NaT')
valid_mask = times_la.notnull()
sol_sample = site.get_solarposition(times_la[valid_mask].iloc[::20].dt.tz_convert('UTC'))
night_sample = sol_sample[sol_sample['apparent_elevation'] < -6.0]
night_idx = night_sample.index.tz_convert('America/Los_Angeles').tz_localize(None)
night_vals = df.set_index('dt').reindex(night_idx)[col_val].dropna()
print(f"Night sample values count: {len(night_vals):,}")
print(f"Night max: {night_vals.max():.2f} kW, mean: {night_vals.mean():.4f} kW, >0 count: {(night_vals > 0).sum()}")

# Check flat-lined stretches during daytime
print("\nChecking flat-lined daytime stretches...")
# Consecutive identical non-zero values
diff_vals = vals.diff()
diff_vals[(vals == 0) | (vals.shift(1) == 0)] = np.nan
flat_runs = (diff_vals == 0)
print(f"Number of consecutive identical non-zero readings: {flat_runs.sum()}")

# High values analysis: why is max 1783.68 kW when DC is 893 kW?
high_vals = df[vals > 893.0]
print(f"\nRows with AC power > 893 kW (DC nameplate): {len(high_vals):,} ({len(high_vals)/len(df)*100:.3f}%)")
if not high_vals.empty:
    print("Highest 10 values:")
    print(df.nlargest(10, col_val)[[col_ts, col_val]].to_string())

print("\n=== e) Interval average vs Instantaneous snapshot ===")
print("System metadata notes / comments:")
print("  System comments:", meta.get('System', {}).get('comments'))
print("  Meter info:", meta.get('Meters', {}))
metrics = meta.get('Metrics', {})
for k, v in metrics.items():
    if 'meter' in k.lower():
        print(f"  Meter metric {k}: {v}")

print("\n=== f) Licence / Terms text ===")
print("Data Prize metadata info:")
print("  comments:", meta.get('System', {}).get('comments'))
print("  public_name:", meta.get('System', {}).get('public_name'))
