import pandas as pd
import pvlib

df = pd.read_csv("data/raw/2107_meter_15m_data.csv")
col_ts, col_val = df.columns[:2]
df['dt'] = pd.to_datetime(df[col_ts])
df['hour_min'] = df['dt'].dt.strftime('%H:%M')

# Average diurnal profile in June vs December
df['month'] = df['dt'].dt.month
june = df[df['month'] == 6].groupby('hour_min')[col_val].mean()
dec = df[df['month'] == 12].groupby('hour_min')[col_val].mean()

print(f"June average peak power at: {june.idxmax()} ({june.max():.2f} kW)")
print(f"December average peak power at: {dec.idxmax()} ({dec.max():.2f} kW)")

# Let's check solar noon in UTC for June 15 and Dec 15:
site = pvlib.location.Location(38.996306, -122.134111)
sn_june_utc = site.get_solarposition(pd.date_range('2021-06-15 00:00', '2021-06-15 23:59', freq='1min', tz='UTC'))['apparent_elevation'].idxmax()
sn_dec_utc = site.get_solarposition(pd.date_range('2021-12-15 00:00', '2021-12-15 23:59', freq='1min', tz='UTC'))['apparent_elevation'].idxmax()

print(f"Solar noon June 15: {sn_june_utc.strftime('%H:%M UTC')}")
print(f"  In PDT (UTC-7): {sn_june_utc.tz_convert('America/Los_Angeles').strftime('%H:%M')}")
print(f"  In PST (UTC-8): {sn_june_utc.tz_convert('Etc/GMT+8').strftime('%H:%M')}")

print(f"Solar noon Dec 15: {sn_dec_utc.strftime('%H:%M UTC')}")
print(f"  In PST (UTC-8): {sn_dec_utc.tz_convert('America/Los_Angeles').strftime('%H:%M')}")

# Also check July:
july = df[df['month'] == 7].groupby('hour_min')[col_val].mean()
print(f"July average peak power at: {july.idxmax()} ({july.max():.2f} kW)")
