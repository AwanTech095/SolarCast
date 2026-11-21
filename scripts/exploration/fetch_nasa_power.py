import json
import requests
from pathlib import Path

url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
params = {
    "latitude": 38.9963,
    "longitude": -122.1341,
    "parameters": "ALLSKY_SFC_SW_DWN,T2M,RH2M,WS10M",
    "community": "RE",
    "start": "20230101",
    "end": "20230105",
    "format": "JSON",
    "time-standard": "UTC"
}

out_dir = Path("data/raw/samples")
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / "nasa_power_2107_sample.json"

print(f"Requesting NASA POWER data from {url}...")
resp = requests.get(url, params=params, timeout=30)
resp.raise_for_status()

data = resp.json()
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print(f"Saved response to {out_file} ({out_file.stat().st_size:,} bytes)")

# Parse parameter metadata
meta_params = data.get("parameters", {})
print("\n=== NASA POWER Parameter Confirmation ===")
for p, p_info in meta_params.items():
    print(f"Parameter: {p}")
    print(f"  Long name: {p_info.get('longname')}")
    print(f"  Units:     {p_info.get('units')}")

# Print sample hourly timestamps and values
props = data.get("properties", {}).get("parameter", {})
timestamps = list(props.get("ALLSKY_SFC_SW_DWN", {}).keys())
print(f"\nTotal hourly timestamps fetched: {len(timestamps)}")
print("First 5 hourly readings (UTC):")
for ts in timestamps[:5]:
    sw = props.get("ALLSKY_SFC_SW_DWN", {}).get(ts)
    t2m = props.get("T2M", {}).get(ts)
    rh = props.get("RH2M", {}).get(ts)
    ws = props.get("WS10M", {}).get(ts)
    print(f"  {ts} UTC -> SW: {sw} W/m2, T2M: {t2m} C, RH2M: {rh} %, WS10M: {ws} m/s")
