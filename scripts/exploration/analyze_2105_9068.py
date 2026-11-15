import json
import pandas as pd
from pathlib import Path

# Load metadata
with open('data/raw/metadata/2105_system_metadata.json', 'r') as f:
    meta_2105 = json.load(f)
with open('data/raw/metadata/9068_system_metadata.json', 'r') as f:
    meta_9068 = json.load(f)

print("=== 2105 METADATA ===")
print("System:", meta_2105.get('System'))
print("Site:", meta_2105.get('Site'))
print("Mount:", meta_2105.get('Mount'))
print("Meters:", meta_2105.get('Meters'))
print("Inverters count:", len(meta_2105.get('Inverters', {})))
print("Inverters:", meta_2105.get('Inverters'))

# 2105 metrics
print("\n2105 metrics related to AC / power / energy:")
for k, v in meta_2105.get('Metrics', {}).items():
    if any(t in k.lower() or t in str(v.get('common_name')).lower() for t in ['ac', 'power', 'energy', 'kw', 'kwh']):
        print(f"  {k}: common={v.get('common_name')}, units={v.get('units')}, scale={v.get('scale_factor')}, offset={v.get('offset')}, interval={v.get('time_interval')}")

print("\n=== 9068 METADATA ===")
print("System:", meta_9068.get('System'))
print("Site:", meta_9068.get('Site'))
print("Mount:", meta_9068.get('Mount'))
print("Meters:", meta_9068.get('Meters'))
print("Inverters count:", len(meta_9068.get('Inverters', {})))
print("Inverters:", meta_9068.get('Inverters'))

# 9068 metrics
print("\n9068 metrics related to AC / power / energy:")
for k, v in meta_9068.get('Metrics', {}).items():
    if any(t in k.lower() or t in str(v.get('common_name')).lower() for t in ['ac', 'power', 'energy', 'kw', 'kwh', 'sco_mtr']):
        print(f"  {k}: common={v.get('common_name')}, units={v.get('units')}, scale={v.get('scale_factor')}, offset={v.get('offset')}, interval={v.get('time_interval')}")
