import json

with open('data/raw/metadata/9068_system_metadata.json', 'r') as f:
    meta_9068 = json.load(f)

print("System:", meta_9068.get('System'))
print("Site:", meta_9068.get('Site'))
print("Mount:", meta_9068.get('Mount'))
print("Meters:", meta_9068.get('Meters'))
print("Inverters:", meta_9068.get('Inverters'))

print("\n9068 AC Power / Energy metrics:")
for k, v in meta_9068.get('Metrics', {}).items():
    cname = str(v.get('common_name', ''))
    if any(t in cname.lower() or 'power' in k.lower() or 'meter' in k.lower() for t in ['ac power', 'ac energy', 'meter']):
        print(f"  {k}: common='{cname}', units='{v.get('units')}', scale={v.get('scale_factor')}, offset={v.get('offset')}, interval={v.get('time_interval')}")
