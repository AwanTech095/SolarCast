import pandas as pd
from pathlib import Path

# 2105 sample
p_2105 = Path('data/raw/samples/2105_inv01_data.csv.head.csv')
with open(p_2105, 'r', encoding='utf-8') as f:
    lines = [f.readline() for _ in range(10)]
print("2105 inv01 header:")
print(lines[0].strip().split(',')[:10])
print("2105 inv01 row 1:")
print(lines[1].strip()[:150])

df_2105 = pd.read_csv(p_2105, nrows=100)
print("\n2105 date range in sample:", df_2105['measured_on'].iloc[0], "to", df_2105['measured_on'].iloc[-1])

# 9068 sample
p_9068 = Path('data/raw/samples/9068_ac_power_data.csv.head.csv')
with open(p_9068, 'r', encoding='utf-8') as f:
    lines_9068 = [f.readline() for _ in range(10)]
print("\n9068 ac_power header:")
print(lines_9068[0].strip().split(',')[:10])
print("9068 ac_power row 1:")
print(lines_9068[1].strip()[:150])

df_9068 = pd.read_csv(p_9068, nrows=100)
print("\n9068 date range in sample:", df_9068['measured_on'].iloc[0], "to", df_9068['measured_on'].iloc[-1])
