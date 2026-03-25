import sys, os
sys.stdout.reconfigure(encoding='utf-8')

# Check the extracted forecast_data.csv
path = r'C:\Users\XJH\DeepResearch\test_output\extracted\forecast_data.csv'
with open(path, 'r', encoding='utf-8', errors='replace') as f:
    lines = f.readlines()

print(f'Total lines in extracted forecast_data.csv: {len(lines)}')
print(f'First 3 lines:')
for l in lines[:3]:
    print(f'  {repr(l)}')
print(f'Last 3 lines:')
for l in lines[-3:]:
    print(f'  {repr(l)}')

# Check header
header = lines[0].strip()
cols = header.split(',')
print(f'\nColumns: {cols}')
print(f'Data rows: {len(lines)-1}')

# Check if there's a 'type' column showing '历史'/'预测'
if 'type' in header.lower() or '历史' in header or '预测' in header:
    hist_count = sum(1 for l in lines[1:] if '历史' in l or 'hist' in l.lower())
    pred_count = sum(1 for l in lines[1:] if '预测' in l or 'pred' in l.lower())
    print(f'Historical rows: {hist_count}')
    print(f'Prediction rows: {pred_count}')
