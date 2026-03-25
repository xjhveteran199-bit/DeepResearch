import sys
sys.stdout.reconfigure(encoding='utf-8')

# Check the shap_analyzer.py to understand what it exports
shap_path = r'C:\Users\XJH\DeepResearch\DeepPredict\src\utils\shap_analyzer.py'
with open(shap_path, 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

# Find any save/export related code
for kw in ['save', 'export', 'png', 'zip', 'to_csv', 'to_json']:
    idx = content.find(kw)
    if idx >= 0:
        print(f'=== {kw} at {idx} ===')
        print(content[max(0,idx-100):idx+300])
        print()
