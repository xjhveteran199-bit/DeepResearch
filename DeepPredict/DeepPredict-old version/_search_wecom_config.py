import json
from pathlib import Path

# Check backup files for WeCom credentials
files = [
    Path(r'C:\Users\XJH\.openclaw\openclaw.json'),
    Path(r'C:\Users\XJH\.openclaw\openclaw.json.bak'),
    Path(r'C:\Users\XJH\.openclaw\openclaw.json.bak.1'),
    Path(r'C:\Users\XJH\.openclaw\openclaw.json.bak.2'),
]

for f in files:
    if f.exists():
        try:
            data = json.loads(f.read_text(encoding='utf-8'))
            # Look for WeCom related keys
            text = json.dumps(data)
            if 'WECOM' in text or 'wecom' in text.lower() or 'corpid' in text.lower():
                print(f'Found in {f.name}')
                # Print the relevant sections
                for key in ['env', 'channels', 'plugins']:
                    if key in data:
                        section = data[key]
                        if 'wecom' in json.dumps(section).lower() or 'WECOM' in json.dumps(section):
                            print(f'  Section {key}:', json.dumps(section, indent=2)[:500])
        except Exception as e:
            print(f'Error reading {f.name}: {e}')
