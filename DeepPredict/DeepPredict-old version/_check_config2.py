import json
from pathlib import Path

config_path = Path.home() / ".openclaw" / "openclaw.json"
data = json.loads(config_path.read_text(encoding='utf-8'))
print('Top-level keys:', list(data.keys()))
env = data.get('env', {})
print('env keys:', list(env.keys()))
vars_ = env.get('vars', {})
print('All vars:')
for k, v in vars_.items():
    if 'WECOM' in k or 'wechat' in k.lower() or 'corp' in k.lower():
        print(f'  {k}: {str(v)[:50]}')
print()
print('All keys with wecom/agent/corp:')
for k in vars_:
    print(' ', k)
