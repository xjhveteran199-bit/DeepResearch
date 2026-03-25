import json
from pathlib import Path

config_path = Path.home() / ".openclaw" / "openclaw.json"
print('Config path:', config_path)
print('Exists:', config_path.exists())

if config_path.exists():
    data = json.loads(config_path.read_text(encoding='utf-8'))
    env = data.get('env', {}).get('vars', {})
    print('WECOM_CORP_ID:', env.get('WECOM_CORP_ID', 'NOT SET'))
    print('WECOM_CORP_SECRET:', 'SET' if env.get('WECOM_CORP_SECRET') else 'NOT SET')
    print('WECOM_AGENT_ID:', env.get('WECOM_AGENT_ID', 'NOT SET'))
    print('WECOM_PROXY:', env.get('WECOM_PROXY', 'NOT SET'))
