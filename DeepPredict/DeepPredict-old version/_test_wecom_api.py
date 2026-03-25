# -*- coding: utf-8 -*-
import json, urllib.request, sys
from pathlib import Path

config_path = Path.home() / ".openclaw" / "openclaw.json"
data = json.loads(config_path.read_text(encoding='utf-8'))
wecom = data.get('channels', {}).get('wecom', {})
bot_id = wecom.get('botId')
secret = wecom.get('secret')
print(f'Bot ID: {bot_id}')

# Try without proxy first
opener = urllib.request.build_opener()
token_url = f'https://qyapi.weixin.qq.com/cgi-bin/gettoken?corpid={bot_id}&corpsecret={secret}'
try:
    resp = opener.open(token_url, timeout=10)
    token_data = json.loads(resp.read())
    print('Token response:', token_data)
except Exception as e:
    print('No proxy - Error:', type(e).__name__, str(e))
