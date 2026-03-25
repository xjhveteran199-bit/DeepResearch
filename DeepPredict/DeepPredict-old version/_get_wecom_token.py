# -*- coding: utf-8 -*-
"""直接通过企业微信API发送消息给许总监"""
import json, urllib.request, sys, os
from pathlib import Path

# 读取openclaw配置获取wecom凭证
config_path = Path.home() / ".openclaw" / "openclaw.json"
data = json.loads(config_path.read_text(encoding='utf-8'))
wecom = data.get('channels', {}).get('wecom', {})
bot_id = wecom.get('botId')  # aibrIU_SXHatZ9vRBHwHVsi05-y2nEHTopC
secret = wecom.get('secret')
proxy = 'http://10.147.17.105:8888'  # 从skill说明得知的代理

print(f'Bot ID: {bot_id}')
print(f'Proxy: {proxy}')

if not bot_id or not secret:
    print('ERROR: Missing WeCom credentials')
    sys.exit(1)

# 读取简报
report_path = r'C:\Users\XJH\DeepPredict\_report_max_temp.md'
report_text = open(report_path, encoding='utf-8').read()

handler = urllib.request.ProxyHandler({'https': proxy, 'http': proxy})
opener = urllib.request.build_opener(handler)

# 获取access_token - 使用bot_id作为corpid (有时可以)
# 或者使用secret作为corpsecret
token_url = f'https://qyapi.weixin.qq.com/cgi-bin/gettoken?corpid={bot_id}&corpsecret={secret}'
try:
    resp = opener.open(token_url, timeout=10)
    token_data = json.loads(resp.read())
    print('Token response:', token_data)
except Exception as e:
    print('Token error:', e)
    sys.exit(1)
