# -*- coding: utf-8 -*-
"""发送 INDPRO 简报给许总监"""
import json, urllib.request, sys
from pathlib import Path

config_path = Path.home() / ".openclaw" / "openclaw.json"
data = json.loads(config_path.read_text(encoding='utf-8'))
wecom = data.get('channels', {}).get('wecom', {})
bot_id = wecom.get('botId')
secret = wecom.get('secret')
proxy = wecom.get('proxy', 'http://10.147.17.105:8888')

print(f'Bot ID: {bot_id}, Proxy: {proxy}')

if not bot_id or not secret:
    print('ERROR: Missing WeCom credentials')
    sys.exit(1)

# 读取简报
report_path = r'C:\Users\XJH\DeepPredict\_report_indpro.md'
report_text = open(report_path, encoding='utf-8').read()

# 构建opener
handler = urllib.request.ProxyHandler({'https': proxy, 'http': proxy})
opener = urllib.request.build_opener(handler)

# 获取access_token
token_url = f'https://qyapi.weixin.qq.com/cgi-bin/gettoken?corpid={bot_id}&corpsecret={secret}'
try:
    resp = opener.open(token_url, timeout=10)
    token_data = json.loads(resp.read())
    print('Token response:', token_data)
    if token_data.get('errcode', 0) != 0:
        print('ERROR getting token:', token_data)
        sys.exit(1)
    access_token = token_data['access_token']
except Exception as e:
    print('ERROR getting access token:', e)
    sys.exit(1)

# 分割消息（企业微信限制每条消息长度）
MAX_LEN = 1800
lines = report_text.split('\n')
messages = []
current = ''
for line in lines:
    if len(current) + len(line) + 1 <= MAX_LEN:
        current = current + line + '\n'
    else:
        if current:
            messages.append(current.rstrip('\n'))
        current = line + '\n'
if current.strip():
    messages.append(current.rstrip('\n'))
print(f'Messages to send: {len(messages)}')

# 发送消息给 xujh（许总监）
user_id = 'xujh'
send_url = f'https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token={access_token}'

for i, msg_text in enumerate(messages):
    payload = {
        'touser': user_id,
        'msgtype': 'text',
        'agentid': 1000002,
        'text': {'content': msg_text},
        'safe': 0
    }
    req = urllib.request.Request(
        send_url,
        data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    try:
        resp = opener.open(req, timeout=15)
        result = json.loads(resp.read())
        print(f'Message {i+1} result: errcode={result.get("errcode")}, errmsg={result.get("errmsg")}')
        if result.get('errcode') == 0:
            print(f'  ✅ Message {i+1} sent successfully')
        else:
            print(f'  ❌ Message {i+1} failed: {result}')
    except Exception as e:
        print(f'Message {i+1} error: {e}')

print('\nDone!')
