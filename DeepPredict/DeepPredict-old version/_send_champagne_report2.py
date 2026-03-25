# -*- coding: utf-8 -*-
"""通过企业微信API发送DeepPredict简报给许总监（无代理）"""
import json, urllib.request, sys
from pathlib import Path

# 读取openclaw配置获取wecom凭证
config_path = Path.home() / ".openclaw" / "openclaw.json"
data = json.loads(config_path.read_text(encoding='utf-8'))
wecom = data.get('channels', {}).get('wecom', {})
bot_id = wecom.get('botId')
secret = wecom.get('secret')

print(f'Bot ID: {bot_id}')
if not bot_id or not secret:
    print('ERROR: Missing WeCom credentials')
    sys.exit(1)

# 读取简报
report_path = r'C:\Users\XJH\DeepPredict\deeppredict_report_2026-03-24_1601.md'
report_text = open(report_path, encoding='utf-8').read()
print(f'Report length: {len(report_text)} chars')

opener = urllib.request.build_opener()

# 获取access_token
token_url = f'https://qyapi.weixin.qq.com/cgi-bin/gettoken?corpid={bot_id}&corpsecret={secret}'
try:
    resp = opener.open(token_url, timeout=15)
    token_data = json.loads(resp.read())
    print('Token response:', token_data)
    if token_data.get('errcode', 0) != 0:
        print('Token error:', token_data)
        sys.exit(1)
    access_token = token_data['access_token']
except Exception as e:
    print('Token error:', e)
    sys.exit(1)

# 查找许总监的userid
lookup_url = f'https://qyapi.weixin.qq.com/cgi-bin/user/simplelist?access_token={access_token}&department_id=1&fetch_child=1'
try:
    resp = opener.open(lookup_url, timeout=10)
    users_data = json.loads(resp.read())
    print('Users:', json.dumps(users_data, ensure_ascii=False)[:500])
except Exception as e:
    print('User lookup error:', e)
    users_data = {'errcode': 0, 'userlist': []}

user_id = None
if users_data.get('errcode') == 0:
    for user in users_data.get('userlist', []):
        name = user.get('name', '')
        if '许' in name:
            user_id = user.get('userid')
            print(f'Found: {name} -> {user_id}')
            break

if not user_id:
    # 尝试使用通讯录精确查询
    search_url = f'https://qyapi.weixin.qq.com/cgi-bin/user/get?access_token={access_token}&userid={bot_id}'
    user_id = 'XuDirector'

print(f'Using userId: {user_id}')

# 分割消息（企业微信限制约680中文字符）
MAX_LEN = 1800
lines = report_text.split('\n')
messages = []
current = ''
for line in lines:
    if len(current) + len(line) + 1 <= MAX_LEN:
        current += line + '\n'
    else:
        if current:
            messages.append(current.rstrip('\n'))
        current = line + '\n'
if current.strip():
    messages.append(current.rstrip('\n'))

print(f'Sending {len(messages)} message(s)')

for i, msg in enumerate(messages):
    send_url = f'https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token={access_token}'
    payload = {
        'touser': user_id,
        'msgtype': 'text',
        'agentid': 1000002,
        'text': {'content': msg},
        'safe': 0,
    }
    req = urllib.request.Request(
        send_url,
        data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
    )
    try:
        resp = opener.open(req, timeout=10)
        result = json.loads(resp.read())
        print(f'Message {i+1}: {result}')
    except Exception as e:
        print(f'Message {i+1} error: {e}')
