# -*- coding: utf-8 -*-
"""通过企业微信API发送DeepPredict简报给许总监"""
import json, urllib.request, sys, os
from pathlib import Path

# 读取openclaw配置获取wecom凭证
config_path = Path.home() / ".openclaw" / "openclaw.json"
data = json.loads(config_path.read_text(encoding='utf-8'))
wecom = data.get('channels', {}).get('wecom', {})
bot_id = wecom.get('botId')  # aibrIU_SXHatZ9vRBHwHVsi05-y2nEHTopC
secret = wecom.get('secret')
proxy = 'http://10.147.17.105:8888'

print(f'Bot ID: {bot_id}')
print(f'Proxy: {proxy}')

if not bot_id or not secret:
    print('ERROR: Missing WeCom credentials')
    sys.exit(1)

# 读取简报
report_path = r'C:\Users\XJH\DeepPredict\deeppredict_report_2026-03-24_1601.md'
report_text = open(report_path, encoding='utf-8').read()
print(f'Report length: {len(report_text)} chars')

handler = urllib.request.ProxyHandler({'https': proxy, 'http': proxy})
opener = urllib.request.build_opener(handler)

# 获取access_token
token_url = f'https://qyapi.weixin.qq.com/cgi-bin/gettoken?corpid={bot_id}&corpsecret={secret}'
try:
    resp = opener.open(token_url, timeout=10)
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
# 先尝试通过通讯录API查找
lookup_url = f'https://qyapi.weixin.qq.com/cgi-bin/user/simplelist?access_token={access_token}&department_id=1&fetch_child=1'
try:
    resp = opener.open(lookup_url, timeout=10)
    users_data = json.loads(resp.read())
    print('Users response:', json.dumps(users_data, ensure_ascii=False)[:500])
except Exception as e:
    print('User lookup error:', e)
    users_data = {'errcode': 0, 'userlist': []}

# 发送消息 - 查找许总监
user_id = None
if users_data.get('errcode') == 0:
    for user in users_data.get('userlist', []):
        name = user.get('name', '')
        if '许' in name or 'XU' in name.upper() or 'xudirector' in name.lower():
            user_id = user.get('userid')
            print(f'Found user: {name} -> {user_id}')
            break
        # 也检查账号
        if user.get('account', '') and ('xu' in user['account'].lower() or '许' in user['account']):
            user_id = user.get('userid')
            print(f'Found user by account: {user}')

# 如果找不到，尝试直接使用已知ID
if not user_id:
    # 从channels配置中找可能的userId
    user_id = wecom.get('userId') or wecom.get('defaultUser') or 'XuDirector'
    print(f'Using fallback userId: {user_id}')

# 分割消息（企业微信限制2048字节，约680中文字符）
MAX_LEN = 1800
messages = []
if len(report_text) <= MAX_LEN:
    messages = [report_text]
else:
    lines = report_text.split('\n')
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

print(f'Sending {len(messages)} message(s) to {user_id}')

# 发送每条消息
for i, msg in enumerate(messages):
    send_url = f'https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token={access_token}'
    payload = {
        'touser': user_id,
        'msgtype': 'text',
        'agentid': 0,  # agentid需要配置，此处传0让API自行处理
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
        print(f'Message {i+1} result:', result)
    except Exception as e:
        print(f'Message {i+1} error:', e)
