# -*- coding: utf-8 -*-
"""通过企业微信API发送消息 - 尝试多种认证方式"""
import json, urllib.request, sys
from pathlib import Path

config_path = Path.home() / ".openclaw" / "openclaw.json"
data = json.loads(config_path.read_text(encoding='utf-8'))
wecom = data.get('channels', {}).get('wecom', {})
bot_id = wecom.get('botId')
secret = wecom.get('secret')

print('botId:', bot_id)
print('secret:', secret[:8] + '...' if secret else 'None')

opener = urllib.request.build_opener()

# 读取简报
report_path = r'C:\Users\XJH\DeepPredict\deeppredict_report_2026-03-24_1601.md'
report_text = open(report_path, encoding='utf-8').read()

# 分割消息
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
print('Messages:', len(messages))

# 方法1: 尝试作为corp/app credentials获取token
def try_corp_api():
    print('\n=== Method 1: Corp API ===')
    try:
        token_url = 'https://qyapi.weixin.qq.com/cgi-bin/gettoken?corpid=' + bot_id + '&corpsecret=' + secret
        req = urllib.request.Request(token_url)
        resp = opener.open(req, timeout=15)
        result = json.loads(resp.read())
        print('Token result:', result)
        return result
    except Exception as e:
        print('Corp API failed:', e)
        return None

result = try_corp_api()
if result and result.get('access_token'):
    access_token = result['access_token']
    print('Got access_token:', access_token[:20] + '...')
    
    # 发送消息
    send_url = 'https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token=' + access_token
    # 查找用户
    user_id = 'XuDirector'
    
    for i, msg in enumerate(messages):
        payload = json.dumps({
            'touser': user_id,
            'msgtype': 'text',
            'agentid': 1000002,
            'text': {'content': msg}
        }, ensure_ascii=False).encode('utf-8')
        req = urllib.request.Request(send_url, data=payload, headers={'Content-Type': 'application/json'})
        try:
            resp = opener.open(req, timeout=15)
            r = json.loads(resp.read())
            print('Message', i+1, 'result:', r)
        except Exception as e:
            print('Message', i+1, 'error:', e)
else:
    print('\nCorp API failed, trying webhook...')
    # 方法2: Webhook
    webhook_url = 'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=' + bot_id
    for i, msg in enumerate(messages):
        payload = json.dumps({
            'msgtype': 'text',
            'text': {'content': msg}
        }, ensure_ascii=False).encode('utf-8')
        req = urllib.request.Request(webhook_url, data=payload, headers={'Content-Type': 'application/json'})
        try:
            resp = opener.open(req, timeout=15)
            r = json.loads(resp.read())
            print('Message', i+1, 'webhook result:', r)
        except Exception as e:
            print('Message', i+1, 'webhook error:', e)
