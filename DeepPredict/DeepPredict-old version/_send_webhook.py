# -*- coding: utf-8 -*-
"""通过企业微信Webhook发送DeepPredict简报给许总监"""
import json, urllib.request, sys
from pathlib import Path

config_path = Path.home() / ".openclaw" / "openclaw.json"
data = json.loads(config_path.read_text(encoding='utf-8'))
wecom = data.get('channels', {}).get('wecom', {})
webhook_key = wecom.get('botId')
proxy = wecom.get('proxy', 'http://10.147.17.105:8888')

print('Webhook key:', webhook_key)
print('Proxy:', proxy)

if not webhook_key:
    print('ERROR: Missing webhook key')
    sys.exit(1)

report_path = r'C:\Users\XJH\DeepPredict\deeppredict_report_2026-03-24_1601.md'
report_text = open(report_path, encoding='utf-8').read()
print('Report length:', len(report_text), 'chars')

try:
    handler = urllib.request.ProxyHandler({'https': proxy, 'http': proxy})
    opener = urllib.request.build_opener(handler)
    test_url = 'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=' + webhook_key
    test_payload = json.dumps({'msgtype': 'text', 'text': {'content': 'test'}}, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request(test_url, data=test_payload, headers={'Content-Type': 'application/json'})
    resp = opener.open(req, timeout=10)
    test_result = json.loads(resp.read())
    print('Proxy test result:', test_result)
except Exception as e:
    print('Proxy test failed:', e, ', trying direct')
    opener = urllib.request.build_opener()

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

print('Sending', len(messages), 'message(s) via webhook')

send_url = 'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=' + webhook_key

for i, msg in enumerate(messages):
    payload = json.dumps({'msgtype': 'text', 'text': {'content': msg}}, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request(send_url, data=payload, headers={'Content-Type': 'application/json'})
    try:
        resp = opener.open(req, timeout=15)
        result = json.loads(resp.read())
        err = result.get('errcode')
        errmsg = result.get('errmsg', '')[:100]
        print('Message', i+1, ': errcode=', err, ', errmsg=', errmsg)
    except Exception as e:
        print('Message', i+1, 'error:', e)
