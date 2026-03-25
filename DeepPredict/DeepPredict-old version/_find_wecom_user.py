# -*- coding: utf-8 -*-
"""通过企业微信通讯录API查找许总监并发送消息"""
import json, urllib.request, sys
from pathlib import Path

# 读取openclaw配置
config_path = Path.home() / ".openclaw" / "openclaw.json"
data = json.loads(config_path.read_text(encoding='utf-8'))
wecom = data.get('channels', {}).get('wecom', {})
bot_id = wecom.get('botId')
secret = wecom.get('secret')
proxy = wecom.get('proxy', 'http://10.147.17.105:8888')

print(f'Bot ID: {bot_id}')
print(f'Proxy: {proxy}')

handler = urllib.request.ProxyHandler({'https': proxy, 'http': proxy})
opener = urllib.request.build_opener(handler)

# 获取access_token (使用Secret获取的token)
# 注意: 这里用的是"应用secret"而不是"商户secret"
# botId格式通常是: aibrIU_xxx  这不是标准corpid格式
# 需要找正确的corpid

# 尝试从环境变量或其他配置找
import os
for k, v in os.environ.items():
    if 'WECOM' in k.upper() or 'CORP' in k.upper():
        print(f'ENV: {k}={str(v)[:30]}')

# 尝试获取token看看需要什么格式
token_url = f'https://qyapi.weixin.qq.com/cgi-bin/gettoken?corpid={bot_id}&corpsecret={secret}'
try:
    resp = opener.open(token_url, timeout=10)
    token_data = json.loads(resp.read())
    print('Token response:', token_data)
    access_token = token_data.get('access_token')
    if not access_token:
        print('No access token, errcode:', token_data.get('errcode'))
        print('errmsg:', token_data.get('errmsg'))
except Exception as e:
    print('Token error:', e)
