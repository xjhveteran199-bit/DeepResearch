# -*- coding: utf-8 -*-
"""直接通过企业微信API发送消息"""
import json, urllib.request, sys, os
from pathlib import Path

# 读取openclaw配置获取wecom凭证
config_path = Path.home() / ".openclaw" / "openclaw.json"
data = json.loads(config_path.read_text(encoding='utf-8'))
wecom = data.get('channels', {}).get('wecom', {})
bot_id = wecom.get('botId')
secret = wecom.get('secret')
proxy = wecom.get('proxy', 'http://10.147.17.105:8888')  # 默认代理

print(f'Bot ID: {bot_id}')
print(f'Secret: {\"SET\" if secret else \"NOT SET\"}')
print(f'Proxy: {proxy}')

if not bot_id or not secret:
    print('ERROR: Missing WeCom credentials')
    sys.exit(1)

# 读取简报
report_path = r'C:\Users\XJH\DeepPredict\_report_max_temp.md'
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

# 查找许总监的userid
# 先尝试通过名字获取通讯录
lookup_url = f'https://qyapi.weixin.qq.com/cgi-bin/user/simplelist?access_token={access_token}&department_id=1&fetch_child=1'
try:
    resp = opener.open(lookup_url, timeout=10)
    users_data = json.loads(resp.read())
    print('User list response:', users_data)
except Exception as e:
    print('ERROR getting user list:', e)
    users_data = {'errmsg': str(e)}

# 尝试发送消息，先用userid查询
# 许总监 - 尝试多种可能的userid
user_ids_to_try = ['XuZongjian', 'xuzongjian', '许总监']
# 搜索用户
search_url = f'https://qyapi.weixin.qq.com/cgi-bin/user/simplelist?access_token={access_token}&department_id=1'
try:
    resp = opener.open(search_url, timeout=10)
    dept_users = json.loads(resp.read())
    print('Dept users:', dept_users)
    if dept_users.get('errcode') == 0:
        for u in dept_users.get('uselist', []):
            name = u.get('name', '')
            if '许' in name or '总监' in name or 'xu' in name.lower():
                print(f'Found: {u}')
                user_ids_to_try.insert(0, u.get('userid'))
except Exception as e:
    print('Search error:', e)

# 发送消息（尝试不同的userid）
send_url_base = f'https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token={access_token}'

for uid in user_ids_to_try[:3]:
    payload = {
        'touser': uid,
        'msgtype': 'text',
        'agentid': 0,  # 使用0试试
        'text': {'content': report_text},
        'safe': 0
    }
    req = urllib.request.Request(
        send_url_base,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    try:
        resp = opener.open(req, timeout=10)
        result = json.loads(resp.read())
        print(f'Send to {uid}:', result)
        if result.get('errcode') == 0:
            print(f'SUCCESS sent to {uid}')
            break
        else:
            print(f'Failed to {uid}:', result.get('errmsg'))
    except Exception as e:
        print(f'Error sending to {uid}:', e)
