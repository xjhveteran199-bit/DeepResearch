# -*- coding: utf-8 -*-
"""尝试发送企业微信消息"""
import subprocess, sys, os

report_text = open(r'C:\Users\XJH\DeepPredict\_report_max_temp.md', encoding='utf-8').read()
script = r'C:\Users\XJH\.agents\skills\wecom-notify\scripts\send_wecom.py'

# 尝试设置环境变量（可能WECOM配置在别处）
env = os.environ.copy()

# 尝试发送
r = subprocess.run(
    [sys.executable, script, report_text],
    capture_output=True, text=True,
    timeout=30, env=env
)
print('Return code:', r.returncode)
print('STDOUT:', r.stdout[:500])
print('STDERR:', r.stderr[:500])
