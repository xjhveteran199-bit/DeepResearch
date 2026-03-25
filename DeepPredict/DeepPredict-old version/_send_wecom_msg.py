# -*- coding: utf-8 -*-
"""发送简报给许总监"""
import subprocess, sys, os

# 读取简报
report_path = r'C:\Users\XJH\DeepPredict\_report_max_temp.md'
report = open(report_path, encoding='utf-8').read()
print(f'简报长度: {len(report)} 字符')

# 发送
script = r'C:\Users\XJH\.agents\skills\wecom-notify\scripts\send_wecom.py'

# 尝试发到许总监
for name in ['许总监', 'XuZongjian', 'xuzongjian']:
    print(f'\n尝试发送到: {name}')
    r = subprocess.run(
        [sys.executable, script, report, '--to', name],
        capture_output=True, text=True,
        timeout=30
    )
    print('STDOUT:', r.stdout[:500])
    print('STDERR:', r.stderr[:500])
    if r.returncode == 0:
        print(f'成功发送到 {name}')
        break
