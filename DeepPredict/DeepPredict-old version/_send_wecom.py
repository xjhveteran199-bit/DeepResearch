import subprocess, sys
result = subprocess.run(
    ['python', r'C:\Users\XJH\.agents\skills\wecom-notify\scripts\send_wecom.py'],
    capture_output=True, text=True
)
print('STDOUT:', result.stdout[:500])
print('STDERR:', result.stderr[:500])
