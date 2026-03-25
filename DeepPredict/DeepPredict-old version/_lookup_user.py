import subprocess, json

result = subprocess.run(
    ['python', r'C:\Users\XJH\.openclaw\extensions\wecom-openclaw-plugin\skills\wecom-contact-lookup\wecom_mcp', 'call', 'contact', 'get_userlist', '{}'],
    capture_output=True, text=True,
    cwd=r'C:\Users\XJH\.openclaw\extensions\wecom-openclaw-plugin'
)
print('STDOUT:', result.stdout[:2000])
print('STDERR:', result.stderr[:500])
print('RC:', result.returncode)
