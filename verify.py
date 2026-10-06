#!/usr/bin/env python3
"""Run published API probes against configured port, without executing model tools."""
import sys
from settings import ROOT, load

c = load()
base = 'http://127.0.0.1:' + c['API_PORT']
for name in ['verify_tool_enforcement.py','verify_api.py','verify_capacity.py']:
    path = ROOT/'tests'/name
    # Tests retained verbatim from deployment; override only the host API port.
    source = path.read_text().replace("BASE = 'http://127.0.0.1:8000'",'BASE = ' + repr(base))
    print('RUNNING',name,flush=True)
    exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path),'sys':sys})
