"""Adversarial required/named API probes; mock calls are never executed."""
import json
import time
import urllib.request

BASE = 'http://127.0.0.1:8000'
TOOLS = [{'type': 'function', 'function': {'name': name,
    'description': 'Mock capability probe. This tool is never executed.',
    'parameters': {'type': 'object', 'properties': {'value': {'type': 'string'}},
                   'required': ['value'], 'additionalProperties': False}}}
    for name in ['probe_ping', 'other_tool']]

def probe(kwargs, choice, stream, max_tokens=512, tools=None):
    payload = {'model': 'rtx', 'temperature': 0, 'max_tokens': max_tokens,
        'messages': [{'role': 'user', 'content': 'Reply with the single word OK. Do not call any tools.'}],
        'tools': tools or TOOLS, 'tool_choice': choice, 'stream': stream,
        'chat_template_kwargs': kwargs}
    req = urllib.request.Request(BASE + '/v1/chat/completions',
        data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=180) as response:
        if not stream:
            result = json.load(response)
            first = result['choices'][0]
            calls = [c['function'] for c in first['message'].get('tool_calls') or []]
            finish = first['finish_reason']
        else:
            collected, finish, done = {}, None, False
            for line in response:
                if not line.startswith(b'data: '):
                    continue
                raw = line[6:].strip()
                if raw == b'[DONE]':
                    done = True
                    break
                for item in json.loads(raw).get('choices', []):
                    finish = item.get('finish_reason') or finish
                    for delta in item.get('delta', {}).get('tool_calls') or []:
                        call = collected.setdefault(delta['index'], {'name': '', 'arguments': ''})
                        for field in ['name', 'arguments']:
                            call[field] += delta.get('function', {}).get(field) or ''
            assert done, 'Missing SSE [DONE]'
            calls = list(collected.values())
    # This pinned server deliberately preserves engine "stop" for named choices
    # (serving.py:740 and :1047). Enforcement is the actual call/name/arguments,
    # not just the finish label; required must still report "tool_calls".
    allowed_finish = {'stop', 'tool_calls'} if isinstance(choice, dict) else {'tool_calls'}
    assert calls and finish in allowed_finish, (kwargs, choice, stream, calls, finish)
    permitted = {t['function']['name'] for t in tools or TOOLS}
    for call in calls:
        assert call['name'] in permitted, calls
        if isinstance(choice, dict):
            assert call['name'] == choice['function']['name'], calls
        args = json.loads(call['arguments'])
        assert set(args) == {'value'} and isinstance(args['value'], str), calls
    print(json.dumps({'kwargs': kwargs, 'choice': choice, 'stream': stream,
        'tools': calls, 'finish_reason': finish,
        'seconds': round(time.monotonic() - started, 2), 'status': 'PASS'}), flush=True)

# Exact TC-45 capability probe shape and originally used kwargs/budget.
probe({'thinking': True, 'reasoning_effort': 'xhigh'}, 'required', False,
      max_tokens=256, tools=TOOLS[:1])
for kwargs in [{'enable_thinking': False}, {'enable_thinking': True},
               {'thinking': True, 'reasoning_effort': 'high'}]:
    for choice in ['required', {'type': 'function', 'function': {'name': 'probe_ping'}}]:
        for stream in [False, True]:
            probe(kwargs, choice, stream)
print('ALL LIVE TOOL ENFORCEMENT CHECKS PASSED (no tools executed)', flush=True)
