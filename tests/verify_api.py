"""API correctness and short throughput/concurrency checks; no tools are executed."""
import ast
import concurrent.futures
import json
import time
import urllib.request

BASE = 'http://127.0.0.1:8000'
MODELS = ['azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound', 'rtx']


def request(path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data, headers={'Content-Type': 'application/json'})
    return urllib.request.urlopen(req, timeout=240)


def chat(prompt, model='rtx', max_tokens=128, **extra):
    payload = {'model': model, 'temperature': 0, 'max_tokens': max_tokens,
               'chat_template_kwargs': {'enable_thinking': False},
               'messages': [{'role': 'user', 'content': prompt}], **extra}
    started = time.monotonic()
    with request('/v1/chat/completions', payload) as response:
        result = json.load(response)
    return result, time.monotonic() - started


with request('/health') as response:
    assert response.status == 200
with request('/v1/models') as response:
    listed = [entry['id'] for entry in json.load(response)['data']]
assert listed == MODELS, listed
print('Health and both model IDs: PASS', flush=True)
for model in MODELS:
    result, elapsed = chat('Reply with exactly UPGRADE_OK.', model=model)
    content = result['choices'][0]['message']['content']
    assert content and content.strip() == 'UPGRADE_OK', result
    assert result['model'] in MODELS, result
    print(f'Chat {model}: PASS ({elapsed:.2f}s)', flush=True)

result, elapsed = chat('Calculate 17 + 29. The final answer must be only the integer.',
    max_tokens=512, chat_template_kwargs={'enable_thinking': True}, reasoning_effort='medium')
message = result['choices'][0]['message']
assert message['content'].strip() == '46', result
assert message.get('reasoning') or message.get('reasoning_content'), result
print('Thinking template + reasoning parser: PASS', flush=True)

result, elapsed = chat('Call echo with code AUTO_TOOL_OK. This is only a mock parser test.',
    max_tokens=256,
    tools=[{'type': 'function', 'function': {'name': 'echo', 'description': 'Mock; never executed.',
        'parameters': {'type': 'object', 'properties': {'code': {'type': 'string'}},
                       'required': ['code'], 'additionalProperties': False}}}], tool_choice='auto')
calls = result['choices'][0]['message'].get('tool_calls') or []
assert len(calls) == 1, result
assert calls[0]['function']['name'] == 'echo', result
assert json.loads(calls[0]['function']['arguments']) == {'code': 'AUTO_TOOL_OK'}, result
print('Automatic mock tool selection: PASS', flush=True)

result, elapsed = chat('Return a JSON object with ready set to true.',
    response_format={'type': 'json_schema', 'json_schema': {
        'name': 'ready', 'strict': True,
        'schema': {'type': 'object', 'properties': {'ready': {'type': 'boolean'}},
                   'required': ['ready'], 'additionalProperties': False}}})
assert json.loads(result['choices'][0]['message']['content']) == {'ready': True}, result
print('Structured JSON schema: PASS', flush=True)

expected = {'path': 'greet.py', 'content': 'def greet(name):\n    return f"Hello, {name}!"\n'}
payload = {'model': 'rtx', 'temperature': 0, 'max_tokens': 256, 'stream': True,
    'chat_template_kwargs': {'enable_thinking': False},
    'messages': [{'role': 'user', 'content': 'Use write_file to create greet.py containing this exact Python code (actual line breaks, not literal backslash-n):\n```python\n' + expected['content'] + '```'}],
    'tools': [{'type': 'function', 'function': {'name': 'write_file', 'description': 'A mock parser test; never executed.',
        'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}, 'content': {'type': 'string'}},
                       'required': ['path', 'content'], 'additionalProperties': False}}}],
    'tool_choice': {'type': 'function', 'function': {'name': 'write_file'}}}
calls = {}
with request('/v1/chat/completions', payload) as response:
    for line in response:
        if not line.startswith(b'data: '):
            continue
        raw = line[6:].strip()
        if raw == b'[DONE]':
            break
        for choice in json.loads(raw).get('choices', []):
            for delta in choice.get('delta', {}).get('tool_calls', []):
                call = calls.setdefault(delta['index'], {'name': '', 'arguments': ''})
                function = delta.get('function', {})
                call['name'] += function.get('name') or ''
                call['arguments'] += function.get('arguments') or ''
assert len(calls) == 1, calls
call = next(iter(calls.values()))
arguments = json.loads(call['arguments'])
assert call['name'] == 'write_file' and arguments['path'] == expected['path'], calls
assert arguments['content'].rstrip('\n') == expected['content'].rstrip('\n'), calls
ast.parse(arguments['content'])
print('Streamed tool call with quotes and newlines: PASS (mock not executed)', flush=True)

prompt = 'Write a detailed tutorial about Python generators with several code examples. Keep writing until the token limit.'
def throughput(index):
    result, elapsed = chat(prompt + f' Use example topic number {index}.', max_tokens=256)
    message = result['choices'][0]['message']
    assert message.get('content') and not message.get('tool_calls'), result
    tokens = result['usage']['completion_tokens']
    assert tokens >= 64, result
    return {'tokens': tokens, 'seconds': round(elapsed, 3), 'end_to_end_tok_s': round(tokens / elapsed, 2)}

for index in range(2):
    print('SINGLE', json.dumps(throughput(index)), flush=True)
started = time.monotonic()
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    concurrent_results = list(pool.map(throughput, [2, 3]))
elapsed = time.monotonic() - started
print('CONCURRENT', json.dumps({'requests': concurrent_results,
    'aggregate_end_to_end_tok_s': round(sum(r['tokens'] for r in concurrent_results) / elapsed, 2)}), flush=True)
for index in range(6):
    result, elapsed = chat(f'Calculate {31 + index} + 17. Reply with only the integer.')
    assert result['choices'][0]['message']['content'].strip() == str(48 + index), result
print('Repeated arithmetic: PASS (6 requests)', flush=True)
background = 'This paragraph is irrelevant reference material for a cache test. ' * 400
for index in range(3):
    result, elapsed = chat('Ignore the background and answer only the arithmetic question at the end.\n<background>\n'
        + background + f'\n</background>\nCalculate {70 + index} + 17. Reply with only the integer.')
    assert result['choices'][0]['message']['content'].strip() == str(87 + index), result
    print('LONG_PREFIX', json.dumps({'prompt_tokens': result['usage']['prompt_tokens'],
        'cached_tokens': (result['usage'].get('prompt_tokens_details') or {}).get('cached_tokens'),
        'seconds': round(elapsed, 3)}), flush=True)
print('Long shared-prefix arithmetic: PASS (3 requests)', flush=True)
print('ALL API CHECKS PASSED', flush=True)
