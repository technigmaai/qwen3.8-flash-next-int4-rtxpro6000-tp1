"""Eight real SSE streams, mixed grammars and optional near-512k chat request."""
import concurrent.futures
import json
import sys
import threading
import time
import urllib.request

BASE = 'http://127.0.0.1:8000'


def post(path, payload):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    return urllib.request.urlopen(req, timeout=900)


def stream(payload):
    payload = {'model': 'rtx', 'temperature': 0, 'max_tokens': 256,
               'chat_template_kwargs': {'enable_thinking': False},
               **payload, 'stream': True, 'stream_options': {'include_usage': True}}
    started = time.monotonic()
    text, finish, calls, done, usage = '', None, {}, False, None
    with post('/v1/chat/completions', payload) as response:
        for line in response:
            if not line.startswith(b'data: '):
                continue
            raw = line[6:].strip()
            if raw == b'[DONE]':
                done = True
                break
            chunk = json.loads(raw)
            assert 'error' not in chunk, chunk
            usage = chunk.get('usage') or usage
            for choice in chunk.get('choices', []):
                finish = choice.get('finish_reason') or finish
                delta = choice.get('delta', {})
                text += delta.get('content') or ''
                for tc in delta.get('tool_calls') or []:
                    call = calls.setdefault(tc['index'], {'name': '', 'arguments': ''})
                    f = tc.get('function', {})
                    call['name'] += f.get('name') or ''
                    call['arguments'] += f.get('arguments') or ''
    assert done and finish is not None, {'done': done, 'finish': finish, 'text': text[-200:]}
    return {'text': text, 'calls': calls, 'usage': usage, 'finish': finish,
            'seconds': round(time.monotonic() - started, 2)}


def messages(text):
    return [{'role': 'user', 'content': text}]


def eight_streams():
    barrier = threading.Barrier(8)

    def work(index):
        barrier.wait()
        result = stream({'messages': messages(f'Write a detailed tutorial on Python generators with many code examples, topic {index}. Continue until the token limit.'),
                         'max_tokens': 512})
        assert result['text'] and result['usage']['completion_tokens'] >= 128, result
        print('STREAM', index, {k: v for k, v in result.items() if k not in ['text', 'calls']}, flush=True)
        return result

    peak = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(work, i) for i in range(8)]
        while not all(f.done() for f in futures):
            with urllib.request.urlopen(BASE + '/metrics', timeout=10) as response:
                for line in response.read().decode().splitlines():
                    if line.startswith('vllm:num_requests_running{'):
                        peak = max(peak, int(float(line.rsplit(' ', 1)[1])))
            time.sleep(0.5)
        results = [f.result() for f in futures]
    assert peak == 8, peak
    print('EIGHT STREAMS PASS; observed running requests:', peak, flush=True)


def mixed(index):
    if index % 2 == 0:
        result = stream({'messages': messages('Return ready true in the requested JSON schema.'),
            'temperature': 0.6 if index % 4 == 0 else 0,
            'response_format': {'type': 'json_schema', 'json_schema': {
                'name': 'ready', 'strict': True, 'schema': {
                    'type': 'object', 'properties': {'ready': {'type': 'boolean'}},
                    'required': ['ready'], 'additionalProperties': False}}}})
        assert json.loads(result['text']) == {'ready': True}, result
    else:
        result = stream({'messages': messages(f'Call echo with code TEST_{index}. This is only a parser test.'),
            'temperature': 0.6 if index % 4 == 1 else 0,
            'tools': [{'type': 'function', 'function': {'name': 'echo', 'description': 'Mock; never executed.',
                'parameters': {'type': 'object', 'properties': {'code': {'type': 'string'}},
                               'required': ['code'], 'additionalProperties': False}}}],
            'tool_choice': {'type': 'function', 'function': {'name': 'echo'}}})
        assert len(result['calls']) == 1, result
        call = next(iter(result['calls'].values()))
        assert call['name'] == 'echo' and json.loads(call['arguments']) == {'code': f'TEST_{index}'}, result
    return result


def long_context():
    phrase = 'This is irrelevant background reference material; it contains no secret identifier.\n'
    def build(n):
        return messages('The secret identifier near the beginning is ORCHID_173929.\n<background>\n'
                        + phrase * n + '\n</background>\nReturn only the secret identifier that appeared before the background.')
    def count(n):
        with post('/tokenize', {'model': 'rtx', 'messages': build(n),
                               'chat_template_kwargs': {'enable_thinking': False}}) as response:
            return json.load(response)['count']
    small, large = count(100), count(1000)
    repeats = int((523000 - small) / ((large - small) / 900)) + 100
    actual = count(repeats)
    assert 520000 <= actual <= 524000, actual
    print('LONG CHAT starting:', actual, 'prompt tokens; MTP3 stays enabled', flush=True)
    result = stream({'messages': build(repeats), 'max_tokens': 128})
    print('LONG CHAT result:', json.dumps(result), flush=True)
    assert result['usage']['prompt_tokens'] >= 520000, result
    assert result['text'].strip() == 'ORCHID_173929', result
    print('NEAR-512K CHAT PASS (single synthetic retrieval; not a full quality benchmark)', flush=True)


if '--long' in sys.argv:
    long_context()
else:
    eight_streams()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(mixed, range(32)))
    print('MIXED STRUCTURED SSE PASS: 16 JSON + 16 forced tools, all finish_reason and DONE', flush=True)
