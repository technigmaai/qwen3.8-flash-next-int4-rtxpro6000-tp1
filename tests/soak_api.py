"""Bounded multi-client correctness soak. No client tools are executed."""
import concurrent.futures
import json
import os
import time
import urllib.request

workers = int(os.environ.get('WORKERS', '2'))
duration = float(os.environ.get('SOAK_SECONDS', '120'))
assert 1 <= workers <= 8 and 1 <= duration <= 600
deadline = time.monotonic() + duration
started = time.monotonic()


def worker(number):
    count = 0
    while time.monotonic() < deadline:
        value = 100 + number * 1000 + count
        payload = {'model': 'rtx', 'temperature': 0, 'max_tokens': 64,
                   'chat_template_kwargs': {'enable_thinking': False},
                   'messages': [{'role': 'user', 'content': f'Calculate {value} + 17. Reply with only the integer.'}]}
        req = urllib.request.Request('http://127.0.0.1:8000/v1/chat/completions', data=json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=240) as response:
            result = json.load(response)
        assert result['choices'][0]['message']['content'].strip() == str(value + 17), result
        count += 1
        if count % 100 == 0:
            print(f'worker {number}: {count} correct responses, elapsed {time.monotonic() - started:.1f}s', flush=True)
    return count


with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
    counts = list(pool.map(worker, range(workers)))
print('SOAK PASS', json.dumps({'seconds': round(time.monotonic() - started, 2), 'correct_responses': sum(counts),
                              'workers': workers}), flush=True)
