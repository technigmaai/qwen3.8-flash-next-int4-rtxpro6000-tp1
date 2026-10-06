#!/usr/bin/env python3
"""Generate public source hashes; never include private settings/caches/logs."""
import argparse
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
p = argparse.ArgumentParser()
p.add_argument('--digest', required=True)
args = p.parse_args()
files = {}
for path in sorted(root.rglob('*')):
    relative = path.relative_to(root)
    if not path.is_file() or any(x in relative.parts for x in ['.git','runtime','logs','data','runs','__pycache__','.hf-download']):
        continue
    if path.name in ['.env','manifest.json','.DS_Store'] or path.suffix == '.pyc':
        continue
    files[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
value = {'image':'technigmaai/qwen3.8-flash-next-int4-rtxpro6000-tp1:v1.0.0-sm120-amd64-cu130',
    'registry_digest':args.digest,'platform':'linux/amd64','cuda_arch':'sm_120a',
    'source_pins': {'saren':'01c5914f322716b39fd71d5584ed800955582e65',
        'azampatti':'73c4fa07bddd5f4ae04c3384973ab5b4c4b61a80',
        'vllm_runtime':'8e685d198','vllm_parser_fix':'46638857fdbb30e0c232c9e8f9cb1ff6d6f545c3'},
    'files':files}
(root/'manifest.json').write_text(json.dumps(value,indent=2)+'\n')
print('Recorded',len(files),'public file hashes')
