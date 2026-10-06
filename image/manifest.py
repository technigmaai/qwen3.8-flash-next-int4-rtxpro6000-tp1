#!/usr/bin/env python3
"""Generate public source hashes; never include private settings/caches/logs."""
import argparse
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
p = argparse.ArgumentParser()
p.add_argument('--digest', required=True)
p.add_argument('--image', default='technigmaai/qwen3.8-flash-next-int4-rtxpro6000-tp1:v1.0.0-sm120-amd64-cu130')
p.add_argument('--output', default='manifest.json')
p.add_argument('--fp8-indexer', action='store_true')
args = p.parse_args()
if Path(args.output).name != args.output or not args.output.endswith('.json'):
    p.error('--output must be a JSON filename in the repository root')
files = {}
for path in sorted(root.rglob('*')):
    relative = path.relative_to(root)
    if not path.is_file() or any(x in relative.parts for x in ['.git','runtime','logs','data','runs','__pycache__','.hf-download']):
        continue
    if (path.name in ['.env','.DS_Store'] or path.suffix == '.pyc'
            or (path.parent == root and path.name.startswith('manifest') and path.suffix == '.json')):
        continue
    files[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
value = {'image':args.image,
    'registry_digest':args.digest,'platform':'linux/amd64','cuda_arch':'sm_120a',
    'source_pins': {'saren':'01c5914f322716b39fd71d5584ed800955582e65',
        'azampatti':'73c4fa07bddd5f4ae04c3384973ab5b4c4b61a80',
        'vllm_runtime':'8e685d198','vllm_parser_fix':'46638857fdbb30e0c232c9e8f9cb1ff6d6f545c3'},
    'files':files}
if args.fp8_indexer:
    value['source_pins']['vllm_fp8_indexer'] = '6fceb71365803ced6bee5f2a0972df05358f7075'
    value['profile'] = {'max_model_len':524288, 'max_num_seqs':8,
        'max_num_batched_tokens':8192, 'kv_cache_memory_bytes':15000000000,
        'kv_cache_dtype':'fp8_e4m3', 'indexer_kv_dtype':'fp8', 'mtp_tokens':3,
        'template_sha256':'e57684bae4156211a55473c5a63be976a405a37ab5be5ae0e5abf1df5349c4b2'}
(root/args.output).write_text(json.dumps(value,indent=2)+'\n')
print('Recorded',len(files),'public file hashes')
