"""Literal .env configuration; never evaluate shell syntax or modify model data."""
import json
import re
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_ID = 'azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound'
REVISION = '1464274120d36a4d8fcaa934552334a7d83ce0fd'
DEFAULT_IMAGE = 'technigmaai/qwen3.8-flash-next-int4-rtxpro6000-tp1:v1.0.0-sm120-amd64-cu130'
TEMPLATE_SHA256 = 'e57684bae4156211a55473c5a63be976a405a37ab5be5ae0e5abf1df5349c4b2'

def load(path=None):
    path = Path(path or ROOT / '.env')
    if not path.is_file():
        raise ValueError('Copy .env.example to .env and edit host paths first.')
    values = {}
    for index, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        key, separator, raw = line.partition('=')
        if not separator or not re.fullmatch(r'[A-Z][A-Z0-9_]*', key):
            raise ValueError(f'Invalid .env line {index}')
        if key in values:
            raise ValueError(f'Duplicate key: {key}')
        parts = shlex.split(raw, comments=True)
        if len(parts) != 1 or any(c in parts[0] for c in ['$','`','\n','\r']):
            raise ValueError(f'{key}: use one literal value (quote spaces).')
        values[key] = parts[0]
    required = ['IMAGE','COMPOSE_PROJECT_NAME','CONTAINER_NAME','GPU_DEVICE',
        'HOST_UID','HOST_GID','API_PORT','MODEL_HOST_DIR','MODEL_REVISION',
        'RUNTIME_HOST_DIR','MODEL_ALIASES','MAX_MODEL_LEN','MAX_NUM_SEQS',
        'MAX_NUM_BATCHED_TOKENS','KV_CACHE_MEMORY_BYTES','KV_CACHE_DTYPE',
        'MTP_TOKENS','DRAFT_LOGIT_SCALE']
    missing = set(required) - values.keys()
    if missing:
        raise ValueError(f'Missing settings: {sorted(missing)}')
    values.setdefault('INDEXER_KV_DTYPE', 'bf16')
    if values['INDEXER_KV_DTYPE'] not in ('bf16', 'fp8'):
        raise ValueError('INDEXER_KV_DTYPE must be bf16 or fp8.')
    for key in ['GPU_DEVICE','HOST_UID','HOST_GID','API_PORT','MAX_MODEL_LEN',
                'MAX_NUM_SEQS','MAX_NUM_BATCHED_TOKENS','KV_CACHE_MEMORY_BYTES','MTP_TOKENS']:
        if not values[key].isdigit():
            raise ValueError(f'{key}: expected a nonnegative integer')
    for key in ['MODEL_HOST_DIR','RUNTIME_HOST_DIR']:
        p = Path(values[key])
        if not p.is_absolute() or '..' in p.parts or '/your-user/' in str(p):
            raise ValueError(f'{key}: replace placeholder with a literal absolute path')
        if any(c in str(p) for c in [',',':','~']):
            raise ValueError(f'{key}: unsupported mount path character')
    model = Path(values['MODEL_HOST_DIR']).resolve()
    runtime = Path(values['RUNTIME_HOST_DIR']).resolve()
    if model == runtime or model in runtime.parents or runtime in model.parents or len(runtime.parts)<4:
        raise ValueError('Use a dedicated runtime directory separate from model data.')
    if values['MODEL_REVISION'] != REVISION:
        raise ValueError('This release qualifies only the pinned model revision.')
    if not 1 <= int(values['API_PORT']) <= 65535:
        raise ValueError('API_PORT out of range')
    if not 1 <= int(values['MAX_MODEL_LEN']) <= 524288:
        raise ValueError('Only context lengths up to the validated YaRN 2x profile are supported.')
    if not 1 <= int(values['MAX_NUM_SEQS']) <= 8:
        raise ValueError('This release qualifies at most eight sequence slots.')
    if int(values['MAX_NUM_BATCHED_TOKENS']) < int(values['MAX_NUM_SEQS']):
        raise ValueError('Batch token budget must accommodate sequence slots.')
    if int(values['KV_CACHE_MEMORY_BYTES']) <= 0:
        raise ValueError('KV cache size must be positive')
    if values['KV_CACHE_DTYPE'] != 'fp8_e4m3' or values['MTP_TOKENS'] != '3' or values['DRAFT_LOGIT_SCALE'] != '2':
        raise ValueError('This package qualifies FP8 E4M3, MTP3 and draft scale 2.')
    if not {MODEL_ID,'rtx'}.issubset(values['MODEL_ALIASES'].split()):
        raise ValueError('Retain the full model ID and rtx aliases.')
    for key in ['COMPOSE_PROJECT_NAME','CONTAINER_NAME']:
        if not re.fullmatch(r'[a-z0-9][a-z0-9_.-]*', values[key]):
            raise ValueError(f'Unsafe {key}')
    return values

if __name__ == '__main__':
    try:
        print(json.dumps(load(), indent=2))
    except ValueError as e:
        sys.exit(str(e))
