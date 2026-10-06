#!/usr/bin/env python3
"""Generate independent draft config and image-specific scale module; no model edits."""
import json
import os
import struct
import subprocess
import sys
from pathlib import Path
from settings import ROOT, load

def main():
    c = load()
    source = Path(c['MODEL_HOST_DIR']) / 'snapshots' / c['MODEL_REVISION']
    runtime = Path(c['RUNTIME_HOST_DIR'])
    if not (source / 'config.json').is_file():
        raise ValueError('Pinned model not downloaded; see docs/DEPLOYMENT.md')
    draft = runtime / 'draft-k10-512k'
    marker = draft / '.rtx-generated'
    if draft.exists() and not marker.exists():
        raise ValueError('Refusing to overwrite an existing non-generated draft directory')
    config = json.loads((source / 'config.json').read_text())
    text = config.get('text_config', config)
    text['num_experts_per_tok'] = 10
    extra = source / 'model_extra_tensors.safetensors'
    with extra.open('rb') as f:
        length = struct.unpack('<Q', f.read(8))[0]
        if length > 100_000_000:
            raise ValueError('Unexpected safetensors header size')
        header = json.loads(f.read(length))
    rows = [v['shape'][0] for k,v in header.items() if k.endswith('mlp.shared_expert.gate_proj.weight')]
    if len(rows) != 1:
        raise ValueError('Expected one MTP shared expert tensor')
    text['shared_expert_intermediate_size'] = rows[0]
    text['max_position_embeddings'] = config['max_position_embeddings'] = 524288
    text['rope_parameters'] = {'mrope_interleaved': True, 'mrope_section': [11,11,10],
        'rope_type': 'yarn', 'rope_theta': 10000000, 'partial_rotary_factor': 0.25,
        'factor': 2.0, 'original_max_position_embeddings': 262144}
    config['rope_parameters'] = {k:v for k,v in text['rope_parameters'].items()
        if k not in ['factor','original_max_position_embeddings']}
    config['rope_parameters']['rope_type'] = 'default'
    draft.mkdir(parents=True, exist_ok=True)
    marker.write_text(c['MODEL_REVISION'] + '\n')
    for item in source.iterdir():
        if item.name == 'config.json':
            continue
        destination = draft / item.name
        target = f'/models/qwen/snapshots/{c["MODEL_REVISION"]}/{item.name}'
        if destination.is_symlink():
            if os.readlink(destination) != target:
                raise ValueError(f'Unexpected draft link: {destination}')
        elif destination.exists():
            raise ValueError(f'Refusing to overwrite {destination}')
        else:
            destination.symlink_to(target)
    (draft / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    subprocess.run([sys.executable,str(ROOT/'third_party/azampatti/draft_scale.py'),
        '--image',c['IMAGE'],'--out',str(runtime/'draft-scale/mtp.py')], check=True)
    print('Prepared independent YaRN 2x / top-k 10 / MTP shared width', rows[0])

if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as e:
        sys.exit(str(e))
