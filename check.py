#!/usr/bin/env python3
"""Configuration/source/mount validation; idle GPU/port checks before startup."""
import argparse
import hashlib
import json
import socket
import subprocess
import sys
from pathlib import Path
from settings import ROOT, TEMPLATE_SHA256, load

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config-only', action='store_true', help='Skip occupied port/GPU checks; never launches a server.')
    options = p.parse_args()
    c = load()
    template = ROOT/'templates/chat_template.jinja'
    assert hashlib.sha256(template.read_bytes()).hexdigest() == TEMPLATE_SHA256, 'Template differs'
    model = Path(c['MODEL_HOST_DIR'])/'snapshots'/c['MODEL_REVISION']
    for name in ['config.json','tokenizer.json','model.safetensors.index.json','model_extra_tensors.safetensors']:
        assert (model/name).is_file(), f'Missing pinned model file: {name}'
    index = json.loads((model/'model.safetensors.index.json').read_text())
    for name in set(index['weight_map'].values()):
        assert (model/name).is_file(), f'Missing weight shard: {name}'
    assert (model/'ple-table').is_dir(), 'PLE table missing'
    for index in range(5,38):
        assert (model/'ple-table'/f'model-{index:05d}-of-00131.safetensors').is_file(), f'Missing PLE shard {index}'
    runtime = Path(c['RUNTIME_HOST_DIR'])
    draft = runtime/'draft-k10-512k'
    assert (draft/'.rtx-generated').read_text().strip() == c['MODEL_REVISION']
    config = json.loads((draft/'config.json').read_text())
    text = config.get('text_config',config)
    assert text['num_experts_per_tok'] == 10 and text['shared_expert_intermediate_size'] == 640
    assert text['max_position_embeddings'] == 524288 and text['rope_parameters']['factor'] == 2.0
    image = json.loads(subprocess.check_output(['docker','image','inspect',c['IMAGE']],text=True))[0]
    assert image['Architecture'] == 'amd64' and image['Os'] == 'linux'
    labels = image['Config'].get('Labels',{})
    assert '52830' in labels.get('rtx.parser.fix','') and '52805' in labels.get('rtx.xgrammar.fix','')
    if c['INDEXER_KV_DTYPE'] == 'fp8':
        assert '54890' in labels.get('rtx.indexer.fix', ''), 'FP8 indexer requested but image lacks the backport label'
    assert (runtime/'draft-scale/image_id').read_text().strip() == image['Id'], 'Regenerate scale module for this image'
    assert (runtime/'draft-scale/module_path').read_text().strip() == '/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/mtp.py'
    assert 'scale=_draft_scale' in (runtime/'draft-scale/mtp.py').read_text()
    subprocess.run(['docker','compose','--env-file',str(ROOT/'.env'),'-f',str(ROOT/'compose.yaml'),'config','--quiet'],check=True)
    print('Pinned files, template, independent draft, image fixes, MTP module and Compose config: PASS')
    if not options.config_only:
        with socket.socket() as s:
            s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            s.bind(('0.0.0.0',int(c['API_PORT'])))
        free = subprocess.check_output(['nvidia-smi','-i',c['GPU_DEVICE'],
            '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip()
        assert int(free) >= 85000, f'GPU is not idle: only {free} MiB free'
        print('Idle GPU and available port: PASS')
    print('Settings:',json.dumps({k:c[k] for k in ['API_PORT','MAX_MODEL_LEN','MAX_NUM_SEQS','MAX_NUM_BATCHED_TOKENS','KV_CACHE_MEMORY_BYTES','MTP_TOKENS']}))

if __name__ == '__main__':
    try:
        main()
    except (AssertionError,ValueError,OSError,subprocess.CalledProcessError) as e:
        sys.exit(str(e))
