"""Verify the portable entrypoint's exact engine config without loading weights."""
import json
import os
import subprocess
import tempfile
from pathlib import Path
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.entrypoints.cli.serve import ServeSubcommand
from vllm.engine.arg_utils import AsyncEngineArgs

# A throwaway command-capture shim lives only in this temporary test container.
# The entrypoint checks read-only model/template paths, then invokes this shim,
# not a real server. No API port is bound and no model tensors are loaded.
with tempfile.TemporaryDirectory() as directory:
    shim = Path(directory)/'vllm'
    shim.write_text('#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
    shim.chmod(0o755)
    env = dict(os.environ, PATH=directory+':'+os.environ['PATH'])
    command = json.loads(subprocess.check_output(['bash','/deployment/entrypoint.sh'],env=env,text=True))
parser = FlexibleArgumentParser()
ServeSubcommand().subparser_init(parser.add_subparsers(dest='subcommand'))
args = parser.parse_args(command)
args.model = args.model_tag
engine = AsyncEngineArgs.from_cli_args(args).create_engine_config()
assert engine.model_config.max_model_len == 524288
assert engine.scheduler_config.max_num_seqs == 8
assert engine.scheduler_config.max_num_batched_tokens == 8192
assert engine.cache_config.cache_dtype == 'fp8_e4m3'
assert engine.cache_config.kv_cache_memory_bytes == 13000000000
assert engine.attention_config.resolve_indexer_kv_dtype('bf16') == os.environ.get('INDEXER_KV_DTYPE', 'bf16')
assert engine.speculative_config.num_speculative_tokens == 3
assert args.served_model_name == ['azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound','rtx']
for name,config in [('target',engine.model_config),('draft',engine.speculative_config.draft_model_config)]:
    assert config.max_model_len == 524288
    assert config.hf_text_config.rope_parameters['rope_type'] == 'yarn'
    assert config.hf_text_config.rope_parameters['factor'] == 2.0
    print(name,'context/YaRN: PASS',flush=True)
print('Portable entrypoint 512k / SEQS8 / batch8192 / 13GB FP8 / MTP3 / aliases: PASS',flush=True)
