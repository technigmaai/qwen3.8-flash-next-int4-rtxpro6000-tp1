"""Build-time checks; no CUDA context or model loading."""
import ast
from pathlib import Path

root = Path('/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next')
for relative in ('common/qsa_cache.py', 'nvidia/indexer_qsa.py', 'nvidia/ops/qsa.py'):
    ast.parse((root / relative).read_text())
source = (root / 'nvidia/indexer_qsa.py').read_text()
assert 'resolve_indexer_kv_dtype("bf16")' in source
assert 'dtype=self.indexer_dtype' in source
assert 'self.raw_key_cache = QSAKeyStateCache(\n            head_size=self.index_head_dim,\n            dtype=torch.bfloat16,' in source
scorer = (root / 'nvidia/ops/qsa.py').read_text()
assert 'QSA indexer Q and compressed K cache dtypes must match' in scorer
assert 'torch.ops._C_det.persistent_topk' in scorer
print('FP8 backport AST, BF16 fallback/raw ring, dtype guard and deterministic top-k: PASS')
