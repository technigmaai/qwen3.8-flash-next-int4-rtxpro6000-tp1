# FP8 indexer backport tests

`test_pre_indexer_upstream.py` is the Apache-2.0 vLLM test from merged PR #54890,
revision `6fceb71365803ced6bee5f2a0972df05358f7075`, with only the model import
namespace adapted from `qwen4_exp` to the installed preview's `qwen3_8_flash_next`.
Source: https://github.com/vllm-project/vllm/blob/6fceb71365803ced6bee5f2a0972df05358f7075/tests/models/qwen4_exp/test_qsa_pre_indexer.py

`test_scoring.py` checks the older combined scorer against an FP32 reference
using the same quantized inputs. It covers BF16 fallback, FP8, one decode row,
mixed request mappings, multi-token/prefill batches, and the retained SM120
deterministic top-k at the real 2048-token budget (512 compressed groups).
Different FP8/BF16 selected indices are not themselves a failure: quantization
can change scores and ties. End-to-end retrieval checks are also required.

Run in the candidate image with the GPU available (do not run a second model):

```sh
docker run --rm --gpus device=0 --shm-size 2g \
  -e VLLM_QSA_DET_TOPK=1 -e VLLM_QSA_DET_LIB=/opt/llm/kernel-det/_C_det.so \
  --entrypoint python3 \
  technigmaai/qwen3.8-flash-next-int4-rtxpro6000-tp1:v1.0.1-rc.1-sm120-amd64-cu130 \
  -m pytest -q /opt/llm/tests/fp8_indexer
```
