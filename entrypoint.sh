#!/usr/bin/env bash
set -euo pipefail
MODEL="/models/qwen/snapshots/${MODEL_REVISION:?}"
export VLLM_PLE_MMAP_DIR="$MODEL/ple-table"
[[ -f "$MODEL/config.json" && -f /runtime/draft-k10-512k/config.json ]] || { echo "Pinned model or prepared MTP draft is missing" >&2; exit 1; }
[[ "$(sha256sum /deployment/chat_template.jinja | cut -d ' ' -f 1)" == e57684bae4156211a55473c5a63be976a405a37ab5be5ae0e5abf1df5349c4b2 ]] || { echo "Template checksum mismatch" >&2; exit 1; }
read -r -a ALIASES <<< "${MODEL_ALIASES:?}"
ROPE=()
if (( MAX_MODEL_LEN > 262144 )); then
  ROPE=(--hf-overrides '{"text_config":{"max_position_embeddings":524288,"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":2.0,"original_max_position_embeddings":262144}}}')
fi
SPLIT='["vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::qwen3_8_flash_next_ple_short_conv","vllm::qwen3_8_flash_next_qsa_with_output","vllm::linear_attention","vllm::qwen_gdn_attention_core","vllm::qwen_gdn_attention_core_fused_norm_packed","vllm::sparse_attn_indexer","vllm::ple_mmap_lookup"]'
INDEXER=()
case "${INDEXER_KV_DTYPE:-bf16}" in
  bf16) ;;
  fp8) INDEXER=(--attention-config '{"indexer_kv_dtype":"fp8"}') ;;
  *) echo "INDEXER_KV_DTYPE must be bf16 or fp8" >&2; exit 1 ;;
esac
exec vllm serve "$MODEL" --served-model-name "${ALIASES[@]}" \
  --host 0.0.0.0 --port 8000 --load-format fastsafetensors \
  --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization 0.93 --kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES" \
  --kv-cache-dtype "$KV_CACHE_DTYPE" --enable-prefix-caching --enable-chunked-prefill \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  -cc.cudagraph_mode=PIECEWISE "-cc.splitting_ops=$SPLIT" \
  --no-enable-flashinfer-autotune --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --chat-template /deployment/chat_template.jinja "${ROPE[@]}" "${INDEXER[@]}" \
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS},\"model\":\"/runtime/draft-k10-512k\",\"max_model_len\":${MAX_MODEL_LEN},\"rejection_sample_method\":\"block\",\"draft_sample_method\":\"greedy\"}"
