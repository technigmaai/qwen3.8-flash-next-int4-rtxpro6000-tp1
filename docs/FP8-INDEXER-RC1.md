# FP8 QSA indexer: v1.0.1-rc.1

Status: published as **v1.0.1-rc.1**, retained as the active experimental RTX
service after functional qualification. **Not promoted to stable.**
An isolated follow-up memory comparison passed and did not
reproduce the earlier mixed-traffic headroom loss. These are bounded tests,
not multi-day production or exhaustive model-quality qualification.

## Scope

Backport of merged vLLM PR #54890, pinned head
`6fceb71365803ced6bee5f2a0972df05358f7075`, adapted to preview `8e685d198`.
Changes are restricted to the compressed indexer cache, query dtype and dtype
guards/backend declarations. The old combined scorer, cache layout, SM120
deterministic top-k and existing tuning are preserved. All 13 initialized
indexers (12 target + 1 MTP) log FP8 compressed keys and BF16 raw rings.

Candidate tag: `v1.0.1-rc.1-sm120-amd64-cu130`.
Local OCI image-config digest: `sha256:f3a96e6070aab99286783d4f118e2114ff80d4035bf47d5f9cbe39c0e93a523e`.
Published Docker Hub manifest-list digest:
`sha256:7a53dbe5398e50183a87fa91732ba6d526bff1c70bd889e7b31444fa9fac4fe9`.
The published image matches the qualified local digest; no rebuild or stable
retag was performed. See [the release manifest](../manifest-v1.0.1-rc.1.json).

## Host trial, 2026-10-06

RTX PRO 6000 Blackwell 96 GB, TP1; same pinned AutoRound checkpoint, original
Froggeric template, both aliases, port 8000, YaRN 2x / 524288 context,
MTP3 / greedy draft / block rejection / scale 2, eight sequence slots,
8192 batch tokens and **15000000000 bytes = decimal 15 GB ≈ 13.97 GiB**
main FP8 KV pool. This is a launch parameter, not an image default.

| Check | Result |
|---|---|
| GPU kernel suite | 23 passed: 16 upstream fused cases + 7 scorer/guard cases |
| Host configuration tests | 7 passed |
| Exact target/draft CLI configuration | Passed |
| Main/draft indexer dtype proof | 13 FP8 compressed caches; raw rings remain BF16 |
| Health, both aliases, reasoning, automatic/escaped tools, JSON, arithmetic | Passed |
| Required/named tool-choice probes | 13 passed; no real tools executed |
| Eight SSE streams | All 512-token streams finished cleanly; observed eight running |
| Mixed structured streams | 16 JSON + 16 forced-tool streams passed |
| Multi-prefill | Four/eight 11034-token requests passed |
| Near-limit synthetic retrieval | 522992 prompt + 11 output tokens; exact key; 74.15 s |
| Eight-worker soak | 1951 correct replies in 60.19 s |
| 800-token concurrent MTP diagnostic | Passed; 0/9 dead drafters, observed 1/3/6 requests |
| Engine failures | No CUDA OOM, illegal memory access, or restart observed |

KV capacity increased from **938050 to 966390 token slots** (+28340, about 3.02%)
at the same fixed 15 GB pool. Configured context is unchanged; capacity is shared,
not eight simultaneous full-context requests.

After API probes, the GPU had 1819 MiB free. After multi-prefill and the long
retrieval/soak/MTP tests it had only **231 MiB free** (97020 MiB used), versus
1653 MiB on the previous recorded BF16-indexer 15 GB trial. Driver-reported
free memory cannot distinguish live allocations from PyTorch's reusable
allocator cache. This uncontrolled result motivated the isolated follow-up below.
Three cache preemptions were recorded; they are not CUDA OOMs.

Some external client traffic overlapped the qualification, so these timings
are not a controlled A/B throughput benchmark. The synthetic retrieval and
short soak are not exhaustive model-quality evaluation. Existing nonfatal
grammar-error behavior must not be confused with an FP8 correctness claim.

The first additional scoring fixture requested unsupported k=16 from our
SM120 deterministic kernel. It was corrected to the actual model budget
(k=512 compressed groups); all 23 tests then passed. No unpublished RC image
was pushed during these fixture/build corrections.

## Isolated memory comparison, 2026-10-06

Two fresh containers ran sequentially on the same GPU with localhost-only
HTTP binding, preventing external inference from entering either trial.
Both retained their existing compile caches and used the same model, context,
batch budget, MTP settings and 15 GB KV budget. A temporary worker subclass
delegated inference to the original implementation and recorded live tensor,
allocator-reserved, inactive-segment and fragmentation statistics. It did not
add allocator cleanup or change the serving algorithms.

Both builds passed the same API checks, eight SSE/mixed structured streams,
four/eight 11034-token prefills, a 522992-token synthetic retrieval request
and an eight-worker 20-second arithmetic soak. The comparison is a bounded
memory/functional check, not a statistically controlled performance benchmark.

Final idle worker measurements (MiB):

| Measurement | BF16 indexer | FP8 indexer |
|---|---:|---:|
| Live tensor memory | 88490.13 | 88502.83 |
| Allocator-reserved memory | 94656.00 | 94630.00 |
| Reserved minus live | 6165.87 | 6127.17 |
| Inactive split fragments | 305.84 | 267.14 |
| Driver-free memory | 1654.25 | 1680.25 |
| Allocator OOMs / allocation retries | 0 / 0 | 0 / 0 |

Model-loading and post-profile memory were identical down to the byte.
Physical KV allocations stayed below the fixed budget: 14981324800 bytes
with BF16 indexing and 14979993600 with FP8 indexing. After multi-prefill,
live tensors returned to the earlier idle level in both builds while reserved
memory grew by 3770 MiB. Both ended with 6144655360 bytes in fully inactive
allocator segments and a 419430400-byte retained workspace.

The controlled run did **not reproduce an approximately 1.4 GiB intrinsic
FP8 memory regression**; FP8 instead finished with 26 MiB more driver-free
memory. Different workload/allocation history is consistent with the earlier
observation, but the old process had no allocator telemetry: its exact excess
cannot be retrospectively classified or attributed to a specific client.
Driver-free memory alone understates the memory reusable by the allocator;
these figures still do not establish a maximum safe KV budget.

The original FP8 service was restored without the diagnostic worker. Its
command, environment, mounts and restart policy were verified unchanged.
API and four/eight-client prefill checks passed again; afterward the GPU had
1703 MiB free, zero running/waiting requests and zero engine restarts.
No cleanup patch, allocator tuning, image promotion or external publication
was performed during this investigation.

## Reproduce

Build with `bash image/build-fp8-indexer.sh`. For a full source rebuild, first
run `bash image/build.sh`, then set `RC_BASE=qwen38-rtx-source:rebuilt` for the
candidate builder. GPU kernel instructions are in `tests/fp8_indexer/README.md`.
For deployment, select the candidate image and `INDEXER_KV_DTYPE=fp8`, regenerate
the image-specific draft scale module, and run the preflight checks. The
tested host profile additionally sets `KV_CACHE_MEMORY_BYTES=15000000000`,
`MAX_NUM_BATCHED_TOKENS=8192`, `MAX_NUM_SEQS=8`, `MAX_MODEL_LEN=524288` and
`MTP_TOKENS=3`; the retained Froggeric template and both aliases stay unchanged.

Stable defaults remain the original v1.0.0 / 13 GB / BF16-indexer profile.
The 15 GB host trial above does not separately qualify the portable 13 GB
profile with FP8 indexing. Keep the old image and stopped rollback container.
