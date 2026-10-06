# FP8 QSA indexer: v1.0.1-rc.1

Status: built locally on the RTX and retained as the active experimental
service after functional qualification. **Not promoted to stable or published
as a release.** The unexpectedly small post-prefill free-memory margin requires
investigation before recommending this profile for production.

## Scope

Backport of merged vLLM PR #54890, pinned head
`6fceb71365803ced6bee5f2a0972df05358f7075`, adapted to preview `8e685d198`.
Changes are restricted to the compressed indexer cache, query dtype and dtype
guards/backend declarations. The old combined scorer, cache layout, SM120
deterministic top-k and existing tuning are preserved. All 13 initialized
indexers (12 target + 1 MTP) log FP8 compressed keys and BF16 raw rings.

Candidate tag: `v1.0.1-rc.1-sm120-amd64-cu130`.
Local image ID: `sha256:f3a96e6070aab99286783d4f118e2114ff80d4035bf47d5f9cbe39c0e93a523e`.
Local manifest-list digest: `sha256:7a53dbe5398e50183a87fa91732ba6d526bff1c70bd889e7b31444fa9fac4fe9`.
The local digest is not a claim that the image is available in Docker Hub.

## Host trial, 2026-10-06

RTX PRO 6000 Blackwell 96 GB, TP1; same pinned AutoRound checkpoint, original
Froggeric template, both aliases, port 8000, YaRN 2x / 524288 context,
MTP3 / greedy draft / block rejection / scale 2, eight sequence slots,
8192 batch tokens and decimal 15 GB main FP8 KV pool.

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
allocator cache. The reason for this difference has not been established.
Three cache preemptions were recorded; they are not CUDA OOMs.

Some external client traffic overlapped the qualification, so these timings
are not a controlled A/B throughput benchmark. The synthetic retrieval and
short soak are not exhaustive model-quality evaluation. Existing nonfatal
grammar-error behavior must not be confused with an FP8 correctness claim.

The first additional scoring fixture requested unsupported k=16 from our
SM120 deterministic kernel. It was corrected to the actual model budget
(k=512 compressed groups); all 23 tests then passed. No unpublished RC image
was pushed during these fixture/build corrections.

## Reproduce

Build with `bash image/build-fp8-indexer.sh`. For a full source rebuild, first
run `bash image/build.sh`, then set `RC_BASE=qwen38-rtx-source:rebuilt` for the
candidate builder. GPU kernel instructions are in `tests/fp8_indexer/README.md`.
For deployment, select the candidate image and `INDEXER_KV_DTYPE=fp8`, regenerate
the image-specific draft scale module, and run the preflight checks.

Stable defaults remain the original v1.0.0 / 13 GB / BF16-indexer profile.
The 15 GB host trial above does not separately qualify the portable 13 GB
profile with FP8 indexing. Keep the old image and stopped rollback container.
