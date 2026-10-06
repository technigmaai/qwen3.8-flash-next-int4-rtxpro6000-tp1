# Qwen3.8-Flash-Next INT4 on NVIDIA RTX PRO 6000 — TP1

Run `azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound` on one NVIDIA
RTX PRO 6000 Blackwell 96 GB GPU with vLLM, MTP3 and an OpenAI-compatible API.

[Docker Hub](https://hub.docker.com/r/technigmaai/qwen3.8-flash-next-int4-rtxpro6000-tp1)
· [Full setup guide](docs/DEPLOYMENT.md)
· [Validation summary](validation-summary.json)
· [Rebuild guide](docs/REBUILD.md)
· [Versioning](docs/VERSIONING.md)

The tested FP8 QSA indexer candidate, `v1.0.1-rc.1-sm120-amd64-cu130`, runs
with **15 GB KV, 8192 batch tokens, eight sequence slots, 512k context and MTP3**.
It passed bounded functional/kernel checks and an isolated memory comparison.
It is published as a **release candidate, not a stable release**; published
`v1.0.0` remains unchanged. See [candidate results](docs/FP8-INDEXER-RC1.md) and
[versioning](docs/VERSIONING.md). FP8 indexer caching is an explicit opt-in,
separate from the main attention cache's existing FP8 dtype.

Derived from Saren-Arterius's pinned Qwen3.8 serving stack and azampatti's
classic AutoRound recipe. This RTX adaptation compiles deterministic QSA for
**SM120**, uses disk-backed PLE and FP8 KV, and backports two upstream vLLM
fixes for MTP grammar termination and enforced tool choice. It is **Linux
AMD64**, not an ARM64 GB10/Spark image.

## Architecture

One model container, one GPU, tensor parallelism **TP1**. No Ray, distributed
worker or second model server is required. The MTP drafter uses the same model
weights with its own generated config. The large PLE table remains disk-backed;
fast local NVMe and sufficient host RAM/page-cache headroom matter.

Weights stay in your Hugging Face hub cache, mounted read-only. They are not
included in the runtime image. The whole model cache repo is mounted so its
`snapshots/` links can reach its **repo-local `blobs/`**; no separate shared-blob
directory is required. Generated configs and runtime caches stay elsewhere.

## Included patches

| Patch / adaptation | Purpose | Included in |
|---|---|---|
| blazux FP8 main QSA KV support | Stores the main attention KV in FP8 E4M3; separate from indexer caching | v1.0.0 and RC |
| jschmied deterministic QSA top-k, compiled for SM120a | Deterministic sparse block selection on this RTX architecture | v1.0.0 and RC |
| [vLLM #52830](https://github.com/vllm-project/vllm/pull/52830), parser/reasoning adapters | Preserves request-specific thinking settings for shared parser engines; required/named tool enforcement is tested | v1.0.0 and RC |
| [vLLM #52805](https://github.com/vllm-project/vllm/pull/52805), XGrammar termination | Stops structured-output token batches at grammar termination, including speculative batches | v1.0.0 and RC |
| [vLLM #54890](https://github.com/vllm-project/vllm/pull/54890), adapted FP8 QSA indexer | Compresses normalized indexer keys and queries to FP8; raw compressor rings stay BF16 | RC only; opt-in |

The pinned Saren base also includes disk-backed PLE, INT4/FP8 hybrid layer
dispatch, quantized target/MTP LM heads, FLA shared-memory/warp adjustments,
Mamba state-copy race/bounds hardening and aligned prefill splitting. Optional
base prompt-pinning/debug/profiling hooks are not enabled by this profile.
Draft top-k 10, logit scale 2 and the Froggeric template are serving-profile
adaptations, not additional upstream bug fixes. Exact source pins, retained
notices and rebuild inputs are in [THIRD_PARTY.md](THIRD_PARTY.md) and
[the rebuild guide](docs/REBUILD.md).

## Tested FP8 release-candidate profile — 15 GB KV

Candidate image: `technigmaai/qwen3.8-flash-next-int4-rtxpro6000-tp1:v1.0.1-rc.1-sm120-amd64-cu130`.
The candidate is available on Docker Hub. This tested RTX configuration does
not change the stable `.env.example` defaults.

| Setting | Tested RC value |
|---|---|
| Fixed main KV budget | **15000000000 bytes = decimal 15 GB ≈ 13.97 GiB**, FP8 E4M3 |
| QSA indexer cache / raw rings | FP8 E4M3 / BF16 |
| Recorded boot KV capacity | **966390 shared token slots**; BF16 indexer at the same budget: 938050 (+3.02%) |
| Maximum context | 524288 tokens, prompt plus output; YaRN factor 2 on target and draft |
| Sequence slots / batch tokens | **8 / 8192** |
| Speculation | **MTP3**; block rejection, greedy drafting, draft top-k 10, logit scale 2 |
| Template / API | Retained Froggeric v22.5; port 8000; full azampatti model ID and `rtx` aliases |

Pull the candidate image, then start from `.env.example`, retain its other
settings and replace these values in your private `.env`:

```bash
docker pull technigmaai/qwen3.8-flash-next-int4-rtxpro6000-tp1:v1.0.1-rc.1-sm120-amd64-cu130
```

```dotenv
IMAGE=technigmaai/qwen3.8-flash-next-int4-rtxpro6000-tp1:v1.0.1-rc.1-sm120-amd64-cu130
KV_CACHE_MEMORY_BYTES=15000000000
INDEXER_KV_DTYPE=fp8
```

Rerun `./service.sh prepare` for the selected image, then the preflight checks.
The indexer option becomes `--attention-config '{"indexer_kv_dtype":"fp8"}'`.
The 15 GB budget is a launch setting, **not hardcoded in the image**, and is
shared by requests; eight full-512k requests do not fit simultaneously.

An isolated old-versus-FP8 comparison ended with **1654 versus 1680 MiB
driver-free memory**, with zero allocator OOMs/retries. Both allocators retained
substantial reusable memory after prefill. The earlier mixed-traffic 231 MiB
observation was not reproduced; this is not an arbitrary-workload OOM guarantee
or permission to increase the KV budget. See [the detailed results](docs/FP8-INDEXER-RC1.md).

## Published stable runtime and defaults — v1.0.0

| Setting | Value |
|---|---|
| Image | `technigmaai/qwen3.8-flash-next-int4-rtxpro6000-tp1:v1.0.0-sm120-amd64-cu130` |
| Platform | Linux AMD64; RTX PRO 6000 Blackwell 96 GB / SM120 |
| Runtime | CUDA 13.0; PyTorch 2.13.0+cu130; vLLM `0.1.dev20073+g8e685d198` |
| Model revision | `1464274120d36a4d8fcaa934552334a7d83ce0fd` |
| Maximum configured context | 524288 tokens, prompt plus output; YaRN factor 2 |
| Sequence slots / batch tokens | 8 / 8192 |
| Fixed KV pool | 13000000000 bytes (decimal 13 GB); FP8 E4M3 |
| Recorded boot KV capacity | 813354 shared token slots, approximately 1.55× maximum context |
| Speculation | MTP3; block rejection, greedy drafting; draft top-k 10 and logit scale 2 |
| Parsers | `qwen3_coder` tools; `qwen3` reasoning |
| Chat template | Froggeric `qwen3.8-froggeric-v22.5`, exact retained checksum |
| API | Port 8000; `/v1`, `/health`, `/metrics` |
| Model aliases | Full azampatti model ID and `rtx`; one loaded checkpoint |
| Recovery | Docker `unless-stopped`; no added cron/watchdog |

**Eight slots do not provide eight simultaneous full-512k contexts.** Requests
share one KV pool and prefill budget. The native model context is 262144;
524288 uses YaRN extension on both target and drafter. The synthetic near-limit
retrieval test below is not an exhaustive long-context quality qualification.

This image supplies the runtime. **Clone the repository and use its Compose
recipe:** the entrypoint, original chat template and generated scale module
are mounted by the deployment. Pulling the image alone neither downloads the
weights nor installs this complete serving profile.

## Deployment

Prepare a compatible Linux AMD64 host, the 96 GB Blackwell RTX PRO 6000,
Docker Engine, Compose v2+, NVIDIA Container Toolkit and a compatible driver.
The tested host used driver **595.84**; this is a recorded version, not a claim
about the minimum required driver. Read the [setup guide](docs/DEPLOYMENT.md)
for storage, pinned model download, checks and first-boot troubleshooting.

```bash
git clone https://github.com/technigmaai/qwen3.8-flash-next-int4-rtxpro6000-tp1.git
cd qwen3.8-flash-next-int4-rtxpro6000-tp1
docker pull technigmaai/qwen3.8-flash-next-int4-rtxpro6000-tp1:v1.0.0-sm120-amd64-cu130
cp .env.example .env
```

Edit `.env`: replace both host paths and set UID/GID from `id -u` / `id -g`.
Download the pinned checkpoint before preparation. Then:

```bash
./service.sh prepare
./service.sh check
./service.sh start --approved
./service.sh status
# Wait for health, then run the mock-tool/API/concurrency checks:
./service.sh verify
```

There must be no other model server occupying port 8000 or the selected GPU.
The package does not stop other deployments automatically.

## Operations

```bash
./service.sh status
./service.sh logs --tail 80
./service.sh restart --approved
./service.sh stop --approved
```

Change settings only in your private `.env`, then restart during a planned
interruption. `KV_CACHE_MEMORY_BYTES` is explicit: decreasing batch tokens does
not automatically increase KV capacity. Other memory/concurrency profiles need
their own testing. Published v1.0.0 records the 8192 / C8 / 13 GB profile;
the opt-in FP8 candidate records the separate 8192 / C8 / 15 GB profile above.

## Validation and limits

On **2026-10-06**, the exact published runtime passed ordinary chat, separated
reasoning, automatic tools, JSON schema, escaped/newline tool arguments,
repeated arithmetic/shared-prefix checks, eight simultaneous 512-token SSE
streams and mixed batches of 16 JSON plus 16 named-tool streams.

The actual `tool-eval-bench` **TC-45 passed 2/2** after the parser fix; it was
not excluded as a server failure. Thirteen adversarial probes verified required
and named tool choices with thinking on/off and streaming/non-streaming.
Probe scripts never execute generated tool calls. This pinned server reports
`stop` for named choices even when it returns a valid call; required choices
report `tool_calls`.

The earlier image, with the same model/context/MTP/cache settings but before
the parser-only fix, passed a **522992-token** synthetic retrieval request and
an eight-worker **1904-response** arithmetic soak. Those tests were not repeated
on the final parser-patched image. Startup and bounded regression checks on
the final image showed no CUDA/OOM/assertion/grammar-termination failure.

These are bounded deployment checks, not production qualification or a full
quality/performance benchmark. Multimodal/image workloads, maximum safe KV
allocation and multi-day stability are **not qualified**. Publishing does not
change that. See [recorded measurements and limitations](docs/VALIDATION.md).

## Rebuild, source maintenance and credits

The release publishes the exact already-tested runtime, not a full vLLM
upgrade. Included pinned base sources and [rebuild instructions](docs/REBUILD.md)
allow rebuilding without private intermediate images. `manifest.json` retains
the v1.0.0 release snapshot; [manifest-v1.0.1-rc.1.json](manifest-v1.0.1-rc.1.json)
records the candidate's public source checksums, registry digest and tested profile.

Thanks to Saren-Arterius, azampatti, blazux, jschmied, Froggeric, Qwen, NVIDIA
and the vLLM/PyTorch/FlashInfer/FLA maintainers. Copied and derived files retain
their applicable notices; see [THIRD_PARTY.md](THIRD_PARTY.md). Model and base
image terms remain separate. Model weights, private `.env` settings, generated
caches, host addresses and raw user benchmark traces are excluded.
