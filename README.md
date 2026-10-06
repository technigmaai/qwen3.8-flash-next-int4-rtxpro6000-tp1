# Qwen3.8-Flash-Next INT4 on NVIDIA RTX PRO 6000 — TP1

Run `azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound` on one NVIDIA
RTX PRO 6000 Blackwell 96 GB GPU with vLLM, MTP3 and an OpenAI-compatible API.

[Docker Hub](https://hub.docker.com/r/technigmaai/qwen3.8-flash-next-int4-rtxpro6000-tp1)
· [Full setup guide](docs/DEPLOYMENT.md)
· [Validation summary](validation-summary.json)
· [Rebuild guide](docs/REBUILD.md)

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

## Published runtime and serving profile

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
their own testing; this release records the 8192 / C8 / 13 GB profile.

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
allow rebuilding without private intermediate images. `manifest.json` records
source/file checksums and the release image digest.

Thanks to Saren-Arterius, azampatti, blazux, jschmied, Froggeric, Qwen, NVIDIA
and the vLLM/PyTorch/FlashInfer/FLA maintainers. Copied and derived files retain
their applicable notices; see [THIRD_PARTY.md](THIRD_PARTY.md). Model and base
image terms remain separate. Model weights, private `.env` settings, generated
caches, host addresses and raw user benchmark traces are excluded.
