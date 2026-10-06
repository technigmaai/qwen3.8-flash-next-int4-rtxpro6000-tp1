# Recorded validation and limits

Validation date: 2026-10-06. Host: one Linux AMD64 NVIDIA RTX PRO 6000 Blackwell
96 GB, driver 595.84. This records tests on the original operational deployment;
the release publishes its exact final runtime. Portable Compose packaging
checks are recorded separately and do not imply another full model boot.

## Final parser-patched runtime

- Health and both aliases; configured context 524288 verified through `/v1/models`.
- Startup: 813354-token FP8 KV pool, 1.55× full-context concurrency; MTP scale 2.
- 13 adversarial required/named probes with thinking on/off and streaming/non-
  streaming, including the original 256-token/xhigh capability probe.
- Installed `tool-eval-bench` TC-45: **2/2 PASS**, seed 42, thinking/high backend
  kwargs, one worker. Previously this scenario was excluded due to missing
  server enforcement. It now executes and scores normally.
- Chat, separated reasoning, automatic tool selection, JSON schema, escaped/
  newline tool arguments, six arithmetic and three shared-prefix requests.
- Eight simultaneous 512-token SSE streams; running-request metric reached 8;
  all returned finish reasons and `[DONE]` in approximately 5.72–6.10 seconds.
- Mixed 16 JSON-schema and 16 named-tool SSE requests, temperatures 0 and 0.6.
- No observed CUDA/OOM/assertion/grammar-termination failure in startup/tests.

Mock tool calls in the API probes are parsed/validated, never executed. The
benchmark runs its own local calculator fixture. Valid named calls finish
with `stop` in this pinned server; required calls report `tool_calls`.
Enforcement checks require actual valid calls, not a finish label alone.

## Earlier image with the same serving profile

Before the parser-only PR 52830 change, the prior image already included the
XGrammar termination fix and the same model, YaRN, MTP3, batch and KV settings:

- One 522992-token synthetic retrieval prompt returned the identifier at the
  beginning; 11 output tokens, clean stop/DONE, 80.72 seconds.
- Eight-worker arithmetic soak: 1904 correct responses in 60.14 seconds.
- GPU memory after the near-limit prefill: about 91178 MiB used, 6073 MiB free.

These long-context/soak tests were **not repeated on the parser-patched image**.
The final image's bounded parser/API/concurrency checks are separate evidence.
Small tutorial timing checks are not a clean comparative performance benchmark.

## Explicit limits

No production or multi-day stability qualification, exhaustive long-context
quality evaluation, maximum safe KV-memory search, image/video workload
qualification or eight-full-context concurrency claim. The model's native
context is 262144; 524288 uses YaRN factor 2. The KV pool is shared and may
queue/preempt workloads beyond its capacity. Host page cache and storage
behavior can differ materially across installations.

## Portable package checks

Five host-only settings tests and Python/shell syntax checks passed. Preparation
and Compose/model/draft/template/image checks passed using a separate ignored
runtime directory. A temporary configuration-only container captured the
portable entrypoint and verified actual engine configs for target and draft:
YaRN 2x, 524288 context, MTP3, eight slots, 8192 batched tokens, 13 GB FP8 and
both aliases. It did not load model tensors or bind the API port. The existing
operational container was not restarted or modified during publication.
The published `service.sh verify` wrapper was also run against that existing
healthy service: all 13 adversarial probes, API checks, eight concurrent
streams and mixed structured streams passed. This verifies the public check
entrypoint, not a fresh full-model boot of the portable Compose deployment.
