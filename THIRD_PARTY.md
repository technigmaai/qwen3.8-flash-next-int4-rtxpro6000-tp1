# Source provenance and license notices

This deployment package builds on:

- [Saren-Arterius/qwen3.8-Flash-DGX-AutoRound](https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound/tree/01c5914f322716b39fd71d5584ed800955582e65),
  pin `01c5914f322716b39fd71d5584ed800955582e65`. Required base Dockerfile/source
  files and its license are retained under `third_party/saren/` without edits.
- [azampatti/Qwen3.8-Flash-Next-Int4-FAST](https://github.com/azampatti/Qwen3.8-Flash-Next-Int4-FAST/tree/73c4fa07bddd5f4ae04c3384973ab5b4c4b61a80),
  pin `73c4fa07bddd5f4ae04c3384973ab5b4c4b61a80`. The unchanged draft-scale
  helper and its license are retained under `third_party/azampatti/`. The draft
  configuration generator is adapted to this portable container layout.
- [blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX/tree/b76890d5a033dd00166c792393d39cf908f56034),
  pin `b76890d5a033dd00166c792393d39cf908f56034`, for FP8 QSA KV support.
- [jschmied/qwen38-flash-next-gb10](https://github.com/jschmied/qwen38-flash-next-gb10/tree/e0ef69d4f5575dad00d34e05479eaf4c6547bace),
  pin `e0ef69d4f5575dad00d34e05479eaf4c6547bace`, for deterministic QSA;
  CUDA compilation is explicitly retargeted to SM120a.
- [vLLM PR 52805](https://github.com/vllm-project/vllm/pull/52805) and
  [PR 52830](https://github.com/vllm-project/vllm/pull/52830) for exact upstream
  runtime diffs. vLLM files retain applicable Apache-2.0/SPDX notices.
- [Froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates)
  for the unchanged `qwen3.8-froggeric-v22.5` template. Its model-card license
  metadata is Apache-2.0. The deployed file is pinned by checksum
  `e57684bae4156211a55473c5a63be976a405a37ab5be5ae0e5abf1df5349c4b2`, not by
  a moving `main` download.

Apache-2.0 text is retained in `licenses/Apache-2.0.txt`. Copied/derived files
keep their applicable notices and terms; this repository does not relicense
all upstream work under one blanket license. Build-fetched source retains its
original terms; inspect the linked pinned repositories when redistributing.

Qwen, NVIDIA, vLLM, PyTorch, FlashInfer, Triton, FLA and other base-image
dependencies retain their own licenses. Model weights are **not** included in
this source package or runtime image. Their terms remain separate; consult
[the selected model](https://huggingface.co/azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound)
and its original sources. No private host cache, credentials or user prompts
are part of the published package.
