# Pinned runtime, patches and rebuild

The release image is the exact tested running runtime, retagged for publication.
Its inherited OCI source label still identifies the Saren base; that is not the
location of this deployment package. Release provenance is in `manifest.json`
and this repository. Nothing was injected from the private host cache.

## Source stack

| Component | Pin |
|---|---|
| Official Qwen/vLLM base | `vllm/vllm-openai:qwen38-flash-next@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8` |
| Saren-Arterius recipe | `01c5914f322716b39fd71d5584ed800955582e65` |
| Azampatti classic recipe/helpers | `73c4fa07bddd5f4ae04c3384973ab5b4c4b61a80` |
| blazux FP8 QSA patch | `b76890d5a033dd00166c792393d39cf908f56034` |
| jschmied deterministic QSA | `e0ef69d4f5575dad00d34e05479eaf4c6547bace` |
| vLLM parser adapters | PR 52830, `46638857fdbb30e0c232c9e8f9cb1ff6d6f545c3` |
| vLLM XGrammar termination | Exact backend diff from merged PR 52805 (`12f64b3`) |

`third_party/saren/` retains the base Dockerfile and needed public sources;
`third_party/azampatti/` retains the draft-scale helper and license. Additional
FP8/deterministic-QSA build inputs are fetched at pinned commits with SHA256
checksums in `image/Dockerfile.rtx`. The large model weights are downloaded
separately, not rebuilt or embedded here.

The base's original GB10-oriented comments are retained as provenance. The
RTX extension explicitly compiles its deterministic QSA CUDA extension for
SM120a rather than SM121a. Do not run the image on ARM64 or assume the package
has been validated on other GPUs.

## Rebuild on a compatible AMD64 CUDA build host

```bash
./image/build.sh
```

This reconstructs the public digest-pinned base and included Saren changes,
applies the SM120/FP8 extension, then both vLLM fixes. It does not depend on
unpublished `technigmaai/qwen38-a5b-vllm` intermediate images. Network access
is needed for pinned base layers and checksummed build inputs. Compilation
may be expensive; `MAX_JOBS=2` limits the extension build.

The result is `qwen38-rtx-source:rebuilt`. A rebuild need not have the release
image's identical manifest digest: compiler/build metadata and layers can
differ. Validate adapters/XGrammar, the exact model configuration and live API
behavior before replacing a release image. Do not upgrade vLLM/Transformers
independently: this model-specific stack includes PLE, quantization, recurrent-
state and draft-head changes beyond ordinary upstream vLLM.

`test_xgrammar.py` checks terminal-token handling, MTP post-stop artifacts,
rollback and reset. `test_parser_adapters.py` uses the cached tokenizer to
check shared adapters, thinking kwargs, enforced required/named XML grammar
and extraction. Neither launches a server or executes a model tool.

The deployment applies draft-logit scale 2 using the image's own module and
verifies the generating image ID. Keep this independent of the downloaded
checkpoint; never edit its config or tensors to prepare the draft.
