# Deployment and troubleshooting

## Host and storage

Use one NVIDIA RTX PRO 6000 **Blackwell 96 GB**, not the older RTX 6000 Ada
48 GB. This package is Linux AMD64 / SM120 / TP1, not GB10 ARM64 or multi-GPU
tensor parallelism. The tested driver was 595.84. Docker must have the NVIDIA
Container Toolkit configured; verify GPU passthrough before deployment.
Python 3.10+ is required for the dependency-free host helpers.

The complete pinned model download was 130836896655 bytes across 123 files.
Allow additional room for the runtime image, CUDA/compile caches and generated
deployment data. Store the model, especially `ple-table/`, on fast local NVMe.
PLE is mmap-backed: this does not mean the host RAM/page cache is free or that
slow network storage will behave like the tested system. The maximum safe host
RAM/storage envelope has not been qualified across other hosts.

## Pinned model download and cache layout

Install/use the Hugging Face CLI in your normal environment. For a predictable
traditional cache layout, a dedicated client environment can use
`huggingface_hub[cli]==0.36.0`, whose cache stores blobs per model repo. This is
a downloader, separate from the image's pinned serving libraries. Do not
replace the image's Transformers/vLLM packages to change the downloader.

```bash
python3 -m venv .hf-download
.hf-download/bin/pip install 'huggingface_hub[cli]==0.36.0'
HF_HUB_DISABLE_XET=1 .hf-download/bin/hf download \
  azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound \
  --revision 1464274120d36a4d8fcaa934552334a7d83ce0fd
```

Use your normal Hugging Face authentication if required; never commit tokens.
Do **not** use `--local-dir` for this hub-cache recipe. Expected layout:

```text
~/.cache/huggingface/hub/
  models--azampatti--Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound/
    blobs/
    refs/
    snapshots/1464274120d36a4d8fcaa934552334a7d83ce0fd/
```

Mount the complete `models--...` directory. Snapshot files may be links into
the same repo's `blobs/`; mounting only the snapshot breaks those links.
A newer downloader must be checked for equivalent repo-local blob storage
before using it. The runtime's shared-blob/Xet-disable flags do not rearrange
files downloaded elsewhere. No model files are relocated or modified by the
deployment helpers. Existing matching downloads can be used directly.
The selected pinned repo includes `ple-table/` with all 33 table files (shards
5–37) alongside the model weights; no separate PLE conversion/download step
is needed for this checkpoint.

## Configuration and preparation

Copy `.env.example` to `.env`. Set absolute `MODEL_HOST_DIR` to the complete
model repo and `RUNTIME_HOST_DIR` to a **separate dedicated directory** writable
by your account. Do not point runtime at the model cache. Replace `HOST_UID`
and `HOST_GID` with your own numeric IDs. Select GPU and API port as needed.

`.env` contains literal values: no shell variables, substitutions or `~`.
The helpers do not evaluate shell code. Do not publish the private `.env`.

Pull the release image, then run `./service.sh prepare`. It creates:

- An independent draft `config.json` with top-k 10, the MTP head's own shared
  expert width (640), and YaRN factor 2 at 524288 positions.
- Container-resolved symlinks to the same read-only weights; no weight copies.
- An image-specific scaled MTP module using azampatti's retained helper.

The wrapper's unused top-level RoPE stays `default` to avoid the pinned
Transformers constructor issue; the nested text config contains YaRN. Engine
configuration checks verify that target and draft actually use the scaled
text configuration. Preparation refuses unknown existing draft directories.
Rerun preparation if you change the image; its module/image guard must match.

Run `./service.sh check` before startup. It checks pinned files, image labels,
the exact template, generated draft/scale module, Compose configuration, idle
GPU memory and the host port. `--config-only` skips occupied GPU/port checks
for non-mutating diagnostics while another deployment is running.

The start command does not pull implicitly or stop another service. It uses
`--pull never`; download the image first. A fresh model load and compilation
may take longer than cached restarts. Use logs and `/health`, not only Docker's
`running` state, to decide when the API is ready.

## API and reasoning

Use `http://HOST:8000/v1` with either:

- `azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound`
- `rtx`

They are aliases for one checkpoint, not separate loaded models. The template
defaults to thinking enabled. Use `chat_template_kwargs.enable_thinking=false`
to disable it; `thinking` is not the template's explicit on/off switch.
The retained template maps `high`/`xhigh` reasoning effort to its xhigh mode.
The API is exposed to the host network via its published port; restrict access
with your normal firewall/authentication proxy before exposing it publicly.
This recipe does not add API authentication.

## First-boot failures

- **GPU/port busy:** stop your existing deployment deliberately. The helper
  will not do that for you. Do not run two full-model containers on one GPU.
- **Tokenizer or weight file missing:** verify the pinned snapshot and repo-
  local blobs; mount the entire cache repo and retain symlinks when copying it.
- **Permission denied:** runtime must be writable by configured UID/GID;
  original model files and snapshot symlink targets must be readable.
- **Image-specific scale mismatch:** pull the intended image and rerun prepare.
- **OOM:** the 13 GB KV pool is explicit and ignores auto memory sizing. Keep
  headroom for activation/workspace growth, long prefills and other processes.
  Do not infer a larger safe cache just from idle free VRAM.
- **TC-45 excluded:** confirm the published parser-patched image/labels, then
  rerun `./service.sh verify`; changing the prompt to request a tool is not an
  enforcement test. The adversarial probe must return a call even when told
  not to call tools.

No automatic host-service changes, watchdog or model-cache cleanup is installed.
