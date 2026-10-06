#!/usr/bin/env bash
# Build only; never pushes, retags stable, or restarts a service.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RC_IMAGE="${RC_IMAGE:-technigmaai/qwen3.8-flash-next-int4-rtxpro6000-tp1:v1.0.1-rc.1-sm120-amd64-cu130}"
RC_BASE="${RC_BASE:-technigmaai/qwen3.8-flash-next-int4-rtxpro6000-tp1@sha256:2da71c1a8d47c2869bea794e67c2ee8b8026d2f029677561d735dcf192047791}"
docker build --platform linux/amd64 --build-arg "BASE=$RC_BASE" \
  -f "$ROOT/image/Dockerfile.fp8-indexer" -t "$RC_IMAGE" "$ROOT"
echo "Built candidate: $RC_IMAGE. Qualify before stable promotion."
