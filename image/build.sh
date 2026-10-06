#!/usr/bin/env bash
# Rebuild from included pinned public sources, not unpublished base images.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
docker build --platform linux/amd64 -t qwen38-rtx-source:base third_party/saren
docker build --platform linux/amd64 --build-arg BASE=qwen38-rtx-source:base \
  -f image/Dockerfile.rtx -t qwen38-rtx-source:sm120 image
docker build --platform linux/amd64 --build-arg BASE=qwen38-rtx-source:sm120 \
  -f image/Dockerfile.fixes -t qwen38-rtx-source:rebuilt .
echo "Rebuilt runtime: qwen38-rtx-source:rebuilt; qualify it before replacing a release image."
