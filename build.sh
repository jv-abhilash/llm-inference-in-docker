#!/usr/bin/env bash
set -euo pipefail
# shellcheck disable=SC2046
export $(grep -v '^#' .env | xargs) || true

docker build \
  --build-arg BASE_IMAGE="${BASE_IMAGE:-pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime}" \
  --build-arg EXTRA_PY_PKGS="${EXTRA_PY_PKGS:-hf_transfer}" \
  -t llm-router:custom .