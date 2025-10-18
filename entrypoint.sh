#!/usr/bin/env bash
# Robust, env-aware launcher for the LLM router server
set -euo pipefail

# -------- Locate script & project root (so .env works no matter where you run from) ------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# -------- Load .env (or override with ENV_FILE) ------------------------------------------
ENV_FILE="${ENV_FILE:-"$SCRIPT_DIR/.env"}"
if [[ -f "$ENV_FILE" ]]; then
  # Export all variables defined in .env
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi

# --- Optional driver gate: require >= MIN_DRIVER_VERSION if set ---
if [[ -n "${MIN_DRIVER_VERSION:-}" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found. NVIDIA driver not visible to container." >&2
    exit 1
  fi
  HOST_DRV="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)"
  # version compare using sort -V
  if [ "$(printf '%s\n' "$MIN_DRIVER_VERSION" "$HOST_DRV" | sort -V | head -n1)" != "$MIN_DRIVER_VERSION" ]; then
    echo "ERROR: Host driver $HOST_DRV < required $MIN_DRIVER_VERSION" >&2
    exit 1
  fi
  echo "Driver OK: host=$HOST_DRV, required>=$MIN_DRIVER_VERSION"
fi

# -------- Sensible defaults --------------------------------------------------------------
: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"   # your .env uses 7000; that will override this default
: "${HF_HOME:="$SCRIPT_DIR/.cache/hf"}"
: "${TRANSFORMERS_CACHE:="$HF_HOME"}"

# -------- Prepare caches (absolute or relative) ------------------------------------------
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE"

# -------- Validate LOCAL_MODEL_DIR if provided -------------------------------------------
if [[ -n "${LOCAL_MODEL_DIR:-}" ]]; then
  if [[ ! -d "$LOCAL_MODEL_DIR" ]]; then
    echo "ERROR: LOCAL_MODEL_DIR is set but not found: $LOCAL_MODEL_DIR" >&2
    exit 1
  fi
fi


#  -------- Detect Whether it is Docker or Not ---------------------------------------------------
if [ -f /.dockerenv ]; then
  echo "Runtime: Docker (/.dockerenv present)"
else
  if grep -qE 'docker|containerd|kubepods' /proc/1/cgroup 2>/dev/null; then
    echo "Runtime: Containerized (cgroup match)"
  else
    echo "Runtime: Host (no container markers found)"
  fi
fi

# -------- Quick diagnostics (helps debug Docker vs host env issues) ----------------------
echo "=== LLM Router Startup ==="
echo "ENV  ROUTER_MODEL=${ROUTER_MODEL:-<unset>}"
echo "ENV  LOCAL_MODEL_DIR=${LOCAL_MODEL_DIR:-<unset>} (exists: $( [[ -n "${LOCAL_MODEL_DIR:-}" && -d "$LOCAL_MODEL_DIR" ]] && echo yes || echo no ))"
echo "ENV  HF_HOME=${HF_HOME}"
echo "ENV  TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}"
echo "ENV  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "ENV  HOST=${HOST}  PORT=${PORT}"
echo "ENV  TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-<unset>}"
python - <<'PY'
import torch, os
print("Torch:", torch.__version__, "| CUDA build:", getattr(torch.version, "cuda", None))
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("MODEL_ID:", os.getenv("ROUTER_MODEL"))
print("LOCAL_MODEL_DIR:", os.getenv("LOCAL_MODEL_DIR"))
PY
echo "=== Starting server ==="

# -------- Start FastAPI app --------------------------------------------------------------
exec python -m uvicorn server:app --host "$HOST" --port "$PORT" --workers 1