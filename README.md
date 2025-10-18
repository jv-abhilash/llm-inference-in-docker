# LLM Router (GPU, Docker, HF Cache)

A minimal, reproducible setup to run a Hugging Face LLM (e.g., `Qwen/Qwen2.5-Math-7B-Instruct`) on NVIDIA GPUs using **Docker** and **Docker Compose**.  
Configuration is fully driven by `.env`, allowing you to switch CUDA, PyTorch, or Hugging Face cache paths easily without editing the source code.

---

## 🚀 Features

- GPU-enabled container (PyTorch + CUDA)  
- `.env`-driven configuration for flexible runtime and build parameters  
- Hugging Face cache mounting for fast re-runs  
- Optional offline/local model loading  
- Integrated health check endpoint (`/healthz`)  
- Supports HF Transfer backend for fast downloads (`hf_transfer`)  
- Easily portable to Docker Hub or any GPU machine

---

## 🧩 Requirements

- NVIDIA GPU with compatible **driver (≥ 550.00)**  
- Installed **NVIDIA Container Toolkit**  
- **Docker** and **Docker Compose v2+**  

---

## 📁 Repository Structure

```
.
├── Dockerfile
├── docker-compose.yml
├── entrypoint.sh
├── requirements.txt
├── server.py
├── .env
└── .dockerignore
```

---

## ⚙️ Configuration (.env)

Example `.env` file:

```env
# ---------- Build-time knobs ----------
BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
EXTRA_PY_PKGS=hf_transfer

# ---------- Runtime checks ----------
MIN_DRIVER_VERSION=550.00

# ---------- Model selection ----------
ROUTER_MODEL=Qwen/Qwen2.5-Math-7B-Instruct
# LOCAL_MODEL_DIR=                     # Optional (used for local-only mode)

# ---------- Hugging Face ----------
HF_HOME=/app/.cache/hf
HF_HUB_ENABLE_HF_TRANSFER=1
# HF_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXX   # Optional for gated/private models
# TRANSFORMERS_OFFLINE=1                 # Optional for strict offline mode

# ---------- GPU control ----------
CUDA_VISIBLE_DEVICES=0

# ---------- Server ----------
HOST=0.0.0.0
PORT=7000
```

---

## 🛠️ Build and Run

### 1. Build

```bash
docker compose build
```

### 2. Run

```bash
docker compose up
```

or run detached:

```bash
docker compose up -d
```

### 3. Test the health endpoint

```bash
curl http://localhost:7000/healthz
# → {"ok":true,"model":"Qwen/Qwen2.5-Math-7B-Instruct","loaded":true,"device":"cuda:0"}
```

---

## 🧠 Runtime Modes

### 🟢 **Cache-First Mode (Default)**

- Uses Hugging Face cache to store downloaded model weights.
- Next runs reuse the cached weights.

```yaml
volumes:
  - ./.cache/hf:/app/.cache/hf
```

> This mode allows automatic re-use of cached models between runs.

---

### 🔵 **Local-Only Mode (Offline)**

If you already have the full model downloaded (e.g., `/models/Qwen2.5-Math-7B-Instruct`):

1. Edit `.env`:
   ```env
   LOCAL_MODEL_DIR=/models/qwen
   ```
2. Add to `docker-compose.yml`:
   ```yaml
   volumes:
     - ./.cache/hf:/app/.cache/hf
     - /ABS/PATH/TO/Qwen2.5-Math-7B-Instruct:/models/qwen:ro
   ```
3. Rebuild and run:
   ```bash
   docker compose up --build
   ```

You’ll see logs like:
```
Loading Qwen/Qwen2.5-Math-7B-Instruct from local dir: /models/qwen
```

---

## 🧱 docker-compose.yml

```yaml
services:
  llm-router:
    container_name: llm-router
    build:
      context: .
      args:
        BASE_IMAGE: ${BASE_IMAGE}
        EXTRA_PY_PKGS: ${EXTRA_PY_PKGS}
    image: llm-router:custom
    env_file: .env
    ports:
      - "${PORT:-7000}:7000"
    gpus: all
    volumes:
      - ./.cache/hf:/app/.cache/hf
    restart: unless-stopped
```

---

## 🧩 Dockerfile

```dockerfile
ARG BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

ARG EXTRA_PY_PKGS="hf_transfer"

RUN apt-get update && apt-get install -y --no-install-recommends     git curl tini &&     rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN grep -vi '^torch' requirements.txt > /tmp/reqs.txt || true &&     pip install --no-cache-dir -r /tmp/reqs.txt &&     if [ -n "${EXTRA_PY_PKGS}" ]; then pip install --no-cache-dir ${EXTRA_PY_PKGS}; fi

COPY server.py /app/server.py
COPY entrypoint.sh /app/entrypoint.sh
COPY .env /app/.env
RUN chmod +x /app/entrypoint.sh

EXPOSE 7000
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/app/entrypoint.sh"]
```

---

## 🪄 entrypoint.sh (Summary)

- Loads `.env` into the environment.
- Validates `LOCAL_MODEL_DIR`.
- Checks NVIDIA driver version.
- Prints CUDA info and GPU name.
- Starts Uvicorn server (`server.py`).

---

## ⚡ Common Commands

| Purpose | Command |
|----------|----------|
| Build the image | `docker compose build` |
| Start container | `docker compose up` |
| Stop containers | `docker compose down` |
| View logs | `docker compose logs -f` |
| Rebuild image (after code changes) | `docker compose build --no-cache` |
| Check health | `curl http://localhost:7000/healthz` |

---

## 🐳 Push to Docker Hub

### 1. Tag the image
```bash
docker tag llm-router:custom yourname/llm-router:cu128
```

### 2. Login and push
```bash
docker login
docker push yourname/llm-router:cu128
```

### 3. Run anywhere
```bash
docker run --rm -it --gpus all   --env-file .env   -p 7000:7000   -v "$(pwd)/.cache/hf:/app/.cache/hf"   yourname/llm-router:cu128
```

---

## 🧹 Cleanup

```bash
docker system prune -a
docker volume prune
huggingface-cli scan-cache --dir .cache/hf
```