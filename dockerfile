# Accept build args from CLI (we’ll pass them from .env)
ARG BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

# Re-declare for later layers (best practice)
ARG EXTRA_PY_PKGS="hf_transfer"

# System deps + tini for clean PID 1
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl tini && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps, avoiding torch (already in base image)
COPY requirements.txt /app/requirements.txt
RUN grep -vi '^torch' requirements.txt > /tmp/reqs.txt || true && \
    pip install --no-cache-dir -r /tmp/reqs.txt && \
    if [ -n "${EXTRA_PY_PKGS}" ]; then pip install --no-cache-dir ${EXTRA_PY_PKGS}; fi

# App files
COPY server.py /app/server.py
COPY entrypoint.sh /app/entrypoint.sh
COPY .env /app/.env
RUN chmod +x /app/entrypoint.sh

# Expose your app port (env overrides at runtime still fine)
EXPOSE 7000

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/app/entrypoint.sh"]