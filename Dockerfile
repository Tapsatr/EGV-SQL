# vLLM and the Qwen3.6 Gated-DeltaNet kernels JIT-compile CUDA at runtime and need
# nvcc, so we use a CUDA *devel* base (not -runtime, which omits the compiler and
# makes EngineCore die on the first prefill with "nvcc: not found").
#
# CUDA-tag choice: pinned to 12.3 to match BIRD's eval driver (CUDA 12.2/12.3).
# A 12.3 toolkit runs NATIVELY on their 12.3 driver (no forward-compat needed) and
# also runs on newer dev drivers (e.g. our H200 pod reports CUDA 13.0) by plain
# backward compatibility — so the same image works in both places. Do NOT bump this
# above the eval driver's CUDA: the GDN kernels JIT at runtime and a too-new toolkit
# can emit cubins/PTX the older eval driver cannot load.
FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

# Cache dirs all under /app/.cache so they work when the container runs as an
# arbitrary non-root UID (e.g. OpenShift SCC), where HOME may be "/" or unset and
# writes to ~/.cache would fail (EACCES) and kill vLLM EngineCore on first prefill.
# XDG_CACHE_HOME + the explicit vLLM/Triton/flashinfer roots cover every writer.
# For large models, mount a PVC at /app/.cache/huggingface so the ~54 GB download
# persists across restarts instead of filling ephemeral storage.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    HOME=/app \
    XDG_CACHE_HOME=/app/.cache \
    HF_HOME=/app/.cache/huggingface \
    TORCH_HOME=/app/.cache/torch \
    TRITON_CACHE_DIR=/app/.cache/triton \
    VLLM_CACHE_ROOT=/app/.cache/vllm \
    FLASHINFER_WORKSPACE_DIR=/app/.cache/flashinfer

WORKDIR /app

# ninja is required at runtime: Qwen3.6 GDN kernels JIT-compile via torch/vLLM
# and spawn `ninja` (FileNotFoundError without it → EngineCore init fails).
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    sqlite3 \
    git \
    curl \
    build-essential \
    ninja-build \
    libgomp1 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# --- Client env (system Python): inference.py + bge reranker + HTTP client ---
# Pin the reranker's torch to a cu12.1 wheel so it runs NATIVELY on the 12.3 eval
# driver (and on newer dev drivers) without relying on CUDA minor-version compat.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt

# --- Isolated vLLM serving env (separate venv) ---
# vLLM pins its own torch/xformers, so it is kept apart from the client stack.
# It MUST be new enough for the Qwen3.6 arch (Qwen3_5ForConditionalGeneration):
# support merged upstream in PR #34110 and ships in vLLM >= 0.17.0.
# Pin to the exact vLLM we validated on BIRD-dev (Qwen3.6 / GDN on CUDA 12.3).
RUN python3 -m venv /app/venv-vllm && \
    /app/venv-vllm/bin/pip install --no-cache-dir -U pip && \
    /app/venv-vllm/bin/pip install --no-cache-dir "vllm==0.26.0"

COPY . .

# Pre-create the cache dirs and make everything under /app group-writable (GID 0)
# so an arbitrary OpenShift UID can write. Harmless for a plain `docker run` as root.
RUN mkdir -p /app/.cache/huggingface /app/.cache/torch /app/.cache/triton \
             /app/.cache/vllm /app/.cache/flashinfer /data/out && \
    chgrp -R 0 /app /data && \
    chmod -R g=u /app /data

# run_pipeline.sh picks up this binary for the vLLM server; the client runs in
# system Python. They communicate over localhost:8000.
ENV VLLM_BIN=/app/venv-vllm/bin/vllm

# Mount test data and override paths at run time, e.g.:
#   docker run --gpus all \
#     -v /host/test_databases:/data/test_databases \
#     -v /host/test.json:/data/test.json \
#     -e DB_DIR=/data/test_databases -e TEST_JSON=/data/test.json \
#     <image>
CMD ["bash", "run_pipeline.sh"]
