# ══════════════════════════════════════════════════════════════════
# Sentiment Analysis API — Dockerfile
# Base: Ubuntu 22.04 + CUDA 12.8 + cuDNN 9
# Python: 3.11
# ══════════════════════════════════════════════════════════════════

FROM nvidia/cuda:12.8.0-cudnn9-runtime-ubuntu22.04

# ── System labels ──────────────────────────────────────────────
LABEL maintainer="Dragon Buyareshi"
LABEL description="Sentiment Analysis API — DistilBERT + FastAPI + ONNX"
LABEL version="2.1.0"

# ── Env ────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_VISIBLE_DEVICES=0 \
    TOKENIZERS_PARALLELISM=false

# ── System dependencies ─────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3.11-venv \
    build-essential \
    gcc \
    g++ \
    make \
    git \
    curl \
    wget \
    ca-certificates \
    libssl-dev \
    libffi-dev \
    libgomp1 \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    zlib1g-dev \
    libxml2-dev \
    libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Make python3.11 the default ─────────────────────────────────
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# ── Upgrade pip ─────────────────────────────────────────────────
RUN pip install --upgrade pip setuptools wheel

# ── Create app user (non-root) ──────────────────────────────────
RUN useradd -m -u 1000 appuser
WORKDIR /app

# ── Install PyTorch CUDA 12.8 first (heavy layer) ───────────────
RUN pip install \
    torch==2.5.1+cu128 \
    torchvision==0.20.1+cu128 \
    torchaudio==2.5.1+cu128 \
    --extra-index-url https://download.pytorch.org/whl/cu128

# ── Install remaining requirements ──────────────────────────────
COPY requirements.txt .
RUN grep -v "^torch" requirements.txt | grep -v "^#.*PyTorch" | pip install -r /dev/stdin

# ── Download spaCy English model ────────────────────────────────
RUN python -m spacy download en_core_web_sm

# ── Copy application code ────────────────────────────────────────
COPY api/       ./api/
COPY models/    ./models/

# ── Create directories ───────────────────────────────────────────
RUN mkdir -p logs data/processed \
    && chown -R appuser:appuser /app

USER appuser

# ── Environment defaults ─────────────────────────────────────────
ENV MODEL_DIR=models/checkpoints/final \
    ONNX_PATH=models/checkpoints/model_sim.onnx \
    USE_ONNX=true \
    REDIS_URL=redis://localhost:6379 \
    API_KEYS=dev-key-123 \
    MAX_LENGTH=128 \
    CACHE_TTL=3600 \
    PORT=8000

EXPOSE 8000

# ── Health check ─────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/v1/health || exit 1

# ── Start ────────────────────────────────────────────────────────
CMD ["sh", "-c", \
     "uvicorn api.main:app \
      --host 0.0.0.0 \
      --port ${PORT} \
      --workers 1 \
      --loop uvloop \
      --access-log \
      --log-level info"]
