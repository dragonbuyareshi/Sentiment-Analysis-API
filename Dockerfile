# Multi-stage build for production image
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.11 python3-pip python3.11-venv git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY api/ ./api/
COPY models/checkpoints/ ./models/checkpoints/

ENV MODEL_DIR=models/checkpoints/final \
    ONNX_PATH=models/checkpoints/model_sim.onnx \
    USE_ONNX=true \
    REDIS_URL=redis://redis:6379 \
    API_KEYS=change-this-key

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "api.main:app", \
    "--host", "0.0.0.0", "--port", "8000", \
    "--workers", "1", "--loop", "uvloop"]