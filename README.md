# Sentiment Analysis API

> **3-class sentiment analysis service powered by fine-tuned DistilBERT, ONNX Runtime, FastAPI, and Redis — fully containerised with Docker.**

---

## Project Description

This project builds a production-grade REST API that classifies any piece of text into one of three sentiment categories: **POSITIVE**, **NEUTRAL**, or **NEGATIVE**.

The core idea is to take a state-of-the-art transformer model (DistilBERT), fine-tune it on labelled sentiment datasets, export it to the ONNX format for fast inference, wrap it in a FastAPI server with Redis caching and Bearer-key authentication, then ship the whole stack in Docker so it runs identically on any machine.

The project is structured as a real ML engineering system, not just a notebook experiment. It includes:

- A **training pipeline** with two-phase fine-tuning (frozen backbone → unfrozen top layers)
- An **evaluation pipeline** that produces confusion matrices and per-class F1 scores
- An **ONNX export pipeline** that benchmarks the speedup (typically 2–3x)
- A **FastAPI server** with async Redis caching, Prometheus metrics, and Pydantic validation
- A **Jupyter notebook suite** for EDA, LSTM baseline, and transformer fine-tuning
- A **full pytest test suite** (30+ tests) covering API endpoints and inference unit tests
- A **frontend UI** (single HTML file, served by Nginx) for interactive demo
- A **Docker Compose stack** wiring API + Redis + Nginx + Prometheus together

Target: **≥ 94% accuracy** on SST-2 with **< 50ms inference latency** on an RTX 4060.

---

## Architecture

```
Client / UI  (port 3000)
        |  HTTP JSON
        v
+----------------------------------------------+
|  FastAPI  (uvicorn + uvloop)                 |
|  +--  Bearer Auth Middleware                 |
|  +--  Pydantic Validation                    |
|  +--  Redis Cache  (1 hr TTL)               |
|  +--  SentimentEngine  (singleton)           |
|       +--  Text Preprocessing               |
|       |    HTML unescape, emoji, URLs        |
|       +--  HuggingFace Tokenizer             |
|       +--  ONNX Runtime / PyTorch            |
|            DistilBERT + Softmax              |
+----------------------------------------------+
        |
        v
  JSON Response
  { label, confidence, scores, metadata }
```

---

## Tech Stack

| Layer           | Technology                          |
|-----------------|-------------------------------------|
| Language        | Python 3.11                         |
| Deep Learning   | PyTorch 2.5.1 + CUDA 12.8           |
| Model           | DistilBERT-base-uncased (HuggingFace)|
| Inference       | ONNX Runtime 1.20.1 (GPU)           |
| API Framework   | FastAPI 0.115 + uvicorn             |
| Validation      | Pydantic v2                         |
| Cache           | Redis 7 (async redis-py)            |
| Monitoring      | Prometheus + FastAPI Instrumentator |
| Auth            | Bearer API key middleware           |
| Containerisation| Docker + Docker Compose             |
| Frontend        | Nginx + single-page HTML            |
| Testing         | pytest + pytest-asyncio + httpx     |
| Notebooks       | JupyterLab                          |

---

## Project Structure

```
sentiment-api/
|
+-- api/                        # FastAPI application
|   +-- __init__.py
|   +-- main.py                 # App entry point, all routes
|   +-- inference.py            # SentimentEngine, model load + predict
|   +-- schemas.py              # Pydantic request/response models
|   +-- cache.py                # Async Redis cache layer
|   +-- auth.py                 # Bearer API key dependency
|
+-- models/                     # ML pipeline scripts
|   +-- __init__.py
|   +-- train.py                # DistilBERT fine-tuning (fp16, two-phase)
|   +-- evaluate.py             # Metrics, F1, confusion matrix PNG
|   +-- export_onnx.py          # PyTorch -> ONNX + simplify + benchmark
|
+-- notebooks/                  # Jupyter notebooks
|   +-- 01_eda.ipynb            # Dataset EDA, class balance, text stats
|   +-- 02_baseline_lstm.ipynb  # BiLSTM + GloVe from scratch
|   +-- 03_transformer_finetune.ipynb
|
+-- tests/                      # Test suite
|   +-- __init__.py
|   +-- conftest.py             # Shared pytest fixtures
|   +-- test_api.py             # 25+ API integration tests
|   +-- test_inference.py       # Unit tests: preprocessing, softmax, auth
|
+-- data/
|   +-- raw/                    # Downloaded datasets
|   +-- processed/              # EDA outputs, plots
|
+-- ui/
|   +-- index.html              # NEXUS frontend (Nginx, port 3000)
|
+-- Dockerfile                  # CUDA 12.8 + Ubuntu 22.04 image
+-- docker-compose.yml          # Full stack
+-- nginx.conf                  # Nginx static + API proxy
+-- prometheus.yml              # Prometheus scrape config
+-- requirements.txt            # All Python deps (CUDA 12.8 pinned)
+-- pytest.ini                  # pytest config
+-- .env.example                # Environment variable template
+-- download_datasets.py        # One-shot dataset downloader
+-- README.md
```

---

## Setup in Theia IDE

Theia IDE runs on a remote Ubuntu server. Open a terminal with `Terminal -> New Terminal` and follow every step in order.

---

### Step 1 — Check Prerequisites

```bash
lsb_release -a          # Ubuntu 22.04 recommended
python3 --version       # need 3.10+
nvidia-smi              # should show your GPU
nvcc --version          # should show CUDA version
docker --version
docker compose version  # need v2
```

---

### Step 2 — Install System Libraries

```bash
sudo apt-get update && sudo apt-get install -y \
  python3.11 python3.11-dev python3-pip python3.11-venv \
  build-essential gcc g++ make git curl wget \
  libssl-dev libffi-dev libxml2-dev libxslt1-dev \
  libjpeg-dev zlib1g-dev libgomp1 libopenblas-dev \
  redis-server redis-tools
```

---

### Step 3 — Install CUDA 12.8

Skip this step if `nvcc --version` already shows 12.6 or higher.

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-12-8 libcudnn9-cuda-12 libcudnn9-dev-cuda-12 cuda-drivers

echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

nvcc --version
nvidia-smi
```

---

### Step 4 — Create Python Virtual Environment

```bash
cd /path/to/sentiment-api

python3.11 -m venv .venv
source .venv/bin/activate

python --version          # Python 3.11.x
```

To auto-activate in every Theia terminal:
```bash
echo "source $(pwd)/.venv/bin/activate" >> ~/.bashrc
```

---

### Step 5 — Install PyTorch (CUDA 12.8)

PyTorch must be installed separately before requirements.txt:

```bash
pip install --upgrade pip setuptools wheel

pip install \
  torch==2.5.1+cu128 \
  torchvision==0.20.1+cu128 \
  torchaudio==2.5.1+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128
```

Verify GPU is visible:

```bash
python -c "
import torch
print('CUDA:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0))
print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory/1e9,1), 'GB')
"
```

---

### Step 6 — Install All Requirements

```bash
pip install -r requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu128

python -m spacy download en_core_web_sm

python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

---

### Step 7 — Set Environment Variables

```bash
cp .env.example .env
```

Edit `.env` in Theia (click the file in the explorer):

```
MODEL_DIR=models/checkpoints/final
ONNX_PATH=models/checkpoints/model_sim.onnx
USE_ONNX=true
MAX_LENGTH=128
API_KEYS=dev-key-123
REDIS_URL=redis://localhost:6379
CACHE_TTL=3600
PORT=8000
LOG_LEVEL=INFO
```

Load in terminal:

```bash
set -a && source .env && set +a
```

---

### Step 8 — Download Datasets

```bash
python download_datasets.py
```

Downloads SST-2, IMDB, Amazon Polarity, Yelp to `data/raw/`. Takes 5–15 minutes, ~3-4 GB.

For GloVe (LSTM notebook only):

```bash
wget http://nlp.stanford.edu/data/glove.6B.zip -P data/raw/
unzip data/raw/glove.6B.zip -d data/raw/
```

---

### Step 9 — Train the Model

```bash
python models/train.py
```

Options:

```bash
python models/train.py --model distilbert-base-uncased --epochs 4 --batch 32 --lr 2e-5
```

Watch GPU in another terminal:

```bash
watch -n 2 nvidia-smi
```

Training takes ~25 minutes on RTX 4060. Model saved to `models/checkpoints/final/`.

---

### Step 10 — Evaluate

```bash
python models/evaluate.py
```

Prints accuracy, macro F1, classification report.
Saves `models/checkpoints/confusion_matrix.png` — open in Theia's image viewer.

---

### Step 11 — Export to ONNX

```bash
python models/export_onnx.py
```

Saves `models/checkpoints/model_sim.onnx` and prints speedup benchmark.
Typical output: `PyTorch: 44ms | ONNX: 18ms | Speedup: 2.48x`

---

### Step 12 — Run API Locally

```bash
# Start Redis
sudo systemctl start redis

# Load env
set -a && source .env && set +a

# Start API
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Test:

```bash
# Health (no auth needed)
curl http://localhost:8000/api/v1/health

# Analyze
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Authorization: Bearer dev-key-123" \
  -H "Content-Type: application/json" \
  -d '{"text": "This is absolutely fantastic!"}'

# Batch
curl -X POST http://localhost:8000/api/v1/batch \
  -H "Authorization: Bearer dev-key-123" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Loved it!", "Terrible.", "It was okay."]}'
```

Open Swagger UI: `http://localhost:8000/docs`

---

### Step 13 — Run Tests

```bash
# All tests with coverage
pytest tests/ -v --cov=api --cov-report=term-missing

# API tests only
pytest tests/test_api.py -v

# Inference tests only
pytest tests/test_inference.py -v

# HTML report — open htmlcov/index.html in Theia
pytest tests/ --cov=api --cov-report=html
```

---

### Step 14 — Notebooks

```bash
source .venv/bin/activate
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

Open `http://localhost:8888` in Theia browser preview.

| Notebook | What it does |
|---|---|
| `01_eda.ipynb` | Class balance, length stats, word frequencies |
| `02_baseline_lstm.ipynb` | BiLSTM + GloVe from scratch, training loop |
| `03_transformer_finetune.ipynb` | DistilBERT fine-tune, curves, error analysis |

---

### Step 15 — Deploy with Docker

```bash
# Install GPU support for Docker
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU passthrough
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi

# Build and start everything
docker compose up --build -d

# Check status
docker compose ps
docker compose logs -f api

# Stop
docker compose down
```

Services after startup:

| URL | Service |
|---|---|
| http://localhost:3000 | NEXUS Frontend UI |
| http://localhost:8000/docs | API Swagger UI |
| http://localhost:8000/api/v1/health | Health check |
| http://localhost:9090 | Prometheus |

---

## API Reference

### Authentication

```
Authorization: Bearer <api-key>
```

Default: `dev-key-123` — change `API_KEYS` in `.env` for production.

### POST /api/v1/analyze

Request:
```json
{
  "text": "The product quality exceeded all my expectations!",
  "model": "distilbert-finetuned",
  "language": "en"
}
```

Response:
```json
{
  "label": "POSITIVE",
  "confidence": 0.9847,
  "scores": {
    "POSITIVE": 0.9847,
    "NEUTRAL":  0.0112,
    "NEGATIVE": 0.0041
  },
  "metadata": {
    "model_version": "v2.1.0",
    "processing_ms": 18.4,
    "token_count": 9,
    "cached": false
  }
}
```

### POST /api/v1/batch

```json
{ "texts": ["Great!", "Terrible.", "Okay."] }
```

### GET /api/v1/health

No auth required. Used by Docker healthcheck and load balancers.

### GET /metrics

Prometheus metrics. Scraped automatically by the Prometheus container.

---

## Model Details

| Property | Value |
|---|---|
| Base model | distilbert-base-uncased |
| Parameters | 66 million |
| Training precision | fp16 mixed precision |
| Max sequence length | 128 tokens |
| Output classes | NEGATIVE / NEUTRAL / POSITIVE |
| Target accuracy | >= 94% on SST-2 |
| ONNX latency | ~18ms (RTX 4060) |
| PyTorch latency | ~45ms (RTX 4060) |
| VRAM (training) | ~5.2 GB (batch=32, fp16) |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| MODEL_DIR | models/checkpoints/final | HuggingFace model path |
| ONNX_PATH | models/checkpoints/model_sim.onnx | ONNX model path |
| USE_ONNX | true | Use ONNX Runtime |
| MAX_LENGTH | 128 | Max tokenizer length |
| API_KEYS | dev-key-123 | Comma-separated valid keys |
| REDIS_URL | redis://localhost:6379 | Redis connection |
| CACHE_TTL | 3600 | Cache TTL seconds |
| PORT | 8000 | Server port |
| LOG_LEVEL | INFO | Logging level |

---

## Quick Command Reference

```bash
# Activate environment
source .venv/bin/activate
set -a && source .env && set +a

# Data
python download_datasets.py

# Training pipeline
python models/train.py
python models/evaluate.py
python models/export_onnx.py

# Run API
sudo systemctl start redis
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Tests
pytest tests/ -v --cov=api --cov-report=term-missing

# Notebooks
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

# Docker
docker compose up --build -d
docker compose ps
docker compose logs -f api
docker compose down
```

---

## Troubleshooting

**CUDA not available after install:**
```bash
source ~/.bashrc
nvcc --version
pip install torch==2.5.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128 --force-reinstall
```

**Out of GPU memory during training:**
```bash
python models/train.py --batch 16
```

**Redis connection refused:**
```bash
sudo systemctl start redis
redis-cli ping    # should return PONG
```

**Model not found on API start:**
```bash
python models/train.py    # must train first
```

**Docker GPU passthrough fails:**
```bash
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**Port already in use:**
```bash
sudo lsof -i :8000
sudo kill -9 <PID>
```

**Theia loses venv on new terminal:**
```bash
echo "source /path/to/sentiment-api/.venv/bin/activate" >> ~/.bashrc
```

---

## License

MIT — Dragon Buyareshi · Sentiment Analysis API
