# SENTIMENT ANALYSIS API — NEURAL INFERENCE SYSTEM

> *“From raw human emotion → structured intelligence in <80ms.”*

A **production-ready, high-performance Sentiment Analysis API** powered by Transformer models, optimized with ONNX, and deployed via FastAPI.

---

## OVERVIEW

This project is not just a basic NLP model — it is a **full-stack AI inference system** designed for:

* Real-time sentiment classification
* Transformer-based deep understanding
* Production deployment with caching, monitoring, and scaling
* 3-class classification: **POSITIVE / NEUTRAL / NEGATIVE**

---

## SYSTEM ARCHITECTURE

```
Client Request (JSON)
        ↓
API Gateway (FastAPI)
        ↓
Preprocessing Pipeline
        ↓
Transformer Model (DistilBERT)
        ↓
ONNX Runtime (Optimized Inference)
        ↓
Redis Cache Layer
        ↓
JSON Response
```

### Key Components

| Layer            | Description                                     |
| ---------------- | ----------------------------------------------- |
| **API Layer**    | FastAPI with validation & async handling        |
| **NLP Pipeline** | Tokenization, normalization, language detection |
| **Model Engine** | Fine-tuned DistilBERT / RoBERTa                 |
| **Optimization** | ONNX Runtime (GPU/CPU acceleration)             |
| **Cache**        | Redis (low latency repeated queries)            |
| **Monitoring**   | Prometheus + logging                            |

---

## FEATURES

* **<80ms inference latency**
* **94%+ accuracy target**
* **Transformer-based deep learning**
* **ONNX optimized deployment**
* Modular architecture (train → evaluate → deploy)
* Supports **batch inference**
* Redis-based caching
* Built-in evaluation pipeline

---

## PROJECT STRUCTURE

```
sentiment-api/
│
├── models/
│   ├── train.py              # Training pipeline
│   ├── evaluate.py          # Metrics & evaluation
│   ├── export_onnx.py       # ONNX conversion
│
├── api/
│   ├── main.py              # FastAPI server
│   ├── inference.py         # Model inference logic
│   ├── schemas.py           # Request/response models
│   ├── cache.py             # Redis caching layer
│
├── data/
│   └── raw/                 # Datasets (SST-2, IMDB, etc.)
│
├── docker/
│   └── Dockerfile
│
├── requirements.txt
└── README.md
```

---

## DATASETS

| Dataset           | Purpose                   |
| ----------------- | ------------------------- |
| SST-2             | Primary training dataset  |
| IMDB              | Long-text augmentation    |
| Amazon Reviews    | Multi-domain learning     |
| Yelp              | Domain robustness         |
| Twitter Sentiment | Short-text generalization |

---

## INSTALLATION

### Clone Repository

```bash
git clone https://github.com/your-repo/sentiment-api.git
cd sentiment-api
```

---

### Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## DATASET SETUP

Download datasets automatically:

```bash
python download_datasets.py
```

Data will be stored in:

```
data/raw/
```

---

## MODEL TRAINING

Train Transformer model:

```bash
python models/train.py
```

### Training Details

* Model: `distilbert-base-uncased`
* Precision: `fp16`
* Batch Size: `32`
* Epochs: `4 + 2 (fine-tuning)`
* Output:

```
models/checkpoints/final/
```

---

## MODEL EVALUATION

```bash
python models/evaluate.py --model models/checkpoints/final
```

### Metrics Generated

* Accuracy
* F1 Score (macro & weighted)
* Confusion Matrix

---

## ONNX OPTIMIZATION

Export trained model:

```bash
python models/export_onnx.py
```

### Benefits

* Faster inference
* Lower memory usage
* Production-ready deployment

---

## RUN API SERVER

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

---

## API ENDPOINTS

### Sentiment Prediction

```http
POST /predict
```

#### Request

```json
{
  "text": "This product is amazing!"
}
```

#### Response

```json
{
  "label": "POSITIVE",
  "confidence": 0.97,
  "processing_time_ms": 42
}
```

---

## REDIS CACHING

* Stores repeated queries
* TTL: **1 hour**
* Reduces latency drastically

Run Redis:

```bash
docker run -d -p 6379:6379 redis
```

---

## DOCKER DEPLOYMENT

### Build Image

```bash
docker build -t sentiment-api .
```

### Run Container

```bash
docker run -p 8000:8000 sentiment-api
```

---

## PERFORMANCE TARGETS

| Metric     | Value        |
| ---------- | ------------ |
| Accuracy   | 94%+         |
| Latency    | <80ms        |
| Throughput | 100+ req/sec |
| Model Size | ~265MB       |

---

## MODEL STRATEGY

| Model        | Use Case               |
| ------------ | ---------------------- |
| DistilBERT   | Fast, production-ready |
| RoBERTa      | Higher accuracy        |
| LSTM + GloVe | Lightweight fallback   |

---

## ADVANCED OPTIMIZATIONS

* Mixed Precision Training (fp16)
* Gradient Accumulation
* Layer Freezing + Progressive Unfreezing
* ONNX Graph Optimization
* Batch Inference Pipeline

---

## HARDWARE REQUIREMENTS

| Component | Minimum        |
| --------- | -------------- |
| GPU       | RTX 4060 (8GB) |
| RAM       | 16GB           |
| Storage   | 200GB          |
| CUDA      | 12.x           |

---

## TESTING

```bash
pytest
```

Load testing:

```bash
locust
```

---

## MONITORING

* Prometheus metrics
* Request latency tracking
* Model drift detection
* Structured logs (Loguru)

---

## FUTURE EXTENSIONS

* Multilingual sentiment detection
* Emotion classification (joy, anger, etc.)
* Zero-shot sentiment models
* Continual learning pipeline
* Edge deployment (TensorRT)


