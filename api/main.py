"""
FastAPI Sentiment Analysis API
Endpoints: POST /api/v1/analyze | POST /api/v1/batch
           GET  /api/v1/health  | GET  /api/v1/models
"""

import time, asyncio
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from prometheus_fastapi_instrumentator import Instrumentator

from api.schemas import AnalyzeRequest, AnalyzeResponse, BatchRequest, BatchResponse, HealthResponse
from api.inference import SentimentEngine
from api.cache import CacheLayer
from api.auth import verify_api_key

# ── Globals (loaded once at startup) ──────────────────
engine: SentimentEngine = None
cache: CacheLayer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, cache
    logger.info("⚙ Loading model...")
    engine = SentimentEngine()
    engine.load()
    cache = CacheLayer()
    await cache.connect()
    logger.info("✔ API ready")
    yield
    await cache.disconnect()


app = FastAPI(
    title="Sentiment Analysis API",
    version="2.1.0",
    description="3-class sentiment (POSITIVE/NEUTRAL/NEGATIVE) powered by fine-tuned DistilBERT",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app, endpoint="/metrics")


# ── Health ────────────────────────────────────────────
@app.get("/api/v1/health", response_model=HealthResponse)
async def health():
    return {
        "status": "ok",
        "model_loaded": engine is not None and engine.is_ready,
        "cache_connected": cache is not None and cache.connected,
        "model_version": "v2.1.0",
    }


# ── Single Analysis ───────────────────────────────────
@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
async def analyze(
    req: AnalyzeRequest,
    _: str = Depends(verify_api_key),
):
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text cannot be empty")
    if len(req.text) > 5000:
        raise HTTPException(status_code=413, detail="text exceeds 5000 character limit")

    # Check Redis cache first
    cached = await cache.get(req.text)
    if cached:
        cached["metadata"]["cached"] = True
        return cached

    t0 = time.perf_counter()
    result = engine.predict(req.text)
    ms = round((time.perf_counter() - t0) * 1000, 1)

    response = {
        "label": result["label"],
        "confidence": result["confidence"],
        "scores": result["scores"],
        "metadata": {
            "model_version": "v2.1.0",
            "processing_ms": ms,
            "token_count": result["token_count"],
            "cached": False,
        },
    }
    await cache.set(req.text, response)
    return response


# ── Batch Analysis (up to 64 texts) ───────────────────
@app.post("/api/v1/batch", response_model=BatchResponse)
async def batch_analyze(
    req: BatchRequest,
    _: str = Depends(verify_api_key),
):
    if len(req.texts) > 64:
        raise HTTPException(status_code=400, detail="max 64 texts per batch")

    t0 = time.perf_counter()
    results = engine.predict_batch(req.texts)
    ms = round((time.perf_counter() - t0) * 1000, 1)

    return {"results": results, "count": len(results), "processing_ms": ms}


# ── Available Models ──────────────────────────────────
@app.get("/api/v1/models")
async def list_models():
    return {"models": [
        {"id": "distilbert-finetuned", "labels": 3, "latency_ms": 45, "default": True},
        {"id": "distilbert-onnx",      "labels": 3, "latency_ms": 18, "default": False},
    ]}