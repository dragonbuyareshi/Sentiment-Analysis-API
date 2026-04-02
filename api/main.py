"""
api/main.py
───────────
FastAPI application — entry point for the Sentiment Analysis API.

Endpoints:
  POST /api/v1/analyze   — single text prediction
  POST /api/v1/batch     — batch prediction (up to 64 texts)
  GET  /api/v1/health    — liveness + readiness check
  GET  /api/v1/models    — list available models
  GET  /metrics          — Prometheus metrics (auto-instrumented)

Run locally:
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Swagger UI: http://localhost:8000/docs
ReDoc:       http://localhost:8000/redoc
"""

import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from prometheus_fastapi_instrumentator import Instrumentator

from api.auth import verify_api_key
from api.cache import CacheLayer
from api.inference import SentimentEngine
from api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BatchRequest,
    BatchResponse,
    HealthResponse,
    ModelsResponse,
)

# ── Singletons (initialised in lifespan) ──────────────────────────
engine: Optional[SentimentEngine] = None
cache:  Optional[CacheLayer]      = None

MODEL_VERSION = "v2.1.0"


# ── Lifespan (replaces deprecated on_event) ───────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, cache

    # ── Startup ───────────────────────────────────────────────────
    logger.info("⚙  Starting Sentiment Analysis API ...")

    cache = CacheLayer()
    await cache.connect()

    engine = SentimentEngine()
    engine.load()

    logger.info(f"✔ API ready — model={MODEL_VERSION}, engine={engine.info()}")
    yield

    # ── Shutdown ──────────────────────────────────────────────────
    logger.info("Shutting down ...")
    if cache:
        await cache.disconnect()
    logger.info("✔ Shutdown complete")


# ── App ───────────────────────────────────────────────────────────
app = FastAPI(
    title="Sentiment Analysis API",
    description=(
        "3-class sentiment analysis (POSITIVE / NEUTRAL / NEGATIVE) "
        "powered by fine-tuned DistilBERT with ONNX Runtime inference."
    ),
    version=MODEL_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Prometheus ────────────────────────────────────────────────────
Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    excluded_handlers=["/metrics", "/docs", "/redoc", "/openapi.json"],
).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


# ═══════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════

@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    summary="Health & readiness check",
    tags=["System"],
)
async def health() -> HealthResponse:
    """
    Returns API liveness and model readiness.
    No authentication required — suitable for load-balancer probes.
    """
    return HealthResponse(
        status="ok",
        model_loaded=engine is not None and engine.is_ready,
        cache_connected=cache is not None and cache.connected,
        model_version=MODEL_VERSION,
    )


@app.get(
    "/api/v1/models",
    response_model=ModelsResponse,
    summary="List available models",
    tags=["System"],
)
async def list_models() -> ModelsResponse:
    """Returns the list of available sentiment models and their properties."""
    return ModelsResponse(
        models=[
            {
                "id":          "distilbert-finetuned",
                "description": "DistilBERT fine-tuned on SST-2 (PyTorch)",
                "labels":      3,
                "latency_ms":  45,
                "default":     False,
            },
            {
                "id":          "distilbert-onnx",
                "description": "DistilBERT fine-tuned on SST-2 (ONNX Runtime, default)",
                "labels":      3,
                "latency_ms":  18,
                "default":     True,
            },
        ]
    )


@app.post(
    "/api/v1/analyze",
    response_model=AnalyzeResponse,
    summary="Analyze sentiment of a single text",
    tags=["Sentiment"],
)
async def analyze(
    req: AnalyzeRequest,
    _api_key: str = Depends(verify_api_key),
) -> AnalyzeResponse:
    """
    Classify a single piece of text into POSITIVE / NEUTRAL / NEGATIVE.

    - Checks Redis cache first (key = SHA-256 of text)
    - On cache miss: runs inference and caches the result
    - Returns label, per-class confidence scores, and metadata
    """
    # ── Guard: empty / whitespace ──────────────────────────────────
    if not req.text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="text cannot be empty or whitespace only",
        )

    # ── Guard: too long ────────────────────────────────────────────
    if len(req.text) > 5000:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="text exceeds 5,000 character limit",
        )

    # ── Cache lookup ───────────────────────────────────────────────
    cached = await cache.get(req.text)
    if cached:
        cached.setdefault("metadata", {})["cached"] = True
        return AnalyzeResponse(**cached)

    # ── Inference ──────────────────────────────────────────────────
    t0     = time.perf_counter()
    result = engine.predict(req.text)
    ms     = round((time.perf_counter() - t0) * 1000, 1)

    response_dict = {
        "label":      result["label"],
        "confidence": result["confidence"],
        "scores":     result["scores"],
        "metadata": {
            "model_version": MODEL_VERSION,
            "processing_ms": ms,
            "token_count":   result["token_count"],
            "cached":        False,
        },
    }

    # ── Cache store ────────────────────────────────────────────────
    await cache.set(req.text, response_dict)

    return AnalyzeResponse(**response_dict)


@app.post(
    "/api/v1/batch",
    response_model=BatchResponse,
    summary="Analyze sentiment for a batch of texts",
    tags=["Sentiment"],
)
async def batch_analyze(
    req: BatchRequest,
    _api_key: str = Depends(verify_api_key),
) -> BatchResponse:
    """
    Classify up to 64 texts in a single batched GPU forward pass.
    More efficient than calling /analyze in a loop.
    """
    if len(req.texts) > 64:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 64 texts per batch request",
        )

    # Filter blanks
    texts = [t for t in req.texts if t.strip()]
    if not texts:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="All provided texts are empty",
        )

    t0      = time.perf_counter()
    results = engine.predict_batch(texts)
    ms      = round((time.perf_counter() - t0) * 1000, 1)

    return BatchResponse(
        results=results,
        count=len(results),
        processing_ms=ms,
    )


# ── Global exception handler ──────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception):
    logger.exception(f"Unhandled error on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again."},
    )
