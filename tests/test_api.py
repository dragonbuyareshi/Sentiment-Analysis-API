"""
tests/test_api.py
─────────────────
Integration & unit tests for the FastAPI sentiment endpoints.
Run: pytest tests/test_api.py -v --cov=api --cov-report=term-missing
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport

# ── Fixtures ──────────────────────────────────────────────────────

MOCK_PREDICT_RESULT = {
    "label": "POSITIVE",
    "confidence": 0.9847,
    "scores": {"NEGATIVE": 0.0041, "NEUTRAL": 0.0112, "POSITIVE": 0.9847},
    "token_count": 9,
}

MOCK_RESPONSE = {
    "label": "POSITIVE",
    "confidence": 0.9847,
    "scores": {"NEGATIVE": 0.0041, "NEUTRAL": 0.0112, "POSITIVE": 0.9847},
    "metadata": {
        "model_version": "v2.1.0",
        "processing_ms": 42.5,
        "token_count": 9,
        "cached": False,
    },
}

MOCK_BATCH_RESULTS = [
    {"label": "POSITIVE",  "confidence": 0.9847, "scores": {"NEGATIVE": 0.0041, "NEUTRAL": 0.0112, "POSITIVE": 0.9847}},
    {"label": "NEGATIVE",  "confidence": 0.9201, "scores": {"NEGATIVE": 0.9201, "NEUTRAL": 0.0542, "POSITIVE": 0.0257}},
    {"label": "NEUTRAL",   "confidence": 0.7103, "scores": {"NEGATIVE": 0.1521, "NEUTRAL": 0.7103, "POSITIVE": 0.1376}},
]

VALID_API_KEY = "Bearer dev-key-123"


@pytest.fixture(scope="module")
def mock_engine():
    engine = MagicMock()
    engine.is_ready = True
    engine.predict.return_value = MOCK_PREDICT_RESULT
    engine.predict_batch.return_value = MOCK_BATCH_RESULTS
    return engine


@pytest.fixture(scope="module")
def mock_cache():
    cache = AsyncMock()
    cache.connected = True
    cache.get.return_value = None  # cache miss by default
    cache.set.return_value = None
    return cache


@pytest.fixture(scope="module")
async def client(mock_engine, mock_cache):
    """Async test client with mocked engine + cache."""
    with (
        patch("api.main.engine", mock_engine),
        patch("api.main.cache", mock_cache),
        patch("api.auth.VALID_KEYS", {"dev-key-123"}),
    ):
        from api.main import app
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            yield ac


# ══════════════════════════════════════════════════════════════════
# HEALTH ENDPOINT
# ══════════════════════════════════════════════════════════════════

class TestHealth:
    @pytest.mark.asyncio
    async def test_health_returns_200(self, client):
        r = await client.get("/api/v1/health")
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_health_payload_structure(self, client):
        r = await client.get("/api/v1/health")
        data = r.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "cache_connected" in data
        assert "model_version" in data

    @pytest.mark.asyncio
    async def test_health_status_ok(self, client):
        r = await client.get("/api/v1/health")
        assert r.json()["status"] == "ok"


# ══════════════════════════════════════════════════════════════════
# ANALYZE ENDPOINT
# ══════════════════════════════════════════════════════════════════

class TestAnalyze:
    @pytest.mark.asyncio
    async def test_analyze_success(self, client, mock_engine):
        mock_engine.predict.return_value = MOCK_PREDICT_RESULT
        r = await client.post(
            "/api/v1/analyze",
            json={"text": "This product is absolutely amazing!"},
            headers={"Authorization": VALID_API_KEY},
        )
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_analyze_response_schema(self, client):
        r = await client.post(
            "/api/v1/analyze",
            json={"text": "Great service, highly recommend."},
            headers={"Authorization": VALID_API_KEY},
        )
        data = r.json()
        assert "label" in data
        assert "confidence" in data
        assert "scores" in data
        assert "metadata" in data
        assert data["label"] in ["POSITIVE", "NEUTRAL", "NEGATIVE"]
        assert 0.0 <= data["confidence"] <= 1.0
        assert set(data["scores"].keys()) == {"POSITIVE", "NEUTRAL", "NEGATIVE"}

    @pytest.mark.asyncio
    async def test_analyze_scores_sum_to_one(self, client):
        r = await client.post(
            "/api/v1/analyze",
            json={"text": "Decent but not great."},
            headers={"Authorization": VALID_API_KEY},
        )
        scores = r.json()["scores"]
        assert abs(sum(scores.values()) - 1.0) < 1e-3

    @pytest.mark.asyncio
    async def test_analyze_metadata_fields(self, client):
        r = await client.post(
            "/api/v1/analyze",
            json={"text": "Works as expected."},
            headers={"Authorization": VALID_API_KEY},
        )
        meta = r.json()["metadata"]
        assert "processing_ms" in meta
        assert "token_count" in meta
        assert "cached" in meta
        assert "model_version" in meta

    @pytest.mark.asyncio
    async def test_analyze_no_auth_returns_401(self, client):
        r = await client.post(
            "/api/v1/analyze",
            json={"text": "some text"},
        )
        assert r.status_code == 401

    @pytest.mark.asyncio
    async def test_analyze_invalid_key_returns_403(self, client):
        r = await client.post(
            "/api/v1/analyze",
            json={"text": "some text"},
            headers={"Authorization": "Bearer invalid-key-xyz"},
        )
        assert r.status_code == 403

    @pytest.mark.asyncio
    async def test_analyze_empty_text_returns_422(self, client):
        r = await client.post(
            "/api/v1/analyze",
            json={"text": "   "},
            headers={"Authorization": VALID_API_KEY},
        )
        assert r.status_code == 422

    @pytest.mark.asyncio
    async def test_analyze_too_long_text_returns_413(self, client):
        r = await client.post(
            "/api/v1/analyze",
            json={"text": "x" * 5001},
            headers={"Authorization": VALID_API_KEY},
        )
        assert r.status_code == 413

    @pytest.mark.asyncio
    async def test_analyze_cache_hit(self, client, mock_cache):
        """When cache has a result, it should return it with cached=True."""
        cached_response = {**MOCK_RESPONSE, "metadata": {**MOCK_RESPONSE["metadata"], "cached": True}}
        mock_cache.get.return_value = cached_response
        r = await client.post(
            "/api/v1/analyze",
            json={"text": "Cached text here."},
            headers={"Authorization": VALID_API_KEY},
        )
        assert r.status_code == 200
        assert r.json()["metadata"]["cached"] is True
        mock_cache.get.return_value = None  # reset


# ══════════════════════════════════════════════════════════════════
# BATCH ENDPOINT
# ══════════════════════════════════════════════════════════════════

class TestBatch:
    @pytest.mark.asyncio
    async def test_batch_success(self, client):
        r = await client.post(
            "/api/v1/batch",
            json={"texts": ["Great!", "Terrible.", "It was okay."]},
            headers={"Authorization": VALID_API_KEY},
        )
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_batch_response_schema(self, client):
        r = await client.post(
            "/api/v1/batch",
            json={"texts": ["text one", "text two"]},
            headers={"Authorization": VALID_API_KEY},
        )
        data = r.json()
        assert "results" in data
        assert "count" in data
        assert "processing_ms" in data

    @pytest.mark.asyncio
    async def test_batch_count_matches_input(self, client, mock_engine):
        texts = ["text one", "text two", "text three"]
        mock_engine.predict_batch.return_value = MOCK_BATCH_RESULTS[:3]
        r = await client.post(
            "/api/v1/batch",
            json={"texts": texts},
            headers={"Authorization": VALID_API_KEY},
        )
        assert r.json()["count"] == len(texts)

    @pytest.mark.asyncio
    async def test_batch_too_many_texts_returns_400(self, client):
        r = await client.post(
            "/api/v1/batch",
            json={"texts": ["text"] * 65},
            headers={"Authorization": VALID_API_KEY},
        )
        assert r.status_code == 400

    @pytest.mark.asyncio
    async def test_batch_single_text(self, client, mock_engine):
        mock_engine.predict_batch.return_value = [MOCK_BATCH_RESULTS[0]]
        r = await client.post(
            "/api/v1/batch",
            json={"texts": ["single text"]},
            headers={"Authorization": VALID_API_KEY},
        )
        assert r.status_code == 200
        assert r.json()["count"] == 1


# ══════════════════════════════════════════════════════════════════
# MODELS ENDPOINT
# ══════════════════════════════════════════════════════════════════

class TestModels:
    @pytest.mark.asyncio
    async def test_models_returns_200(self, client):
        r = await client.get("/api/v1/models")
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_models_has_default(self, client):
        r = await client.get("/api/v1/models")
        models = r.json()["models"]
        defaults = [m for m in models if m.get("default")]
        assert len(defaults) >= 1


# ══════════════════════════════════════════════════════════════════
# INPUT EDGE CASES
# ══════════════════════════════════════════════════════════════════

class TestEdgeCases:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("text", [
        "👍 amazing product! 🔥",
        "Très bien! Vraiment satisfait.",
        "1234567890 numbers only",
        "!@#$%^&* special chars",
        "<b>HTML</b> <script>alert(1)</script>",
        "a" * 4999,  # just under limit
    ])
    async def test_various_inputs_accepted(self, client, text):
        r = await client.post(
            "/api/v1/analyze",
            json={"text": text},
            headers={"Authorization": VALID_API_KEY},
        )
        assert r.status_code == 200, f"Failed for input: {text[:50]}"

    @pytest.mark.asyncio
    async def test_missing_text_field_returns_422(self, client):
        r = await client.post(
            "/api/v1/analyze",
            json={"wrong_field": "some text"},
            headers={"Authorization": VALID_API_KEY},
        )
        assert r.status_code == 422

    @pytest.mark.asyncio
    async def test_non_json_body_returns_422(self, client):
        r = await client.post(
            "/api/v1/analyze",
            content="not json",
            headers={"Authorization": VALID_API_KEY, "Content-Type": "application/json"},
        )
        assert r.status_code == 422
