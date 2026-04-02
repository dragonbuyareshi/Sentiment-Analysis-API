"""
tests/test_inference.py
───────────────────────
Unit tests for the SentimentEngine inference layer.
Run: pytest tests/test_inference.py -v
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock


# ══════════════════════════════════════════════════════════════════
# TEXT PREPROCESSING TESTS
# ══════════════════════════════════════════════════════════════════

class TestPreprocessing:
    """Tests for the _preprocess method in SentimentEngine."""

    @pytest.fixture
    def engine(self):
        with patch("api.inference.AutoTokenizer"), patch("api.inference.AutoModelForSequenceClassification"):
            from api.inference import SentimentEngine
            eng = SentimentEngine()
            eng.is_ready = True
            return eng

    def test_strips_html_entities(self, engine):
        result = engine._preprocess("&amp; great &lt;product&gt;")
        assert "&amp;" not in result
        assert "great" in result

    def test_converts_emoji_to_text(self, engine):
        result = engine._preprocess("Great product 😊")
        assert "😊" not in result  # emoji demojized

    def test_replaces_urls(self, engine):
        result = engine._preprocess("Check https://example.com for details")
        assert "https://example.com" not in result
        assert "[URL]" in result

    def test_collapses_whitespace(self, engine):
        result = engine._preprocess("too   many    spaces")
        assert "  " not in result

    def test_strips_leading_trailing_whitespace(self, engine):
        result = engine._preprocess("  padded text  ")
        assert result == result.strip()

    def test_empty_string(self, engine):
        result = engine._preprocess("")
        assert isinstance(result, str)

    def test_long_text_truncated(self, engine):
        long_text = "word " * 1000
        result = engine._preprocess(long_text)
        assert len(result) <= 128 * 6 + 10  # rough bound

    def test_normal_text_unchanged_structure(self, engine):
        text = "The product is great."
        result = engine._preprocess(text)
        assert "product" in result
        assert "great" in result


# ══════════════════════════════════════════════════════════════════
# SOFTMAX TESTS
# ══════════════════════════════════════════════════════════════════

class TestSoftmax:
    @pytest.fixture
    def engine(self):
        with patch("api.inference.AutoTokenizer"), patch("api.inference.AutoModelForSequenceClassification"):
            from api.inference import SentimentEngine
            return SentimentEngine()

    def test_softmax_sums_to_one(self, engine):
        logits = np.array([[2.0, 1.0, 0.5]])
        probs = engine._softmax(logits)
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_softmax_all_outputs_positive(self, engine):
        logits = np.array([[-5.0, 0.0, 5.0]])
        probs = engine._softmax(logits)
        assert (probs >= 0).all()

    def test_softmax_max_index_matches_argmax_logit(self, engine):
        logits = np.array([[1.0, 5.0, 2.0]])
        probs = engine._softmax(logits)
        assert np.argmax(probs) == np.argmax(logits)

    def test_softmax_numerically_stable_large_values(self, engine):
        logits = np.array([[1000.0, 1001.0, 1002.0]])
        probs = engine._softmax(logits)
        assert not np.isnan(probs).any()
        assert not np.isinf(probs).any()

    def test_softmax_batch(self, engine):
        logits = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        probs = engine._softmax(logits)
        assert probs.shape == (2, 3)
        for row in probs:
            assert abs(row.sum() - 1.0) < 1e-6


# ══════════════════════════════════════════════════════════════════
# PREDICT OUTPUT STRUCTURE
# ══════════════════════════════════════════════════════════════════

class TestPredictOutput:
    @pytest.fixture
    def loaded_engine(self):
        """Engine with mocked tokenizer + inference."""
        with (
            patch("api.inference.AutoTokenizer") as mock_tok_cls,
            patch("api.inference.AutoModelForSequenceClassification"),
            patch("api.inference.ort"),
            patch("api.inference.os.path.exists", return_value=False),
        ):
            from api.inference import SentimentEngine
            eng = SentimentEngine()
            eng.is_ready = True

            # Mock tokenizer output
            mock_tokenizer = MagicMock()
            mock_tokenizer.return_value = {
                "input_ids": np.array([[101, 2023, 102]]),
                "attention_mask": np.array([[1, 1, 1]]),
            }
            eng.tokenizer = mock_tokenizer

            # Mock inference → logits
            eng._run_inference = MagicMock(return_value=np.array([[1.0, 2.0, 5.0]]))
            return eng

    def test_predict_returns_dict(self, loaded_engine):
        result = loaded_engine.predict("Great product!")
        assert isinstance(result, dict)

    def test_predict_has_required_keys(self, loaded_engine):
        result = loaded_engine.predict("Great product!")
        assert set(result.keys()) >= {"label", "confidence", "scores", "token_count"}

    def test_predict_label_is_valid(self, loaded_engine):
        result = loaded_engine.predict("Great product!")
        assert result["label"] in ["POSITIVE", "NEUTRAL", "NEGATIVE"]

    def test_predict_confidence_range(self, loaded_engine):
        result = loaded_engine.predict("Great product!")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_scores_keys(self, loaded_engine):
        result = loaded_engine.predict("Great product!")
        assert set(result["scores"].keys()) == {"POSITIVE", "NEUTRAL", "NEGATIVE"}

    def test_predict_scores_sum_to_one(self, loaded_engine):
        result = loaded_engine.predict("Great product!")
        assert abs(sum(result["scores"].values()) - 1.0) < 1e-4

    def test_predict_confidence_is_max_score(self, loaded_engine):
        result = loaded_engine.predict("Great product!")
        assert abs(result["confidence"] - max(result["scores"].values())) < 1e-4

    def test_predict_batch_returns_list(self, loaded_engine):
        loaded_engine._run_inference = MagicMock(
            return_value=np.array([[1.0, 2.0, 5.0], [5.0, 2.0, 1.0]])
        )
        results = loaded_engine.predict_batch(["Great!", "Terrible!"])
        assert isinstance(results, list)
        assert len(results) == 2

    def test_predict_batch_each_has_label(self, loaded_engine):
        loaded_engine._run_inference = MagicMock(
            return_value=np.array([[1.0, 2.0, 5.0], [5.0, 2.0, 1.0]])
        )
        results = loaded_engine.predict_batch(["Great!", "Terrible!"])
        for r in results:
            assert "label" in r
            assert "confidence" in r
            assert "scores" in r


# ══════════════════════════════════════════════════════════════════
# LABEL MAPPING
# ══════════════════════════════════════════════════════════════════

class TestLabelMapping:
    @pytest.fixture
    def engine(self):
        with patch("api.inference.AutoTokenizer"), patch("api.inference.AutoModelForSequenceClassification"):
            from api.inference import SentimentEngine
            eng = SentimentEngine()
            eng.tokenizer = MagicMock(return_value={"input_ids": np.zeros((1, 5)), "attention_mask": np.ones((1, 5))})
            eng.is_ready = True
            return eng

    @pytest.mark.parametrize("logits,expected_label", [
        (np.array([[5.0, 1.0, 0.5]]), "NEGATIVE"),
        (np.array([[0.5, 5.0, 1.0]]), "NEUTRAL"),
        (np.array([[0.5, 1.0, 5.0]]), "POSITIVE"),
    ])
    def test_label_from_argmax(self, engine, logits, expected_label):
        engine._run_inference = MagicMock(return_value=logits)
        result = engine.predict("test text")
        assert result["label"] == expected_label


# ══════════════════════════════════════════════════════════════════
# CACHE TESTS
# ══════════════════════════════════════════════════════════════════

class TestCacheLayer:
    @pytest.fixture
    def cache(self):
        with patch("api.cache.aioredis"):
            from api.cache import CacheLayer
            return CacheLayer()

    def test_key_generation_deterministic(self, cache):
        key1 = cache._key("hello world")
        key2 = cache._key("hello world")
        assert key1 == key2

    def test_different_texts_different_keys(self, cache):
        key1 = cache._key("positive text")
        key2 = cache._key("negative text")
        assert key1 != key2

    def test_key_format(self, cache):
        key = cache._key("some text")
        assert key.startswith("sentiment:v2:")

    def test_key_length_bounded(self, cache):
        long_text = "word " * 1000
        key = cache._key(long_text)
        assert len(key) < 100  # SHA256 hex is fixed length


# ══════════════════════════════════════════════════════════════════
# AUTH TESTS
# ══════════════════════════════════════════════════════════════════

class TestAuth:
    @pytest.mark.asyncio
    async def test_valid_key_passes(self):
        with patch("api.auth.VALID_KEYS", {"valid-key"}):
            from api.auth import verify_api_key
            result = await verify_api_key("Bearer valid-key")
            assert result == "valid-key"

    @pytest.mark.asyncio
    async def test_invalid_key_raises_403(self):
        from fastapi import HTTPException
        with patch("api.auth.VALID_KEYS", {"valid-key"}):
            from api.auth import verify_api_key
            with pytest.raises(HTTPException) as exc_info:
                await verify_api_key("Bearer bad-key")
            assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_no_auth_raises_401(self):
        from fastapi import HTTPException
        from api.auth import verify_api_key
        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(None)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_malformed_header_raises_401(self):
        from fastapi import HTTPException
        from api.auth import verify_api_key
        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key("just-a-key-no-bearer")
        assert exc_info.value.status_code == 401
