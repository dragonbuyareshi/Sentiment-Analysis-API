"""
api/inference.py
────────────────
Singleton SentimentEngine — loads once at startup, serves all requests.

Supports two backends:
  1. ONNX Runtime (preferred) — 2-3x faster than PyTorch
  2. PyTorch             — fallback when ONNX model is absent

Auto-detects GPU. Falls back to CPU gracefully on OOM or unavailability.
"""

import html as html_lib
import os
import re
from typing import Dict, List, Optional

import emoji
import numpy as np
import torch

from loguru import logger

# ── Constants ──────────────────────────────────────────────────────
MODEL_DIR  = os.getenv("MODEL_DIR",  "models/checkpoints/final")
ONNX_PATH  = os.getenv("ONNX_PATH",  "models/checkpoints/model_sim.onnx")
USE_ONNX   = os.getenv("USE_ONNX",   "true").lower() == "true"
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "128"))

LABELS   = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
ID2LABEL = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}


class SentimentEngine:
    """
    Thread-safe, singleton inference engine.

    Load once:
        engine = SentimentEngine()
        engine.load()

    Then predict:
        result = engine.predict("Great product!")
        results = engine.predict_batch(["Great!", "Terrible."])
    """

    def __init__(self) -> None:
        self.tokenizer   = None
        self.model       = None          # PyTorch model
        self.ort_session = None          # ONNX Runtime session
        self.is_ready    = False
        self.backend     = "none"
        self.device      = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Startup ────────────────────────────────────────────────────

    def load(self) -> None:
        """Load tokenizer + model. Call once at application startup."""
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        logger.info(f"Loading tokenizer from {MODEL_DIR} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

        # ── Try ONNX first ────────────────────────────────────────
        if USE_ONNX and os.path.exists(ONNX_PATH):
            self._load_onnx()
        else:
            self._load_pytorch()

        self.is_ready = True
        logger.info(f"✔ SentimentEngine ready — backend={self.backend}, device={self.device}")

    def _load_onnx(self) -> None:
        import onnxruntime as ort

        logger.info(f"Loading ONNX model from {ONNX_PATH} ...")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = min(4, os.cpu_count() or 4)

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.device == "cuda"
            else ["CPUExecutionProvider"]
        )

        self.ort_session = ort.InferenceSession(ONNX_PATH, opts, providers=providers)
        self.backend = "onnx"
        logger.info(f"  ONNX providers: {self.ort_session.get_providers()}")

    def _load_pytorch(self) -> None:
        from transformers import AutoModelForSequenceClassification

        logger.info(f"Loading PyTorch model from {MODEL_DIR} ...")
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        self.model.eval()

        try:
            self.model = self.model.to(self.device)
            if self.device == "cuda":
                torch.backends.cudnn.benchmark = True
            self.backend = f"pytorch-{self.device}"
        except RuntimeError as exc:
            logger.warning(f"GPU load failed ({exc}), falling back to CPU")
            self.device = "cpu"
            self.model = self.model.to("cpu")
            self.backend = "pytorch-cpu"

    # ── Text Preprocessing ─────────────────────────────────────────

    def _preprocess(self, text: str) -> str:
        """Clean raw input text before tokenization."""
        text = html_lib.unescape(text)                          # &amp; → &
        text = emoji.demojize(text, delimiters=(" ", " "))      # 😊 → smiling face
        text = re.sub(r"https?://\S+|www\.\S+", " [URL] ", text)
        text = re.sub(r"<[^>]+>", " ", text)                    # strip HTML tags
        text = re.sub(r"\s+", " ", text).strip()
        # Hard cap before tokenizer sees it (rough char limit)
        return text[: MAX_LENGTH * 8]

    # ── Tokenization ───────────────────────────────────────────────

    def _tokenize(self, texts: List[str]) -> Dict:
        return_tensors = "np" if self.ort_session else "pt"
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors=return_tensors,
        )

    # ── Inference ──────────────────────────────────────────────────

    def _run_inference(self, encoded: Dict) -> np.ndarray:
        """Run forward pass, return raw logits as numpy array."""
        if self.ort_session:
            logits = self.ort_session.run(
                None,
                {
                    "input_ids":      encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                },
            )[0]
        else:
            enc = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                logits = self.model(**enc).logits.cpu().numpy()
        return logits

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exp     = np.exp(shifted)
        return exp / exp.sum(axis=-1, keepdims=True)

    # ── Public Prediction API ──────────────────────────────────────

    def predict(self, text: str) -> Dict:
        """
        Predict sentiment for a single text.

        Returns:
            {
                "label":       "POSITIVE" | "NEUTRAL" | "NEGATIVE",
                "confidence":  float,
                "scores":      {"POSITIVE": float, "NEUTRAL": float, "NEGATIVE": float},
                "token_count": int,
            }
        """
        clean   = self._preprocess(text)
        encoded = self._tokenize([clean])
        logits  = self._run_inference(encoded)
        probs   = self._softmax(logits)[0]
        idx     = int(np.argmax(probs))

        return {
            "label":       LABELS[idx],
            "confidence":  round(float(probs[idx]), 4),
            "scores": {
                label: round(float(p), 4)
                for label, p in zip(LABELS, probs)
            },
            "token_count": int(encoded["input_ids"].shape[1]),
        }

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Predict sentiment for a list of texts in a single forward pass.
        More efficient than calling predict() in a loop.
        """
        if not texts:
            return []

        cleans  = [self._preprocess(t) for t in texts]
        encoded = self._tokenize(cleans)
        logits  = self._run_inference(encoded)
        probs_all = self._softmax(logits)

        results = []
        for probs in probs_all:
            idx = int(np.argmax(probs))
            results.append({
                "label":      LABELS[idx],
                "confidence": round(float(probs[idx]), 4),
                "scores": {
                    label: round(float(p), 4)
                    for label, p in zip(LABELS, probs)
                },
            })
        return results

    # ── Diagnostics ────────────────────────────────────────────────

    def info(self) -> Dict:
        """Return engine diagnostics for the /health endpoint."""
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        return {
            "backend":    self.backend,
            "device":     self.device,
            "max_length": MAX_LENGTH,
            "is_ready":   self.is_ready,
            "gpu":        gpu_name,
        }
