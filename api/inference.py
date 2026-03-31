"""
Singleton inference engine — supports PyTorch and ONNX Runtime.
Automatically uses GPU if available, graceful CPU fallback.
"""

import os, re, html
from typing import Dict, List, Optional
import numpy as np
import torch
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from loguru import logger
import emoji

MODEL_DIR  = os.getenv("MODEL_DIR",  "models/checkpoints/final")
ONNX_PATH  = os.getenv("ONNX_PATH",  "models/checkpoints/model_sim.onnx")
USE_ONNX   = os.getenv("USE_ONNX",   "true").lower() == "true"
MAX_LENGTH = int(os.getenv("MAX_LENGTH", 128))
LABELS     = ["NEGATIVE", "NEUTRAL", "POSITIVE"]


class SentimentEngine:
    """Thread-safe singleton model loader and predictor."""

    _instance: Optional["SentimentEngine"] = None

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.ort_session = None
        self.is_ready = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self):
        logger.info(f"Loading tokenizer from {MODEL_DIR}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

        if USE_ONNX and os.path.exists(ONNX_PATH):
            logger.info(f"Using ONNX Runtime — {ONNX_PATH}")
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = 4
            providers = ["CUDAExecutionProvider"] if self.device == "cuda" else ["CPUExecutionProvider"]
            self.ort_session = ort.InferenceSession(ONNX_PATH, opts, providers=providers)
        else:
            logger.info(f"Using PyTorch — device={self.device}")
            self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
            self.model.eval().to(self.device)
            if self.device == "cuda":
                torch.backends.cudnn.benchmark = True

        self.is_ready = True
        logger.info("✔ Inference engine ready")

    def _preprocess(self, text: str) -> str:
        """Clean text before tokenization."""
        text = html.unescape(text)                         # & → &
        text = emoji.demojize(text)                        # 😊 → :smiling_face:
        text = re.sub(r"http\S+|www\.\S+", " [URL] ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:MAX_LENGTH * 6]   # rough char limit before tokenize

    def _tokenize(self, texts: List[str]) -> Dict:
        return self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=MAX_LENGTH, return_tensors="np" if self.ort_session else "pt",
        )

    def _run_inference(self, encoded) -> np.ndarray:
        if self.ort_session:
            logits = self.ort_session.run(None, {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            })[0]
        else:
            enc = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                logits = self.model(**enc).logits.cpu().numpy()
        return logits

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        e = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    def predict(self, text: str) -> Dict:
        clean = self._preprocess(text)
        enc = self._tokenize([clean])
        logits = self._run_inference(enc)
        probs = self._softmax(logits)[0]
        idx = int(np.argmax(probs))
        return {
            "label": LABELS[idx],
            "confidence": round(float(probs[idx]), 4),
            "scores": {l: round(float(p), 4) for l, p in zip(LABELS, probs)},
            "token_count": enc["input_ids"].shape[1],
        }

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        cleans = [self._preprocess(t) for t in texts]
        enc = self._tokenize(cleans)
        logits = self._run_inference(enc)
        probs_all = self._softmax(logits)
        results = []
        for probs in probs_all:
            idx = int(np.argmax(probs))
            results.append({
                "label": LABELS[idx],
                "confidence": round(float(probs[idx]), 4),
                "scores": {l: round(float(p), 4) for l, p in zip(LABELS, probs)},
            })
        return results
