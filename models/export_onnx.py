"""
Export fine-tuned model to ONNX format.
ONNX Runtime provides 2-3x faster inference vs PyTorch.
Run: python models/export_onnx.py
"""

import os, time
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import onnx
import onnxruntime as ort
from onnxsim import simplify
from loguru import logger


MODEL_PATH = "models/checkpoints/final"
ONNX_PATH  = "models/checkpoints/model.onnx"
ONNX_SIM   = "models/checkpoints/model_sim.onnx"


def export_to_onnx():
    Path("models/checkpoints").mkdir(parents=True, exist_ok=True)
    logger.info("Loading model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()

    # Dummy input for tracing
    dummy = tokenizer(
        "This product is absolutely amazing!",
        return_tensors="pt", padding="max_length",
        max_length=128, truncation=True,
    )

    logger.info("Exporting to ONNX...")
    torch.onnx.export(
        model,
        tuple(dummy.values()),
        ONNX_PATH,
        opset_version=17,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        },
    )
    logger.info(f"✔ Exported → {ONNX_PATH}")

    # Simplify graph
    model_onnx = onnx.load(ONNX_PATH)
    simplified, ok = simplify(model_onnx)
    if ok:
        onnx.save(simplified, ONNX_SIM)
        logger.info(f"✔ Simplified → {ONNX_SIM}")
    else:
        logger.warning("Simplification failed — using original ONNX")
        ONNX_SIM = ONNX_PATH

    # ── Benchmark ONNX vs PyTorch ─────────────────────────
    logger.info("Benchmarking...")
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_session = ort.InferenceSession(
        ONNX_SIM,
        sess_options,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    inp = {
        "input_ids": dummy["input_ids"].numpy(),
        "attention_mask": dummy["attention_mask"].numpy(),
    }

    RUNS = 200
    # PyTorch warmup + benchmark
    with torch.no_grad():
        for _ in range(5): model(**dummy)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(RUNS): model(**dummy)
    pt_ms = (time.perf_counter() - t0) / RUNS * 1000

    # ONNX warmup + benchmark
    for _ in range(5): ort_session.run(None, inp)
    t0 = time.perf_counter()
    for _ in range(RUNS): ort_session.run(None, inp)
    onnx_ms = (time.perf_counter() - t0) / RUNS * 1000

    logger.info(f"PyTorch : {pt_ms:.2f}ms  |  ONNX : {onnx_ms:.2f}ms  |  Speedup: {pt_ms/onnx_ms:.2f}x")


if __name__ == "__main__":
    export_to_onnx()