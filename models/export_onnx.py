"""
models/export_onnx.py
─────────────────────
Export fine-tuned DistilBERT to ONNX format.

Steps:
  1. Load PyTorch model
  2. Export to ONNX (opset 17, dynamic axes)
  3. Simplify graph with onnxsim
  4. Verify output matches PyTorch
  5. Benchmark: PyTorch vs ONNX Runtime

Run:
  python models/export_onnx.py
  python models/export_onnx.py --model models/checkpoints/final --runs 500
"""

import argparse
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from loguru import logger
from onnxsim import simplify
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def export(
    model_path: str = "models/checkpoints/final",
    output_dir: str = "models/checkpoints",
    n_benchmark_runs: int = 200,
) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    onnx_raw = Path(output_dir) / "model.onnx"
    onnx_sim = Path(output_dir) / "model_sim.onnx"

    # ── Load model ────────────────────────────────────────────────
    logger.info(f"Loading model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    # ── Dummy input ───────────────────────────────────────────────
    sample = "This product is absolutely amazing and I love it!"
    dummy  = tokenizer(
        sample,
        return_tensors="pt",
        padding="max_length",
        max_length=128,
        truncation=True,
    )

    # ── Export ────────────────────────────────────────────────────
    logger.info(f"Exporting to ONNX → {onnx_raw} ...")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy["input_ids"], dummy["attention_mask"]),
            str(onnx_raw),
            opset_version=17,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids":      {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits":         {0: "batch_size"},
            },
            do_constant_folding=True,
        )
    logger.info(f"✔ Exported — {onnx_raw.stat().st_size / 1e6:.1f} MB")

    # ── Simplify ──────────────────────────────────────────────────
    logger.info("Simplifying ONNX graph ...")
    model_onnx = onnx.load(str(onnx_raw))
    onnx.checker.check_model(model_onnx)

    simplified, ok = simplify(model_onnx)
    if ok:
        onnx.save(simplified, str(onnx_sim))
        size_orig = onnx_raw.stat().st_size / 1e6
        size_sim  = onnx_sim.stat().st_size  / 1e6
        logger.info(f"✔ Simplified → {onnx_sim}  ({size_orig:.1f}MB → {size_sim:.1f}MB)")
    else:
        import shutil
        shutil.copy(str(onnx_raw), str(onnx_sim))
        logger.warning("Simplification failed — using original ONNX")

    # ── Verify outputs match ──────────────────────────────────────
    logger.info("Verifying ONNX output matches PyTorch ...")
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if torch.cuda.is_available()
        else ["CPUExecutionProvider"]
    )
    session = ort.InferenceSession(str(onnx_sim), sess_opts, providers=providers)

    ort_inputs = {
        "input_ids":      dummy["input_ids"].numpy(),
        "attention_mask": dummy["attention_mask"].numpy(),
    }

    with torch.no_grad():
        pt_logits  = model(**dummy).logits.numpy()
    ort_logits = session.run(None, ort_inputs)[0]

    max_diff = float(np.abs(pt_logits - ort_logits).max())
    logger.info(f"  Max logit difference: {max_diff:.6f} {'✔ OK' if max_diff < 1e-4 else '⚠ HIGH'}")

    # ── Benchmark ─────────────────────────────────────────────────
    logger.info(f"Benchmarking ({n_benchmark_runs} runs each) ...")

    # PyTorch warmup
    with torch.no_grad():
        for _ in range(10):
            model(**dummy)

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_benchmark_runs):
            model(**dummy)
    pt_ms = (time.perf_counter() - t0) / n_benchmark_runs * 1000

    # ONNX warmup
    for _ in range(10):
        session.run(None, ort_inputs)

    t0 = time.perf_counter()
    for _ in range(n_benchmark_runs):
        session.run(None, ort_inputs)
    ort_ms = (time.perf_counter() - t0) / n_benchmark_runs * 1000

    speedup = pt_ms / ort_ms
    logger.info(f"\n  PyTorch  : {pt_ms:.2f} ms/sample")
    logger.info(f"  ONNX RT  : {ort_ms:.2f} ms/sample")
    logger.info(f"  Speedup  : {speedup:.2f}×")
    logger.info(f"\n✔ ONNX model ready at: {onnx_sim}")
    logger.info("  Set USE_ONNX=true in your environment to use it in the API.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--model",  default="models/checkpoints/final")
    parser.add_argument("--output", default="models/checkpoints")
    parser.add_argument("--runs",   type=int, default=200)
    args = parser.parse_args()
    export(args.model, args.output, args.runs)
