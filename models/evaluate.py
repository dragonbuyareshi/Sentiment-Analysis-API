"""
models/evaluate.py
──────────────────
Evaluate the fine-tuned model on the SST-2 validation split.
Produces: accuracy, macro F1, per-class classification report,
          confusion matrix PNG, and a JSON summary.

Run:
  python models/evaluate.py
  python models/evaluate.py --model models/checkpoints/final --dataset data/raw/sst2
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless — no display required
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from datasets import load_from_disk, load_dataset
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


LABELS = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
LABEL2ID = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}

plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "text.color":       "#c9d1d9",
    "axes.labelcolor":  "#c9d1d9",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "axes.edgecolor":   "#30363d",
})


def evaluate(model_path: str, dataset_path: str) -> dict:
    output_dir = Path(model_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────
    logger.info(f"Loading model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = AutoModelForSequenceClassification.from_pretrained(model_path)
    device    = 0 if torch.cuda.is_available() else -1

    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        top_k=None,
        truncation=True,
        max_length=128,
    )

    # ── Load validation data ──────────────────────────────────────
    logger.info(f"Loading dataset from {dataset_path} ...")
    try:
        ds = load_from_disk(dataset_path)["validation"]
    except Exception:
        logger.info("Falling back to HuggingFace SST-2 download ...")
        ds = load_dataset("sst2")["validation"]

    texts  = list(ds["sentence"])
    labels = [lbl * 2 for lbl in ds["label"]]   # binary → 3-class mapping

    # ── Batch inference ───────────────────────────────────────────
    logger.info(f"Running inference on {len(texts):,} samples ...")
    t0      = time.perf_counter()
    preds   = []
    batch_size = 64

    for i in tqdm(range(0, len(texts), batch_size), desc="Evaluating"):
        batch   = texts[i : i + batch_size]
        results = pipe(batch)
        for r in results:
            best = max(r, key=lambda x: x["score"])["label"]
            preds.append(LABEL2ID.get(best, 0))

    elapsed_s = time.perf_counter() - t0

    # ── Metrics ───────────────────────────────────────────────────
    acc      = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro",    zero_division=0)
    f1_w     = f1_score(labels, preds, average="weighted", zero_division=0)
    report   = classification_report(labels, preds, target_names=LABELS, zero_division=0)

    logger.info(f"\n{'='*55}")
    logger.info(f"  Accuracy   : {acc:.4f}")
    logger.info(f"  F1 Macro   : {f1_macro:.4f}")
    logger.info(f"  F1 Weighted: {f1_w:.4f}")
    logger.info(f"  Throughput : {len(texts)/elapsed_s:.0f} samples/s")
    logger.info(f"{'='*55}")
    print("\n" + report)

    # ── Confusion matrix ──────────────────────────────────────────
    cm  = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=LABELS, yticklabels=LABELS,
        linewidths=0.5, linecolor="#30363d",
    )
    ax.set_title(f"Confusion Matrix  |  Acc={acc:.3f}  F1={f1_macro:.3f}", pad=12)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()

    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"✔ Confusion matrix saved → {cm_path}")

    # ── JSON report ───────────────────────────────────────────────
    summary = {
        "model_path":   model_path,
        "dataset_path": dataset_path,
        "n_samples":    len(labels),
        "accuracy":     round(acc,      4),
        "f1_macro":     round(f1_macro, 4),
        "f1_weighted":  round(f1_w,     4),
        "throughput_sps": round(len(labels) / elapsed_s, 1),
    }
    json_path = output_dir / "eval_report.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"✔ Eval report saved → {json_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned sentiment model")
    parser.add_argument("--model",   default="models/checkpoints/final")
    parser.add_argument("--dataset", default="data/raw/sst2")
    args = parser.parse_args()
    evaluate(args.model, args.dataset)
