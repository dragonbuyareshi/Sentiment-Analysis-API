"""
Evaluate fine-tuned model: accuracy, macro F1, confusion matrix.
Run: python models/evaluate.py --model models/checkpoints/final
"""

import argparse, json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score,
)
from tqdm import tqdm
from loguru import logger


def evaluate_model(model_path: str, dataset_path: str = "data/raw/sst2"):
    logger.info(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = 0 if torch.cuda.is_available() else -1

    pipe = pipeline(
        "text-classification", model=model, tokenizer=tokenizer,
        device=device, top_k=None, truncation=True, max_length=128,
    )

    # Load validation split
    ds = load_from_disk(dataset_path)["validation"]
    texts = ds["sentence"]
    true_labels = ds["label"]   # mapped to 3-class

    # Batch inference
    logger.info(f"Running inference on {len(texts)} samples...")
    all_preds = []
    batch_size = 64
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        results = pipe(batch)
        for r in results:
            pred_label = max(r, key=lambda x: x["score"])["label"]
            all_preds.append(pred_label)

    label_map = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}
    pred_ids = [label_map[p] for p in all_preds]

    # ── Metrics ───────────────────────────────────────────
    acc = accuracy_score(true_labels, pred_ids)
    f1_macro = f1_score(true_labels, pred_ids, average="macro")
    report = classification_report(
        true_labels, pred_ids,
        target_names=["NEGATIVE", "NEUTRAL", "POSITIVE"],
    )

    logger.info(f"\nAccuracy : {acc:.4f}")
    logger.info(f"F1 Macro : {f1_macro:.4f}")
    print("\n" + report)

    # ── Save JSON report ──────────────────────────────────
    Path("models").mkdir(exist_ok=True)
    with open("models/eval_report.json", "w") as f:
        json.dump({"accuracy": acc, "f1_macro": f1_macro}, f, indent=2)

    # ── Confusion Matrix ──────────────────────────────────
    cm = confusion_matrix(true_labels, pred_ids)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["NEG", "NEU", "POS"],
        yticklabels=["NEG", "NEU", "POS"],
    )
    ax.set_title(f"Confusion Matrix (Acc={acc:.3f}, F1={f1_macro:.3f})")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig("models/confusion_matrix.png", dpi=150)
    logger.info("✔ Saved confusion_matrix.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/checkpoints/final")
    parser.add_argument("--dataset", default="data/raw/sst2")
    args = parser.parse_args()
    evaluate_model(args.model, args.dataset)