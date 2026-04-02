"""
models/train.py
───────────────
Fine-tune DistilBERT for 3-class sentiment analysis.
Target labels: NEGATIVE (0) · NEUTRAL (1) · POSITIVE (2)

Hardware: RTX 4060 8 GB — CUDA 12.8
Strategy:
  Phase 1 — freeze DistilBERT backbone, train classifier head only
  Phase 2 — unfreeze top 4 transformer layers, full fine-tune

Run:
  python models/train.py
  python models/train.py --model roberta-base --batch 16
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from datasets import load_dataset, load_from_disk, DatasetDict
from loguru import logger
from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


# ── Configuration ─────────────────────────────────────────────────
@dataclass
class TrainConfig:
    model_name:   str   = "distilbert-base-uncased"
    dataset_path: str   = "data/raw/sst2"
    output_dir:   str   = "models/checkpoints"
    num_labels:   int   = 3
    max_length:   int   = 128
    batch_size:   int   = 32
    num_epochs:   int   = 4
    lr:           float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    fp16:         bool  = True
    grad_accum:   int   = 2
    seed:         int   = 42


ID2LABEL = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
LABEL2ID = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}


# ── Data ──────────────────────────────────────────────────────────
def load_and_tokenize(cfg: TrainConfig, tokenizer) -> DatasetDict:
    """Load SST-2, remap binary labels to 3-class, tokenize."""
    try:
        ds = load_from_disk(cfg.dataset_path)
        logger.info(f"Loaded dataset from disk: {cfg.dataset_path}")
    except Exception:
        logger.info("Local dataset not found — downloading SST-2 from HuggingFace ...")
        ds = load_dataset("sst2")

    def remap_and_tokenize(batch):
        enc = tokenizer(
            batch["sentence"],
            truncation=True,
            max_length=cfg.max_length,
            padding=False,
        )
        # SST-2: 0=negative→0, 1=positive→2; neutral class (1) unused in SST-2
        enc["label"] = [lbl * 2 for lbl in batch["label"]]
        return enc

    tok_ds = ds.map(
        remap_and_tokenize,
        batched=True,
        remove_columns=["sentence", "idx"],
        desc="Tokenizing",
    )
    tok_ds.set_format("torch")
    return tok_ds


# ── Metrics ───────────────────────────────────────────────────────
def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy":    float((preds == labels).mean()),
        "f1_macro":    float(f1_score(labels, preds, average="macro",    zero_division=0)),
        "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
    }


# ── Training ──────────────────────────────────────────────────────
def train(cfg: TrainConfig = None) -> Trainer:
    if cfg is None:
        cfg = TrainConfig()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
    else:
        logger.warning("No GPU detected — training on CPU (this will be slow)")
        cfg.fp16 = False

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # ── Tokenizer & model ─────────────────────────────────────────
    logger.info(f"Loading {cfg.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=cfg.num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # ── Dataset ───────────────────────────────────────────────────
    tok_ds   = load_and_tokenize(cfg, tokenizer)
    collator = DataCollatorWithPadding(tokenizer, return_tensors="pt")

    # ── Phase 1: freeze backbone, train head ──────────────────────
    logger.info("Phase 1 — freezing backbone, training classifier head ...")
    for param in model.distilbert.parameters():
        param.requires_grad = False

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=min(2, cfg.num_epochs),
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=64,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        fp16=cfg.fp16,
        gradient_accumulation_steps=cfg.grad_accum,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to="none",
        logging_steps=50,
        save_total_limit=2,
        seed=cfg.seed,
        dataloader_num_workers=4,
        label_names=["labels"],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tok_ds["train"],
        eval_dataset=tok_ds["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    # ── Phase 2: unfreeze top layers ─────────────────────────────
    if cfg.num_epochs > 2:
        logger.info("Phase 2 — unfreezing top 4 transformer layers ...")
        for i, layer in enumerate(model.distilbert.transformer.layer):
            if i >= len(model.distilbert.transformer.layer) - 4:
                for param in layer.parameters():
                    param.requires_grad = True

        trainer.args.num_train_epochs = cfg.num_epochs - 2
        trainer.args.learning_rate    = 5e-6  # lower LR for fine-tuning
        trainer.train()

    # ── Save final model ──────────────────────────────────────────
    final_dir = os.path.join(cfg.output_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    # Save config snapshot
    config_snapshot = {
        "model_name":  cfg.model_name,
        "num_labels":  cfg.num_labels,
        "max_length":  cfg.max_length,
        "id2label":    ID2LABEL,
        "model_version": "v2.1.0",
    }
    with open(os.path.join(final_dir, "train_config.json"), "w") as f:
        json.dump(config_snapshot, f, indent=2)

    logger.info(f"✔ Model saved to {final_dir}")
    return trainer


# ── CLI ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for sentiment")
    parser.add_argument("--model",   default="distilbert-base-uncased")
    parser.add_argument("--dataset", default="data/raw/sst2")
    parser.add_argument("--output",  default="models/checkpoints")
    parser.add_argument("--epochs",  type=int,   default=4)
    parser.add_argument("--batch",   type=int,   default=32)
    parser.add_argument("--lr",      type=float, default=2e-5)
    parser.add_argument("--no-fp16", action="store_true")
    args = parser.parse_args()

    cfg = TrainConfig(
        model_name=args.model,
        dataset_path=args.dataset,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        fp16=not args.no_fp16,
    )
    train(cfg)
