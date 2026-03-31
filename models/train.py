"""
Fine-tune DistilBERT for 3-class sentiment analysis.
Target: POSITIVE / NEUTRAL / NEGATIVE
Hardware: RTX 4060 8GB, fp16, gradient accumulation
"""

import os, json, logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn
from datasets import load_dataset, load_from_disk, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback,
    DataCollatorWithPadding,
)
from sklearn.metrics import classification_report, f1_score
from loguru import logger

# ── Configuration ─────────────────────────────────────
@dataclass
class TrainConfig:
    model_name: str = "distilbert-base-uncased"
    dataset_path: str = "data/raw/sst2"
    output_dir: str = "models/checkpoints"
    num_labels: int = 3         # pos / neutral / neg
    max_length: int = 128
    batch_size: int = 32
    num_epochs: int = 4
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    fp16: bool = True            # halve VRAM on RTX 4060
    gradient_accumulation: int = 2
    seed: int = 42


def load_and_prepare_data(cfg: TrainConfig, tokenizer) -> DatasetDict:
    """Load SST-2, relabel to 3 classes, tokenize."""
    try:
        ds = load_from_disk(cfg.dataset_path)
    except:
        logger.info("Local dataset not found — downloading SST-2...")
        ds = load_dataset("sst2")

    # SST-2 is binary (0=neg, 1=pos). Map sentence score to 3-class:
    # This is a simplification — in production, use a 3-class dataset directly.
    def remap_labels(example):
        example["label"] = example["label"] * 2   # 0→0 (neg), 1→2 (pos)
        return example

    ds = ds.map(remap_labels)

    def tokenize(batch):
        return tokenizer(
            batch["sentence"],
            truncation=True,
            max_length=cfg.max_length,
            padding=False,
        )

    ds = ds.map(tokenize, batched=True, remove_columns=["sentence", "idx"])
    ds.set_format("torch")
    return ds


def compute_metrics(eval_pred) -> Dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    f1_macro = f1_score(labels, preds, average="macro")
    f1_weighted = f1_score(labels, preds, average="weighted")
    accuracy = (preds == labels).mean()
    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }


def train(cfg: TrainConfig = None):
    if cfg is None:
        cfg = TrainConfig()

    torch.manual_seed(cfg.seed)
    logger.info(f"Training {cfg.model_name} | fp16={cfg.fp16} | device={torch.cuda.get_device_name(0)}")

    # ── Tokenizer & Model ─────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=cfg.num_labels,
        id2label={0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"},
        label2id={"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2},
    )

    # ── Freeze all except classifier for first 2 epochs ───
    for param in model.distilbert.parameters():
        param.requires_grad = False

    # ── Dataset ───────────────────────────────────────────
    ds = load_and_prepare_data(cfg, tokenizer)
    collator = DataCollatorWithPadding(tokenizer)

    # ── Training Arguments ────────────────────────────────
    args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=64,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        fp16=cfg.fp16,
        gradient_accumulation_steps=cfg.gradient_accumulation,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        report_to="none",
        logging_steps=50,
        seed=cfg.seed,
        dataloader_num_workers=4,
        torch_compile=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # ── Phase 1: frozen backbone (2 epochs) ───────────────
    logger.info("Phase 1: Training classifier head only...")
    trainer.train()

    # ── Phase 2: unfreeze top 4 transformer layers ────────
    logger.info("Phase 2: Unfreezing top transformer layers...")
    for i, layer in enumerate(model.distilbert.transformer.layer):
        if i >= 4:
            for param in layer.parameters():
                param.requires_grad = True

    args.num_train_epochs = 2
    args.learning_rate = 5e-6
    trainer.train()

    # ── Save final model ──────────────────────────────────
    model.save_pretrained("models/checkpoints/final")
    tokenizer.save_pretrained("models/checkpoints/final")
    logger.info("✔ Model saved to models/checkpoints/final")

    return trainer


if __name__ == "__main__":
    train()