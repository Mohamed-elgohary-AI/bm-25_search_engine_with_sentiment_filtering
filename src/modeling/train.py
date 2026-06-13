import os
from os import path
from pathlib import Path
from sys import path

import hydra
import mlflow
import pandas as pd
import torch
import typer
from loguru import logger
from omegaconf import DictConfig
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_scheduler,
)

from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.preprocessor import clean_text

app = typer.Typer()


class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def load_data(sample=50_000):
    df = pd.read_parquet(PROCESSED_DATA_DIR / "reviews.parquet")[
        ["Text", "Score"]
    ].dropna()
    df["label"] = df["Score"].map({1: 0, 2: 0, 3: 1, 4: 2, 5: 2})
    if sample:
        df = df.sample(sample, random_state=42)
    return df["Text"].tolist(), df["label"].tolist()


@hydra.main(config_path="../../config", config_name="config")
def train(cfg: DictConfig):
    """Train a BERT model for sentiment classification."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    texts, labels = load_data(sample=cfg.model.sample_size)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)

    split = int(len(texts) * cfg.model.train_split)

    train_dl = DataLoader(
        ReviewDataset(texts[:split], labels[:split], tokenizer, cfg.model.max_length),
        batch_size=cfg.model.batch_size,
        shuffle=True,
    )

    val_dl = DataLoader(
        ReviewDataset(texts[split:], labels[split:], tokenizer, cfg.model.max_length),
        batch_size=cfg.model.batch_size,
        shuffle=False,
    )

    logger.info(
        f"Initializing model {cfg.model.model_name} with {cfg.model.num_labels} labels."
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.model_name, num_labels=cfg.model.num_labels
    )
    optimizer = AdamW(model.parameters(), lr=cfg.model.learning_rate)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=cfg.model.epochs * len(train_dl),
    )

    with mlflow.start_run(run_name="bert-sentiment"):
        mlflow.log_params(
            {
                "model_name": cfg.model.model_name,
                "epochs": cfg.model.epochs,
                "batch_size": cfg.model.batch_size,
                "learning_rate": cfg.model.learning_rate,
                "max_length": cfg.model.max_length,
                "sample_size": cfg.model.sample_size,
            }
        )
        logger.info(
            f"Starting training for {cfg.model.epochs} epochs on {len(train_dl)} batches per epoch."
        )
        for epoch in range(cfg.model.epochs):
            # ── Training loop ──
            model.train()
            total_loss = 0
            for batch in tqdm(train_dl, desc=f"Training Epoch {epoch+1}"):
                batch = {k: v.to(device) for k, v in batch.items()}
                loss = model(**batch).loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_dl)

            # ── Validation loop ──

            model.eval()
            correct = total = 0
            with torch.no_grad():
                for batch in tqdm(val_dl, desc=f"Validation Epoch {epoch+1}"):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    preds = model(**batch).logits.argmax(dim=-1)
                    correct += (preds == batch["labels"]).sum().item()
                    total += batch["labels"].size(0)

            avg_val_accuracy = correct / total
            mlflow.log_metrics(
                {
                    f"train_loss_epoch_{epoch+1}": avg_train_loss,
                    f"val_accuracy_epoch_{epoch+1}": avg_val_accuracy,
                },
                step=epoch,
            )
            logger.info(
                f"Epoch {epoch+1}/{cfg.model.epochs} - "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Accuracy: {avg_val_accuracy:.4f}"
            )
        logger.success("Training completed. Saving model and tokenizer...")
        model.save_pretrained(MODELS_DIR / "bert-sentiment")
        tokenizer.save_pretrained(MODELS_DIR / "bert-sentiment")
        mlflow.log_artifacts(
            str(MODELS_DIR / "bert-sentiment"), artifact_path="bert-sentiment"
        )
        logger.success(f"Model saved to {MODELS_DIR / 'bert-sentiment'}")


if __name__ == "__main__":
    train()
