import json
import copy
import os
from os import path
from pathlib import Path
from sys import path
from sklearn.utils import resample
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import dagshub
import numpy as np
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
from dotenv import load_dotenv

load_dotenv()
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


def load_data(sample=50_000, fold=0, n_splits=5):
    df = pd.read_parquet(PROCESSED_DATA_DIR / "reviews_train.parquet")[
        ["Text", "sentiment"]
    ].dropna()

    # ── Drop neutral ───────────────────────────────────────
    df = df[df["sentiment"] != "neutral"]

    label_map = {"negative": 0, "positive": 1}
    df["label"] = df["sentiment"].map(label_map)

    if sample:
        df = df.sample(sample, random_state=42)

    # ── Undersample majority class ─────────────────────────
    df_neg = df[df["label"] == 0]
    df_pos = df[df["label"] == 1]

    min_count = min(len(df_neg), len(df_pos))

    df_balanced = pd.concat([
        resample(df_neg, n_samples=min_count, random_state=42),
        resample(df_pos, n_samples=min_count, random_state=42),
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    # ── Stratified K-Fold ──────────────────────────────────
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = list(skf.split(df_balanced["Text"], df_balanced["label"]))
    train_idx, val_idx = splits[fold]

    train_df = df_balanced.iloc[train_idx]
    val_df   = df_balanced.iloc[val_idx]

    return (
        train_df["Text"].tolist(), train_df["label"].tolist(),
        val_df["Text"].tolist(),   val_df["label"].tolist(),
    )
@hydra.main(config_path="../../config", config_name="config")
def train(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)

    dagshub.init(
        repo_owner="mohamedabdelmonemelgohary",
        repo_name="bm-25_search_engine_with_sentiment_filtering",
        mlflow=True,
    )
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    # ── Initialize best fold tracking ─────────────────────
    best_val_accuracy = 0
    best_fold = None
    best_model_state = None

    for fold in range(cfg.model.n_splits):
        logger.info(f"Training fold {fold + 1}/{cfg.model.n_splits}")

        train_texts, train_labels, val_texts, val_labels = load_data(
            sample=cfg.model.sample_size,
            fold=fold,
            n_splits=cfg.model.n_splits,
        )

        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.array([0, 1]),
            y=train_labels,
        )
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float).to(device)
        )
        
        train_dl = DataLoader(
            ReviewDataset(train_texts, train_labels, tokenizer, cfg.model.max_length),
            batch_size=cfg.model.batch_size,
            shuffle=True,
        )
        val_dl = DataLoader(
            ReviewDataset(val_texts, val_labels, tokenizer, cfg.model.max_length),
            batch_size=cfg.model.batch_size,
            shuffle=False,
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model.model_name, num_labels=cfg.model.num_labels
        ).to(device)

        optimizer = AdamW(model.parameters(), lr=cfg.model.learning_rate)
        scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=cfg.model.epochs * len(train_dl),
        )

        with mlflow.start_run(run_name=f"bert-sentiment-fold-{fold + 1}"):
            mlflow.log_params({
                "model_name":    cfg.model.model_name,
                "epochs":        cfg.model.epochs,
                "batch_size":    cfg.model.batch_size,
                "learning_rate": cfg.model.learning_rate,
                "max_length":    cfg.model.max_length,
                "sample_size":   cfg.model.sample_size,
                "fold":          fold + 1,
                "n_splits":      cfg.model.n_splits,
            })

            for epoch in range(cfg.model.epochs):
                model.train()
                total_loss = 0
                for batch in tqdm(train_dl, desc=f"Fold {fold+1} Epoch {epoch+1} Train"):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    loss = loss_fn(outputs.logits, batch["labels"])
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    total_loss += loss.item()

                avg_train_loss = total_loss / len(train_dl)

                model.eval()
                correct = total = 0
                with torch.no_grad():
                    for batch in tqdm(val_dl, desc=f"Fold {fold+1} Epoch {epoch+1} Val"):
                        batch = {k: v.to(device) for k, v in batch.items()}
                        preds = model(**batch).logits.argmax(dim=-1)
                        correct += (preds == batch["labels"]).sum().item()
                        total += batch["labels"].size(0)

                avg_val_accuracy = correct / total
                mlflow.log_metrics({
                    f"train_loss_epoch_{epoch+1}":   avg_train_loss,
                    f"val_accuracy_epoch_{epoch+1}": avg_val_accuracy,
                }, step=epoch)

                logger.info(
                    f"Fold {fold+1} | Epoch {epoch+1}/{cfg.model.epochs} | "
                    f"Loss: {avg_train_loss:.4f} | Acc: {avg_val_accuracy:.4f}"
                )

            # ── Track best fold ────────────────────────────
            if avg_val_accuracy > best_val_accuracy:
                best_val_accuracy = avg_val_accuracy
                best_fold = fold + 1
                best_model_state = copy.deepcopy(model.state_dict())
                logger.success(f"New best fold: {fold+1} with val_accuracy: {avg_val_accuracy:.4f}")

    # ── Save best fold only — OUTSIDE the loop ────────────
    logger.success(f"Best fold: {best_fold} with val_accuracy: {best_val_accuracy:.4f}")
    model.load_state_dict(best_model_state)
    model.save_pretrained(MODELS_DIR / "bert-sentiment")
    tokenizer.save_pretrained(MODELS_DIR / "bert-sentiment")
    mlflow.log_artifacts(
        str(MODELS_DIR / "bert-sentiment"), artifact_path="bert-sentiment"
    )
    logger.success(f"Best model (fold {best_fold}) saved to {MODELS_DIR / 'bert-sentiment'}")



if __name__ == "__main__":
    train()
