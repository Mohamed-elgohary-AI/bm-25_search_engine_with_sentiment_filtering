# src/modeling/evaluate.py

import json
from pathlib import Path

import dagshub
import hydra
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from loguru import logger
from omegaconf import DictConfig
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from src.modeling.train import ReviewDataset
from dotenv import load_dotenv
load_dotenv()
LABELS = ["negative", "positive"]
METRICS_DIR = REPORTS_DIR / "bert"
FIGURES_DIR = REPORTS_DIR / "figures"


def load_data(path: str, sample: int, tokenizer, max_len: int):
    df = pd.read_parquet(PROCESSED_DATA_DIR / "reviews_test.parquet")[
        ["Text", "sentiment"]
    ].dropna()

    df = df[df["sentiment"] != "neutral"]

    label_map = {"negative": 0, "positive": 1}
    df["label"] = df["sentiment"].map(label_map)

   # if sample:
     #   df = df.sample(sample, random_state=42)

    val_texts  = df["Text"].tolist()
    val_labels = df["label"].tolist()

    dataset = ReviewDataset(val_texts, val_labels, tokenizer, max_len)
    return DataLoader(dataset, batch_size=32)

def plot_confusion_matrix(cm: np.ndarray):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABELS,
        yticklabels=LABELS,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    path = FIGURES_DIR / "confusion_matrix.png"
    fig.savefig(path)
    plt.close()
    logger.info(f"Saved confusion matrix → {path}")
    return path


def plot_roc_curves(all_labels: list, all_probs: np.ndarray):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])  # prob of positive class
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    plt.tight_layout()
    path = FIGURES_DIR / "roc_curves.png"
    fig.savefig(path)
    plt.close()
    logger.info(f"Saved ROC curve → {path}")
    return path


def plot_per_class_metrics(report: dict):
    metrics = ["precision", "recall", "f1-score"]
    values = {m: [report[l][m] for l in LABELS] for m in metrics}
    x = np.arange(len(LABELS))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, values[metric], width, label=metric)
    ax.set_xticks(x + width)
    ax.set_xticklabels(LABELS)
    ax.set_ylim(0, 1.1)
    ax.set_title("Per-Class Metrics")
    ax.legend()
    plt.tight_layout()
    path = FIGURES_DIR / "per_class_metrics.png"
    fig.savefig(path)
    plt.close()
    logger.info(f"Saved per-class metrics → {path}")
    return path


def plot_confidence_distribution(all_probs: np.ndarray, all_preds: list):
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, label in enumerate(LABELS):
        mask = np.array(all_preds) == i
        if mask.sum() > 0:
            ax.hist(all_probs[mask, i], bins=30, alpha=0.6, label=label)
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution per Class")
    ax.legend()
    plt.tight_layout()
    path = FIGURES_DIR / "confidence_distribution.png"
    fig.savefig(path)
    plt.close()
    logger.info(f"Saved confidence distribution → {path}")
    return path


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def evaluate(cfg: DictConfig):
    # ── Setup ──────────────────────────────────────────────
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    dagshub.init(
        repo_owner="mohamedabdelmonemelgohary",
        repo_name="bm-25_search_engine_with_sentiment_filtering",
        mlflow=True,
    )
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODELS_DIR / "bert-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODELS_DIR / "bert-sentiment"
    ).to(device)
    model.eval()

    val_dl = load_data(
        path=PROCESSED_DATA_DIR / "reviews_test.parquet",
        sample=cfg.model.sample_size,
        tokenizer=tokenizer,
        max_len=cfg.model.max_length,
    )

    # ── Inference ──────────────────────────────────────────
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(val_dl, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_probs = np.array(all_probs)

    # ── Metrics ────────────────────────────────────────────
    report = classification_report(
        all_labels,
        all_preds,
        target_names=LABELS,
        output_dict=True,
    )
    cm = confusion_matrix(all_labels, all_preds)

    metrics = {
        "accuracy": report["accuracy"],
        "f1_macro": report["macro avg"]["f1-score"],
        "precision": report["macro avg"]["precision"],
        "recall": report["macro avg"]["recall"],
        "classification_report": report,
    }

    metrics_path = METRICS_DIR / "metrics_bert.json"
    

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.success(f"Metrics saved → {metrics_path}")

    # ── Plots ──────────────────────────────────────────────
    cm_path = plot_confusion_matrix(cm)
    roc_path = plot_roc_curves(all_labels, all_probs)
    bar_path = plot_per_class_metrics(report)
    conf_path = plot_confidence_distribution(all_probs, all_preds)

    # ── MLflow ─────────────────────────────────────────────
    with mlflow.start_run(run_name="bert-evaluate"):
        mlflow.log_metrics(
            {
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
            }
        )
        mlflow.log_artifact(str(metrics_path), artifact_path="reports/bert")
        mlflow.log_artifact(str(cm_path), artifact_path="reports/figures")
        mlflow.log_artifact(str(roc_path), artifact_path="reports/figures")
        mlflow.log_artifact(str(bar_path), artifact_path="reports/figures")
        mlflow.log_artifact(str(conf_path), artifact_path="reports/figures")

    logger.success("Evaluation complete.")


if __name__ == "__main__":
    evaluate()
