import os
from pathlib import Path

import hydra
import pandas as pd
import typer
from loguru import logger
from omegaconf import DictConfig
from rank_bm25 import BM25Okapi  # type: ignore
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer  # type: ignore

from src.calculate import get_sentiment, helpfulness_ratio
from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.preprocessor import preprocess

app = typer.Typer()


def init_bm25(df):
    global bm25_summary, bm25_text
    bm25_summary = BM25Okapi([preprocess(t) for t in df["Summary"]])
    bm25_text = BM25Okapi([preprocess(t) for t in df["Text"]])


def multi_field_search(query, w_summary=0.6, w_text=0.4):
    tokens = preprocess(query)

    scores_summary = bm25_summary.get_scores(tokens)
    scores_text = bm25_text.get_scores(tokens)

    # Weighted combination
    final_scores = (w_summary * scores_summary) + (w_text * scores_text)
    return final_scores


def full_search(query, df, sentiment_filter=None, min_stars=None, top_n=10):
    tokens = preprocess(query)
    bm25_scores = multi_field_search(query)

    results = []
    for idx, bm25_score in enumerate(bm25_scores):
        row = df.iloc[idx]

        # --- Structured filters (hard filters, applied early) ---
        if min_stars and row["Score"] < min_stars:
            continue

        # --- Sentiment (soft filter or booster) ---
        sentiment = get_sentiment(row["Text"])
        if sentiment_filter and sentiment["label"] != sentiment_filter:
            continue

        # --- Final score ---
        star_boost = (row["Score"] - 1) / 4
        help_boost = helpfulness_ratio(row)

        final_score = (
            0.5 * bm25_score
            + 0.2 * star_boost
            + 0.2 * help_boost
            + 0.1 * sentiment["score"]
        )

        results.append(
            {
                "summary": row["Summary"],
                "text": row["Text"][:200],
                "stars": row["Score"],
                "sentiment": sentiment["label"],
                "score": round(final_score, 4),
            }
        )

    return sorted(results, key=lambda x: x["score"], reverse=True)[:top_n]


@hydra.main(config_path="../config", config_name="config")
def main(
    cfg: DictConfig,
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
):
    logger.info("Training some model...")
    df = pd.read_parquet(
        os.path.join(PROCESSED_DATA_DIR, f"{cfg.data.file_name}-train.parquet")
    )
    init_bm25(df)

    logger.success("Modeling training complete.")
    # -----------------------------------------


if __name__ == "__main__":
    main()
