import json
import pickle

import hydra
import mlflow
import pandas as pd
from loguru import logger
from omegaconf import DictConfig
from rank_bm25 import BM25Okapi

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR


def build_index(cfg: DictConfig):
    """Build and save BM25 search index from processed reviews."""
    df = pd.read_parquet(PROCESSED_DATA_DIR / "reviews.parquet")

    corpus_raw = df["document"].tolist()
    corpus_clean = df["document_clean"].tolist()
    tokenized_corpus = [doc.split() for doc in corpus_clean]

    with mlflow.start_run(run_name="build_bm25_index"):
        mlflow.log_params(
            {
                "k1": cfg.model.k1,
                "b": cfg.model.b,
                "corpus_size": len(df),
                "dataset": "amazon-fine-food-reviews",
            }
        )

        bm25 = BM25Okapi(tokenized_corpus, k1=cfg.model.k1, b=cfg.model.b)

        # Save index + metadata together
        artifact = {
            "bm25": bm25,
            "corpus_raw": corpus_raw,
            "metadata": df[["ProductId", "UserId", "Score", "sentiment"]].to_dict(
                "records"
            ),
        }

        with open(MODELS_DIR / "bm25_index.pkl", "wb") as f:
            pickle.dump(artifact, f)

        # mlflow.log_artifact expects a string path
        mlflow.log_artifact(str(MODELS_DIR / "bm25_index.pkl"))

        # Log corpus stats
        avg_doc_len = sum(len(d.split()) for d in corpus_clean) / len(corpus_clean)
        mlflow.log_metric("avg_doc_length_tokens", avg_doc_len)

        metrics = {"corpus_size": len(df), "avg_doc_length": avg_doc_len}
        with open(REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f)

        print(
            f"Index built for {len(df)} docs. Avg doc length: {avg_doc_len:.1f} tokens"
        )


@hydra.main(config_path="../../config", config_name="config")
def main(cfg: DictConfig):
    """Build and save BM25 search index from processed reviews."""
    logger.info("Building BM25 index...")
    build_index(cfg)
    logger.success("BM25 index built and saved successfully.")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
