import re

import hydra
import nltk
import pandas as pd
from loguru import logger
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from omegaconf import DictConfig
from tqdm import tqdm

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))
stemmer = PorterStemmer()


def clean_text(text: str) -> str:
    """Basic text cleaning: lowercase, remove punctuation, stopwords, and apply stemming."""
    text = re.sub(r"<.*?>", "", text)  # strip HTML tags
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # keep only letters
    text = text.lower()
    tokens = text.split()
    tokens = [stemmer.stem(t) for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)


def load_and_clean(sample=None):
    """Load raw reviews, clean text, and save processed data."""
    df = pd.read_csv(RAW_DATA_DIR / "Reviews.csv", nrows=sample)
    logger.info(f"Loaded {len(df)} reviews from raw data.")

    df = df.drop_duplicates(subset=["UserId", "ProfileName", "Time", "Text"])

    df = df.dropna(subset=["Text", "Summary", "Score"])

    df["sentiment"] = df["Score"].map(
        lambda s: "positive" if s >= 4 else ("negative" if s <= 2 else "neutral")
    )

    # Combine Summary + Text as the searchable document
    df["document"] = df["Summary"].fillna("") + " " + df["Text"]
    df = df[:5000]  # for testing purposes, remove later
    print(f"Cleaning {len(df):,} documents...")
    tqdm.pandas(desc="Cleaning text")
    df["document_clean"] = df["document"].progress_apply(clean_text)

    if sample:
        df = df.sample(sample, random_state=42)

    df.to_parquet(PROCESSED_DATA_DIR / "reviews.parquet", index=False)
    print(f"Saved {len(df)} clean reviews.")
    return df


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    logger.info("Loading and cleaning data...")
    load_and_clean()
    logger.success("Data loaded, cleaned, and saved successfully.")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
