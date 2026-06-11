import pickle
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from src.config import MODELS_DIR

nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))
stemmer = PorterStemmer()


class SearchEngine:
    def __init__(self, index_path=MODELS_DIR / "bm25_index.pkl"):
        with open(index_path, "rb") as f:
            artifact = pickle.load(f)
        self.bm25 = artifact["bm25"]
        self.corpus_raw = artifact["corpus_raw"]
        self.metadata = artifact["metadata"]

    def _preprocess_query(self, query: str) -> list[str]:
        query = re.sub(r"[^a-zA-Z\s]", "", query.lower())
        tokens = query.split()
        return [stemmer.stem(t) for t in tokens if t not in STOPWORDS]

    def search(self, query: str, top_k=10, sentiment_filter=None):
        tokens = self._preprocess_query(query)
        scores = self.bm25.get_scores(tokens)

        # Get top_k * 3 candidates first, then filter by sentiment
        n = top_k * 3 if sentiment_filter else top_k
        top_indices = scores.argsort()[-n:][::-1]

        results = []
        for idx in top_indices:
            meta = self.metadata[idx]
            if sentiment_filter and meta["sentiment"] != sentiment_filter:
                continue
            results.append(
                {
                    "score": float(scores[idx]),
                    "text": self.corpus_raw[idx][:300],  # truncate for display
                    "product_id": meta["ProductId"],
                    "rating": meta["Score"],
                    "sentiment": meta["sentiment"],
                }
            )
            if len(results) == top_k:
                break

        return results
