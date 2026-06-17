# src/pipelines/search.py
import time
import pickle
import re
import nltk
import torch
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.config import MODELS_DIR

nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))
stemmer   = PorterStemmer()
LABELS    = ["negative", "positive"]


class SearchPipeline:
    def __init__(self):
        # ── Load BM25 ──────────────────────────────────────
        with open(MODELS_DIR / "bm25_index.pkl", "rb") as f:
            artifact     = pickle.load(f)
        self.bm25        = artifact["bm25"]
        self.corpus      = artifact["corpus_raw"]
        self.metadata    = artifact["metadata"]

        # ── Load BERT ──────────────────────────────────────
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(MODELS_DIR / "bert-sentiment")
        self.model     = AutoModelForSequenceClassification.from_pretrained(
            MODELS_DIR / "bert-sentiment"
        ).to(self.device)
        self.model.eval()

    def _preprocess_query(self, query: str) -> list[str]:
        query  = re.sub(r"[^a-zA-Z\s]", "", query.lower())
        tokens = query.split()
        return [stemmer.stem(t) for t in tokens if t not in STOPWORDS]

    def _bm25_search(self, query: str, top_k: int = 50) -> list[dict]:
        tokens = self._preprocess_query(query)

        # STEP 1: fast retrieval (no full scoring)
        top_indices = self.bm25.get_top_n(
            tokens,
            range(len(self.corpus)),
            n=top_k
        )

        return [
            {
                "score": 0.0,  # or skip
                "text": self.corpus[i],
                "metadata": self.metadata[i],
            }
            for i in top_indices
        ]

    def _predict_sentiment(self, texts: list[str]) -> list[dict]:
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            probs = torch.softmax(self.model(**encodings).logits, dim=-1)

        preds = probs.argmax(dim=-1)
        return [
            {
                "label":      LABELS[pred.item()],
                "confidence": round(probs[i][pred].item(), 4),
            }
            for i, pred in enumerate(preds)
        ]

    def search(self, query: str, sentiment_filter: str = None, top_k: int = 10) -> list[dict]:
        if not query.strip():
            return []
            # 1. BM25 retrieval
        t_bm25 = time.perf_counter()
        candidates = self._bm25_search(query, top_k=50)
        t_bm25_end = time.perf_counter()
        t_sort = time.perf_counter()
        texts      = [c["text"] for c in candidates]
        t_sort_end = time.perf_counter()
        t_sent = time.perf_counter()
        sentiments = self._predict_sentiment(texts)
        t_sent_end = time.perf_counter()

        results = []
        for candidate, sentiment in zip(candidates, sentiments):
            if sentiment_filter is None or sentiment["label"] == sentiment_filter:
                results.append({
                    "bm25_score":      candidate["score"],
                    "text":            candidate["text"],
                    "product_id":      candidate["metadata"]["ProductId"],
                    "rating":          candidate["metadata"]["Score"],
                    "bert_sentiment":  sentiment["label"],
                    "bert_confidence": sentiment["confidence"],
                })

        print("BM25:", (t_bm25_end - t_bm25) * 1000)
        print("Sort:", (t_sort_end - t_sort) * 1000)
        print("Filter:", (t_sent_end - t_sent) * 1000)
        return results[:top_k]