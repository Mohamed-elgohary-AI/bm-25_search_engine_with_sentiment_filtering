import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import MODELS_DIR

LABELS = ["negative", "positive"]


class SentimentPipeline:
    """
    A pipeline for sentiment analysis using a pre-trained model.
    """

    def __init__(self):
        # Load the sentiment analysis model here
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(MODELS_DIR / "bert-sentiment")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODELS_DIR / "bert-sentiment"
        ).to(self.device)
        self.model.eval()

    def infer_sentiment(self, texts: list[str]) -> list[dict]:
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
                "label": LABELS[pred.item()],
                "confidence": round(probs[i][pred].item(), 4),
            }
            for i, pred in enumerate(preds)
        ]
