from typing import List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class SentimentRequest(BaseModel):
    # Accept either a single string or a list of strings
    texts: Union[str, List[str]]

    @field_validator("texts", mode="before")
    @classmethod
    def coerce_to_list(cls, value):
        # If the input is a single string, wrap it in a list
        if isinstance(value, str):
            return [value]
        return value


class SentimentResponse(BaseModel):
    label: str
    confidence: float


class SearchResult(BaseModel):
    bm25_score: float
    text: str
    product_id: str
    rating: int
    bert_sentiment: str
    bert_confidence: float


class SearchResponse(BaseModel):
    query: str
    sentiment_filter: Optional[str] = (
        None  # Using Optional is cleaner than Union[str, None]
    )
    latency_ms: float
    num_results: int
    results: List[SearchResult]
