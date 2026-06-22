import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, Query

from src.api import state
from src.api.schemas import SearchResponse, SentimentRequest, SentimentResponse
from src.pipelines.search import SearchPipeline
from src.pipelines.sentiment_inference import SentimentPipeline

shared_search_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading AI models into memory...")
    state.sentiment_pipeline = SentimentPipeline()
    state.pipeline = SearchPipeline()

    yield
    print("Shutting down...")


app = FastAPI(title="BM25 Food Review Search", version="2.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model": "bm25+metadata+cached-sentiment"}


@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(10, ge=1, le=50, description="Number of results to return"),
    sentiment: Optional[str] = Query(
        None, description="Filter by sentiment: positive | negative"
    ),
):
    start = time.perf_counter()

    # ✅ Use optimized pipeline (BM25 + metadata only)
    results = state.pipeline.search(q, top_k=top_k, sentiment_filter=sentiment)

    latency_ms = (time.perf_counter() - start) * 1000

    return {
        "query": q,
        "sentiment_filter": sentiment,
        "latency_ms": round(latency_ms, 2),
        "num_results": len(results),
        "results": results,
    }


@app.post("/sentiment")
def get_sentiment(payload: SentimentRequest) -> List[SentimentResponse]:
    """
    Get the sentiment of a given text using the cached sentiment model.
    """
    results = state.sentiment_pipeline.infer_sentiment(payload.texts)
    return [
        SentimentResponse(label=result["label"], confidence=result["confidence"])
        for result in results
    ]
