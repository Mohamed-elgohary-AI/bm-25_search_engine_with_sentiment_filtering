import time
from fastapi import FastAPI, Query

from src.pipelines.search import SearchPipeline

app = FastAPI(title="BM25 Food Review Search", version="2.0")

pipeline = SearchPipeline()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "bm25+metadata+cached-sentiment"
    }


@app.get("/search")
def search(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(10, ge=1, le=50),
    sentiment: str = Query(None, description="positive | negative | neutral"),
):
    start = time.perf_counter()

    # ✅ Use optimized pipeline (BM25 + metadata only)
    results = pipeline.search(
        q,
        top_k=top_k,
        sentiment_filter=sentiment
    )

    latency_ms = (time.perf_counter() - start) * 1000

    return {
        "query": q,
        "sentiment_filter": sentiment,
        "latency_ms": round(latency_ms, 2),
        "num_results": len(results),
        "results": results,
    }