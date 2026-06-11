import time

from fastapi import FastAPI, Query

from src.modeling.engine import SearchEngine

app = FastAPI(title="BM25 Food Review Search", version="1.0")
engine = SearchEngine()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/search")
def search(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(10, ge=1, le=50),
    sentiment: str = Query(None, description="positive | negative | neutral"),
):
    start = time.time()
    results = engine.search(q, top_k=top_k, sentiment_filter=sentiment)
    latency_ms = (time.time() - start) * 1000

    return {
        "query": q,
        "sentiment_filter": sentiment,
        "latency_ms": round(latency_ms, 2),
        "results": results,
    }
