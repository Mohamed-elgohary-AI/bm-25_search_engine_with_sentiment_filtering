import time

import streamlit as st

# Import your global state file and pipelines
import src.api.state as state
from src.pipelines.search import SearchPipeline
from src.pipelines.sentiment_inference import SentimentPipeline
from src.pipelines.summarizer import ReviewSummarizer

st.set_page_config(
    page_title="Food Review Search",
    page_icon="🍕",
    layout="wide",
)


# ── Load pipelines once & map to global state ────────────────
@st.cache_resource
def load_pipelines():
    """
    Acts as your application lifespan loader. Runs once, caches
    the instances globally, and binds them to your state module.
    """
    # 1. Instantiate both models
    sentiment_instance = SentimentPipeline()
    search_instance = SearchPipeline()
    summarizer_instance = ReviewSummarizer()

    return sentiment_instance, search_instance, summarizer_instance


# Bootstrap the instances
sentiment_pipe, search_pipe, summarizer_pipe = load_pipelines()

# Bind them back to your state.py module references so internal code doesn't break
state.sentiment_pipeline = sentiment_pipe
state.pipeline = search_pipe
state.summarizer = summarizer_pipe


# ── Session state init ─────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = []

if "query" not in st.session_state:
    st.session_state.query = ""

if "sentiment_filter" not in st.session_state:
    st.session_state.sentiment_filter = "All"

if "summary" not in st.session_state:
    st.session_state.summary = None

if "summarize" not in st.session_state:
    st.session_state.summarize = False


# ── Search function (triggered on every change) ────────────
def run_search():
    query = st.session_state.query
    sentiment_filter = st.session_state.sentiment_filter

    if query.strip() == "":
        st.session_state.results = []
        st.session_state.summary = None
        return

    filter_val = None if sentiment_filter == "All" else sentiment_filter.lower()

    start = time.perf_counter()
    results = state.pipeline.search(query, sentiment_filter=filter_val, top_k=10)
    end = time.perf_counter()

    st.caption(f"Search took {(end - start) * 1000:.2f} ms")
    st.session_state.results = results

    st.session_state.results = results
    st.session_state.search_time = (end - start) * 1000
    st.session_state.summary = None


# ── Styling ────────────────────────────────────────────────
st.markdown(
    """
<style>
    .main { max-width: 800px; margin: auto; }

    .search-title {
        text-align: center;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        color: #4A90D9;
    }

    .result-card {
        background: #f9f9f9;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #4A90D9;
        color: #000000;
    }

    .result-card p {
        color: #000000;
    }

    .sentiment-positive { color: #2e7d32; font-weight: bold; }
    .sentiment-negative { color: #c62828; font-weight: bold; }
    .bm25-score { color: #888; font-size: 0.85rem; }
</style>
""",
    unsafe_allow_html=True,
)


# ── Header ─────────────────────────────────────────────────
st.markdown(
    '<div class="search-title">Food Review Search</div>', unsafe_allow_html=True
)

# ── Search UI ─────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 4, 1])

with col2:
    st.text_input(
        "search",
        key="query",
        placeholder="Search food reviews...",
        label_visibility="collapsed",
        on_change=run_search,
    )

    st.radio(
        "Sentiment filter",
        options=["All", "Positive", "Negative"],
        horizontal=True,
        key="sentiment_filter",
        on_change=run_search,
    )

    st.toggle(
        "Summarize results with Llama",
        key="summarize",
        on_change=run_search,
    )


# ── Results ────────────────────────────────────────────────
with col2:
    results = st.session_state.results

    if st.session_state.query.strip():
        st.markdown(f"**{len(results)} results** for *{st.session_state.query}*")

        if st.session_state.summarize and results:
            st.markdown("### AI Summary")
            with st.container():
                st.write_stream(
                    state.summarizer.stream(st.session_state.query, results)
                )
            st.divider()

        if not results:
            st.info("No results found. Try a different query or filter.")

        for r in results:
            sentiment_class = f"sentiment-{r['bert_sentiment']}"
            sentiment_emoji = "✅" if r["bert_sentiment"] == "positive" else "❌"

            st.markdown(
                f"""
            <div class="result-card">
                <h5>Product ID:  {r['product_id']}</h5>
                <p>{r['text'][:900]}...</p>
                <p>
                    <span class="{sentiment_class}">
                        {sentiment_emoji} {r['bert_sentiment'].capitalize()}
                    </span>
                    &nbsp;·&nbsp;
                    <span>⭐ {r['rating']}/5</span>
                    &nbsp;·&nbsp;
                    <span>Confidence: {r['bert_confidence']}</span>
                </p>
            </div>
            """,
                unsafe_allow_html=True,
            )

    else:
        st.markdown(
            """
        <div style="text-align:center; color:#888; margin-top:2rem;">
            Search across 500,000+ food reviews
        </div>
        """,
            unsafe_allow_html=True,
        )
