# BM25 Food Review Search Engine with Sentiment Filtering

[![DagsHub](https://img.shields.io/badge/DagsHub-Experiments-orange)](https://dagshub.com/mohamedabdelmonemelgohary/bm-25_search_engine_with_sentiment_filtering)
[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://www.python.org/)
[![uv](https://img.shields.io/badge/uv-package%20manager-purple)](https://github.com/astral-sh/uv)

A production-ready semantic search engine built on **BM25 retrieval** and **DistilBERT sentiment filtering**, trained on 500,000+ Amazon Fine Food reviews. Features a live search UI, AI-powered review summarization using a local LLM (Llama 3.2 via Ollama), and a full MLOps pipeline with DVC, MLflow, and DagsHub.

---

## Architecture

```
User Query
    ↓
BM25 Index (rank-bm25)       ← fast keyword retrieval
    ↓ top 50 candidates
DistilBERT Sentiment Filter  ← binary classification (positive/negative)
    ↓ filtered results
Llama 3.2 (Ollama)           ← local LLM summarization
    ↓
Streamlit UI
```

---

## Tech Stack

| Category            | Tools                    |
| ------------------- | ------------------------ |
| Package Manager     | uv                       |
| Data Versioning     | DVC + DagsHub            |
| Experiment Tracking | MLflow + DagsHub         |
| Search              | BM25 (rank-bm25)         |
| Sentiment Model     | DistilBERT (HuggingFace) |
| Summarization       | Llama 3.2 (Ollama)       |
| UI                  | Streamlit                |
| CI/CD               | GitHub Actions           |
| Serving             | FastAPI                  |

---

## Datasets

- **Amazon Fine Food Reviews** — 568,454 reviews from Kaggle, used for BM25 indexing and DistilBERT fine-tuning

---

## Models

### BM25 Index

- Built on 500,000+ preprocessed food reviews
- Hyperparameters: `k1=1.5`, `b=0.75`
- Artifacts versioned with DVC and stored on DagsHub

### DistilBERT Sentiment Classifier

- Base model: `distilbert-base-uncased`
- Fine-tuned on Amazon Fine Food Reviews
- Binary classification: `positive` / `negative`
- Training strategy: 5-fold cross validation, weighted loss for class imbalance, undersampling
- Best fold selected automatically and saved

### Llama 3.2 (Local)

- Runs entirely on your machine via Ollama
- No API key required
- Summarizes top search results in real time with streaming output

---

## Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/Mohamed-elgohary-AI/bm-25_search_engine_with_sentiment_filtering.git
cd bm-25_search_engine_with_sentiment_filtering
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Pull data and models from DagsHub

```bash
dvc remote modify dagshub --local auth basic
dvc remote modify dagshub --local user <your_dagshub_username>
dvc remote modify dagshub --local password <your_dagshub_token>
uv run dvc pull
```

### 4. Set up environment variables

```bash
# .env
DAGSHUB_USER_TOKEN=your_token_here
```

### 5. Install and start Ollama

Download from **[ollama.com](https://ollama.com)**, then:

```bash
ollama pull llama3.2
ollama serve
```

### 6. Run the Streamlit app

```bash
uv run streamlit run app.py --server.port 8080 --server.address 0.0.0.0
```

---

## MLOps Pipeline

```
data/raw/Reviews.csv
    ↓  dvc stage: download_datasets
data/raw/
    ↓  dvc stage: preprocess
data/processed/reviews.parquet
    ↓  dvc stage: build_index
models/bm25_index.pkl
    ↓  dvc stage: train_bert
models/bert-sentiment/
    ↓  dvc stage: evaluate_bert
reports/bert/metrics_bert.json
reports/figures/
```

Reproduce the full pipeline:

```bash
uv run dvc repro
```

---

## Experiments

All training runs, metrics, and artifacts are tracked on DagsHub:


👉 **[View Experiments on DagsHub](https://dagshub.com/mohamedabdelmonemelgohary/bm-25_search_engine_with_sentiment_filtering)**

---

## Project Structure

```
├── app.py                  # Streamlit UI
├── config/                 # Hydra configs
│   ├── config.yaml
│   ├── data/
│   └── model/
├── data/
│   ├── raw/                # tracked by DVC
│   └── processed/          # tracked by DVC
├── models/                 # tracked by DVC
│   └── bert-sentiment/
├── notebooks/              # EDA and pipeline exploration
├── reports/                # metrics and figures
├── src/
│   ├── api/                # FastAPI + state management
│   ├── ingestion/          # data loading
│   ├── modeling/           # train + evaluate
│   ├── pipelines/          # search + sentiment + summarizer
│   ├── app.py              # Streamlit app
│   └── preprocessor.py
├── dvc.yaml                # pipeline stages
├── pyproject.toml          # uv dependencies
└── .github/workflows/      # CI/CD
```

---

## 👤 Author

**Mohamed Elgohary**

- GitHub: [@Mohamed-elgohary-AI](https://github.com/Mohamed-elgohary-AI)
- DagsHub: [@mohamedabdelmonemelgohary](https://dagshub.com/mohamedabdelmonemelgohary)
