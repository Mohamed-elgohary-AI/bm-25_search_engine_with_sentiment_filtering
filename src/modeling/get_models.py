from rank_bm25 import BM25Okapi
from transformers import AutoModel, AutoTokenizer


def get_bm25(corpus):
    bm25 = BM25Okapi(corpus)
    return bm25


def get_bert(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model
