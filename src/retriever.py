"""
retriever.py
────────────
FAISS dense retrieval and BM25 sparse retrieval.
Both return passages in the same dict format.
"""

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

import sys
sys.path.append("/content/drive/MyDrive/MedQA_Project")
from src.config import TOP_K, BM25_STOPWORDS
from src.knowledge_base import tokenize_for_bm25


def retrieve_faiss(question: str,
                    index: faiss.IndexIVFFlat,
                    kb_docs: list,
                    retriever_model: SentenceTransformer,
                    top_k: int = TOP_K) -> list:
    """
    Retrieve top-K passages using FAISS dense retrieval.

    Args:
        question:        medical question string
        index:           loaded FAISS index
        kb_docs:         knowledge base document list
        retriever_model: loaded SentenceTransformer
        top_k:           number of passages to return

    Returns:
        list of passage dicts sorted by score descending
    """
    query_emb = retriever_model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    distances, indices = index.search(query_emb, top_k)

    passages = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        passages.append({
            "text":             kb_docs[idx]['text'],
            "subject":          kb_docs[idx]['subject'],
            "topic":            kb_docs[idx]['topic'],
            "score":            float(dist),
            "doc_idx":          int(idx),
            "retrieval_method": "faiss",
        })
    return passages


def retrieve_bm25(question: str,
                   options: list,
                   bm25_index: BM25Okapi,
                   kb_docs: list,
                   top_k: int = TOP_K) -> list:
    """
    Retrieve top-K passages using BM25 sparse retrieval
    with query expansion using option text.

    Args:
        question:   medical question string
        options:    list of 4 option strings
        bm25_index: loaded BM25Okapi index
        kb_docs:    knowledge base document list
        top_k:      number of passages to return

    Returns:
        list of passage dicts in same format as FAISS
    """
    option_text = " ".join(options) if options else ""
    expanded    = f"{question} {option_text}"
    tokens      = tokenize_for_bm25(expanded)
    tokens      = [
        t for t in tokens
        if t not in BM25_STOPWORDS and len(t) > 2
    ]

    if not tokens:
        return []

    scores      = bm25_index.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]

    passages = []
    for idx in top_indices:
        if scores[idx] <= 0:
            continue
        passages.append({
            "text":             kb_docs[idx]['text'],
            "subject":          kb_docs[idx]['subject'],
            "topic":            kb_docs[idx]['topic'],
            "score":            float(scores[idx]),
            "doc_idx":          int(idx),
            "retrieval_method": "bm25",
        })
    return passages
