"""
crag.py
───────
CRAG components:
  crag_gate:    cross-encoder confidence gate
  rerank_passages: cross-encoder passage reranking
  crag_retrieve:   full corrective retrieval pipeline
"""

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import (
    CrossEncoder, SentenceTransformer)

import sys
sys.path.append("/content/drive/MyDrive/MedQA_Project")
from src.config import CRAG_TAU, TOP_K
from src.retriever import retrieve_faiss, retrieve_bm25


def crag_gate(question: str,
               passage_text: str,
               cross_encoder: CrossEncoder,
               tau: float = CRAG_TAU) -> dict:
    """
    Evaluate whether a retrieved passage is relevant
    enough to pass to the reader.

    Returns dict with score, passed, decision.
    """
    score  = float(
        cross_encoder.predict(
            [(question, passage_text)])[0])
    passed = score >= tau
    return {
        "score":    score,
        "passed":   passed,
        "decision": "USE_PASSAGE"
                     if passed else "TRIGGER_FALLBACK",
    }


def rerank_passages(question: str,
                     passages: list,
                     cross_encoder: CrossEncoder
                     ) -> list:
    """
    Rerank retrieved passages by cross-encoder score.
    Most relevant passage moves to index 0.
    """
    if len(passages) <= 1:
        return passages

    pairs         = [(question, p['text'])
                      for p in passages]
    rerank_scores = [
        float(s) for s in
        cross_encoder.predict(pairs)
    ]

    for p, s in zip(passages, rerank_scores):
        p['rerank_score'] = s

    return sorted(passages,
                   key=lambda p: p['rerank_score'],
                   reverse=True)


def crag_retrieve(question: str,
                   options: list,
                   faiss_index: faiss.IndexIVFFlat,
                   bm25_index: BM25Okapi,
                   kb_docs: list,
                   retriever_model: SentenceTransformer,
                   cross_encoder: CrossEncoder,
                   tau: float = CRAG_TAU,
                   top_k: int = TOP_K) -> dict:
    """
    Complete CRAG retrieval pipeline.

    Steps:
      1. FAISS dense retrieval (top-K)
      2. CRAG gate on top-1 passage
      3. BM25 fallback if gate fails
      4. Rerank selected passages
      5. Return reranked passages with metadata

    Returns dict with passages, retrieval_type,
    gate_score.
    """
    # Step 1: FAISS
    faiss_passages = retrieve_faiss(
        question, faiss_index, kb_docs,
        retriever_model, top_k)

    if not faiss_passages:
        passages       = retrieve_bm25(
            question, options, bm25_index,
            kb_docs, top_k)
        retrieval_type = "bm25_primary"
        gate_score     = None
    else:
        # Step 2: gate
        gate_result = crag_gate(
            question, faiss_passages[0]['text'],
            cross_encoder, tau)
        gate_score  = gate_result['score']

        if gate_result['passed']:
            passages       = faiss_passages
            retrieval_type = "faiss"
        else:
            # Step 3: BM25 fallback
            bm25_passages = retrieve_bm25(
                question, options, bm25_index,
                kb_docs, top_k)

            if bm25_passages:
                bm25_gate = crag_gate(
                    question,
                    bm25_passages[0]['text'],
                    cross_encoder,
                    tau - 1.0)
                if bm25_gate['score'] > gate_score:
                    passages       = bm25_passages
                    retrieval_type = "bm25_fallback"
                else:
                    passages       = faiss_passages
                    retrieval_type = "faiss_forced"
            else:
                passages       = faiss_passages
                retrieval_type = "faiss_forced"

    # Step 4: rerank
    passages = rerank_passages(
        question, passages, cross_encoder)

    return {
        "passages":       passages,
        "retrieval_type": retrieval_type,
        "gate_score":     gate_score,
    }
