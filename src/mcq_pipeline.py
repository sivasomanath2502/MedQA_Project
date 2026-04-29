"""
mcq_pipeline.py
───────────────
MCQ answering pipeline variants.

1. no_rag_pipeline:   CE scores options, no context
2. rag_pipeline:      FAISS + CE option scoring
3. crag_pipeline:     Full CRAG + CE option scoring
All return the same output dict format.
"""

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import (
    CrossEncoder, SentenceTransformer)

import sys
sys.path.append("/content/drive/MyDrive/MedQA_Project")
from src.config import TOP_K, CRAG_TAU
from src.retriever import retrieve_faiss
from src.crag import crag_retrieve


def score_options(question: str,
                   options: list,
                   passage_text: str,
                   cross_encoder: CrossEncoder) -> dict:
    """
    Score all four options against a passage.
    Returns predicted_idx, predicted_option,
    option_scores, confidence.
    """
    pairs  = [(f"{question} {opt}", passage_text)
               for opt in options]
    scores = cross_encoder.predict(pairs)

    predicted_idx = int(np.argmax(scores))
    s_arr  = np.array(scores, dtype=np.float64)
    exp_s  = np.exp(s_arr - s_arr.max())
    conf   = float(
        exp_s[predicted_idx] / exp_s.sum())

    return {
        "predicted_idx":    predicted_idx,
        "predicted_option": options[predicted_idx],
        "option_scores":    scores.tolist(),
        "confidence":       conf,
    }


def no_rag_pipeline(question: str,
                     options: list,
                     cross_encoder: CrossEncoder
                     ) -> dict:
    """CE scores options against empty context."""
    empty  = "No additional context available."
    result = score_options(
        question, options, empty, cross_encoder)
    return {
        **result,
        "retrieval_type": "no_rag",
        "source_passage": "",
        "source_subject": "None",
        "gate_score":     None,
    }


def rag_pipeline(question: str,
                  options: list,
                  faiss_index: faiss.IndexIVFFlat,
                  kb_docs: list,
                  retriever_model: SentenceTransformer,
                  cross_encoder: CrossEncoder,
                  top_k: int = TOP_K) -> dict:
    """FAISS retrieval + CE option scoring. No CRAG."""
    passages = retrieve_faiss(
        question, faiss_index, kb_docs,
        retriever_model, top_k)

    if not passages:
        top_passage = "No context retrieved."
        source_subj = "Unknown"
    else:
        top_passage = passages[0]['text']
        source_subj = passages[0]['subject']

    result = score_options(
        question, options, top_passage, cross_encoder)

    return {
        **result,
        "retrieval_type": "faiss",
        "source_passage": top_passage[:200],
        "source_subject": source_subj,
        "gate_score":     None,
    }


def crag_pipeline(question: str,
                   options: list,
                   faiss_index: faiss.IndexIVFFlat,
                   bm25_index: BM25Okapi,
                   kb_docs: list,
                   retriever_model: SentenceTransformer,
                   cross_encoder: CrossEncoder,
                   tau: float = CRAG_TAU,
                   top_k: int = TOP_K) -> dict:
    """
    Full CRAG pipeline — final encoder-only system.
    Gate + BM25 fallback + reranking + CE scoring.
    """
    retrieval = crag_retrieve(
        question, options,
        faiss_index, bm25_index, kb_docs,
        retriever_model, cross_encoder,
        tau=tau, top_k=top_k,
    )

    passages = retrieval['passages']

    if not passages:
        top_passage = "No context retrieved."
        source_subj = "Unknown"
    else:
        top_passage = passages[0]['text']
        source_subj = passages[0]['subject']

    result = score_options(
        question, options, top_passage, cross_encoder)

    return {
        **result,
        "retrieval_type": retrieval['retrieval_type'],
        "source_passage": top_passage[:200],
        "source_subject": source_subj,
        "gate_score":     retrieval['gate_score'],
    }
