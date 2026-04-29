"""
knowledge_base.py
─────────────────
Builds, saves, and loads FAISS vector index and
BM25 keyword index over the MedMCQA knowledge base.
"""

import os
import re
import pickle
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

import sys
sys.path.append("/content/drive/MyDrive/MedQA_Project")
from src.config import (
    FAISS_NLIST, FAISS_NPROBE, EMBEDDING_DIM,
    RETRIEVER_MODEL, BM25_STOPWORDS,
    KB_EMBEDDINGS_PATH, FAISS_INDEX_PATH,
    INDEX_CONFIG_PATH, DATA_DIR,
)


def tokenize_for_bm25(text: str) -> list:
    """Tokenise text for BM25 indexing."""
    text   = text.lower()
    text   = re.sub(r'[^\w\s\-]', ' ', text)
    tokens = [
        t for t in text.split()
        if len(t) > 1 and t not in BM25_STOPWORDS
    ]
    return tokens


def generate_embeddings(texts: list,
                         model: SentenceTransformer,
                         batch_size: int = 256,
                         show_progress: bool = True
                         ) -> np.ndarray:
    """
    Generate L2-normalised sentence embeddings.
    L2 normalisation ensures inner product equals
    cosine similarity in FAISS IndexFlatIP.
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def build_faiss_index(embeddings: np.ndarray,
                       nlist: int = FAISS_NLIST,
                       nprobe: int = FAISS_NPROBE
                       ) -> faiss.IndexIVFFlat:
    """
    Build a FAISS IVFFlat index from embeddings.
    Uses METRIC_INNER_PRODUCT on L2-normalised vectors
    which equals cosine similarity.
    """
    dim = embeddings.shape[1]
    print(f"Building FAISS IndexIVFFlat:")
    print(f"  Vectors:   {len(embeddings):,}")
    print(f"  Dimension: {dim}")
    print(f"  nlist:     {nlist}")
    print(f"  nprobe:    {nprobe}")

    quantizer = faiss.IndexFlatIP(dim)
    index     = faiss.IndexIVFFlat(
        quantizer, dim, nlist,
        faiss.METRIC_INNER_PRODUCT
    )

    print("  Training (k-means)...")
    index.train(embeddings)
    print(f"  Trained: {index.is_trained}")

    print("  Adding vectors...")
    index.add(embeddings)
    print(f"  Vectors in index: {index.ntotal:,}")

    index.nprobe = nprobe
    return index


def save_faiss_index(index: faiss.IndexIVFFlat,
                      embeddings: np.ndarray,
                      nprobe: int = FAISS_NPROBE):
    """Save FAISS index, embeddings, and config."""
    os.makedirs(DATA_DIR, exist_ok=True)

    faiss.write_index(index, FAISS_INDEX_PATH)
    size_mb = os.path.getsize(FAISS_INDEX_PATH)/(1024**2)
    print(f"  FAISS index saved: {size_mb:.0f} MB")

    np.save(KB_EMBEDDINGS_PATH, embeddings)
    emb_mb = os.path.getsize(KB_EMBEDDINGS_PATH)/(1024**2)
    print(f"  Embeddings saved:  {emb_mb:.0f} MB")

    config = {
        "nlist":         FAISS_NLIST,
        "nprobe":        nprobe,
        "embedding_dim": embeddings.shape[1],
        "n_vectors":     index.ntotal,
        "metric":        "inner_product",
        "model_name":    RETRIEVER_MODEL,
    }
    with open(INDEX_CONFIG_PATH, "wb") as f:
        pickle.dump(config, f)
    print(f"  Config saved with nprobe={nprobe}")


def load_faiss_index():
    """
    Load FAISS index and config from Drive.

    nprobe is NOT stored in the FAISS binary format.
    It must be restored from config after loading.

    The config file is the authoritative source.
    Falls back to FAISS_NPROBE constant if config
    value is outside valid range [10, nlist].

    Returns:
        index (faiss.IndexIVFFlat), config (dict)
    """
    index = faiss.read_index(FAISS_INDEX_PATH)

    with open(INDEX_CONFIG_PATH, "rb") as f:
        config = pickle.load(f)

    # Validate nprobe from config
    saved_nprobe = config.get('nprobe', FAISS_NPROBE)

    if not isinstance(saved_nprobe, int) or \
       saved_nprobe < 10 or \
       saved_nprobe > FAISS_NLIST:
        print(f"  WARNING: config nprobe={saved_nprobe} "
              f"outside valid range [10, {FAISS_NLIST}]")
        print(f"  Falling back to default: {FAISS_NPROBE}")
        saved_nprobe = FAISS_NPROBE

    index.nprobe     = saved_nprobe
    config['nprobe'] = saved_nprobe

    print(f"FAISS index loaded:")
    print(f"  Vectors: {index.ntotal:,}")
    print(f"  nprobe:  {index.nprobe}")
    print(f"  Trained: {index.is_trained}")

    return index, config


def tune_nprobe(index: faiss.IndexIVFFlat,
                embeddings: np.ndarray,
                n_samples: int = 100,
                top_k: int = 10,
                nprobe_values: list = None) -> tuple:
    """
    Find optimal nprobe by measuring Recall@K vs latency.
    Returns (results_dict, optimal_nprobe).
    """
    import time

    if nprobe_values is None:
        nprobe_values = [1, 5, 10, 20, 50, 100, 200]

    np.random.seed(42)
    sample_idx  = np.random.choice(
        len(embeddings), n_samples, replace=False)
    sample_embs = embeddings[
        sample_idx].astype(np.float32)

    # Ground truth: exact search
    original_nprobe = index.nprobe
    index.nprobe     = FAISS_NLIST
    _, exact_results = index.search(sample_embs, top_k)

    print(f"{'nprobe':>8} {'Recall@3':>10} "
          f"{'Recall@10':>11} {'Latency(ms)':>13}")
    print("-" * 48)

    results = {}
    for nprobe in nprobe_values:
        index.nprobe = nprobe

        start = time.time()
        for _ in range(20):
            index.search(sample_embs[:10], top_k)
        latency = (time.time()-start)/20/10*1000

        _, approx = index.search(sample_embs, top_k)
        r3 = r10 = 0
        for ex, ap in zip(exact_results, approx):
            r3  += len(set(ex[:3])  & set(ap[:3]))  / 3
            r10 += len(set(ex[:10]) & set(ap[:10])) / 10
        r3  /= n_samples
        r10 /= n_samples

        results[nprobe] = {
            "recall_at_3":  r3,
            "recall_at_10": r10,
            "latency_ms":   latency,
        }
        print(f"{nprobe:>8} {r3:>10.3f}  "
              f"{r10:>10.3f}  {latency:>12.2f}")

    # Restore
    index.nprobe = original_nprobe

    # Optimal: lowest nprobe where Recall@3 >= 0.95
    optimal = next(
        (n for n in nprobe_values
          if results[n]['recall_at_3'] >= 0.95),
        nprobe_values[-1]
    )
    print(f"\nOptimal nprobe: {optimal} "
          f"(Recall@3="
          f"{results[optimal]['recall_at_3']:.3f}, "
          f"Latency="
          f"{results[optimal]['latency_ms']:.2f}ms)")
    return results, optimal


def build_bm25_index(kb_docs: list) -> BM25Okapi:
    """Build BM25 index over knowledge base documents."""
    print(f"Building BM25 index over "
          f"{len(kb_docs):,} docs...")
    tokenized = [
        tokenize_for_bm25(doc['text'])
        for doc in kb_docs
    ]
    index = BM25Okapi(tokenized)
    print(f"  Vocabulary size: {len(index.idf):,} terms")
    return index


def save_bm25_index(bm25_index: BM25Okapi,
                     path: str):
    """Save BM25 index to pickle."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bm25_index, f)
    size_mb = os.path.getsize(path)/(1024**2)
    print(f"  BM25 index saved: {size_mb:.1f} MB")


def load_bm25_index(path: str) -> BM25Okapi:
    """Load BM25 index from pickle."""
    with open(path, "rb") as f:
        index = pickle.load(f)
    print(f"  BM25 index loaded: "
          f"{len(index.idf):,} terms")
    return index
