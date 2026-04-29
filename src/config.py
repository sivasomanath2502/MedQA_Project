"""
config.py
─────────
Single source of truth for all constants, paths, and
hyperparameters used across the project.
"""

import os

# ── Google Drive base path ─────────────────────────────
BASE_DIR = "/content/drive/MyDrive/MedQA_Project"

# ── Sub-directories ────────────────────────────────────
DATA_DIR    = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
SRC_DIR     = os.path.join(BASE_DIR, "src")

# ── Data file paths ────────────────────────────────────
TRAIN_QA_PATH        = os.path.join(DATA_DIR, "train_qa.pkl")
VAL_QA_PATH          = os.path.join(DATA_DIR, "val_qa.pkl")
TEST_QA_PATH         = os.path.join(DATA_DIR, "test_qa.pkl")
FULL_KB_DOCS_PATH    = os.path.join(DATA_DIR, "full_kb_docs.pkl")
CLEANED_KB_PATH      = os.path.join(DATA_DIR, "cleaned_kb_docs.pkl")
VAL_EVAL_MCQ_PATH    = os.path.join(DATA_DIR, "val_eval_mcq.pkl")
VAL_TUNE_MCQ_PATH    = os.path.join(DATA_DIR, "val_tune_mcq.pkl")
TEST_MCQ_PATH        = os.path.join(DATA_DIR, "test_mcq.pkl")
CLEAN_TEST_QA_PATH   = os.path.join(DATA_DIR, "clean_test_qa.pkl")
KB_EMBEDDINGS_PATH   = os.path.join(DATA_DIR, "kb_embeddings_cleaned.npy")
TEST_EMBEDDINGS_PATH = os.path.join(DATA_DIR, "test_embeddings.npy")
FAISS_INDEX_PATH     = os.path.join(DATA_DIR, "kb_index_cleaned.faiss")
INDEX_CONFIG_PATH    = os.path.join(DATA_DIR, "index_config.pkl")
LEAKAGE_RESULTS_PATH = os.path.join(DATA_DIR, "leakage_results.pkl")
CLEAN_TEST_MCQ_PATH  = os.path.join(DATA_DIR, "clean_test_mcq.pkl")

# ── Results file paths ─────────────────────────────────
ABLATION_RESULTS_PATH  = os.path.join(RESULTS_DIR, "ablation_results.json")
SUBJECT_ACC_PATH       = os.path.join(RESULTS_DIR, "subject_accuracy.json")
RETRIEVAL_METRICS_PATH = os.path.join(RESULTS_DIR, "retrieval_metrics.json")
FINAL_METRICS_PATH     = os.path.join(RESULTS_DIR, "final_metrics.json")
CRAG_CONFIG_PATH       = os.path.join(MODELS_DIR,  "crag_config.pkl")
FT_MODEL_PATH          = os.path.join(MODELS_DIR,  "cross_encoder_finetuned")

# ── Model names ────────────────────────────────────────
RETRIEVER_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
CE_BASE_MODEL    = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CE_MEDICAL_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
FLAN_T5_MODEL    = "google/flan-t5-large"

# ── FAISS hyperparameters ──────────────────────────────
FAISS_NLIST   = 384
FAISS_NPROBE  = 50
EMBEDDING_DIM = 384

# ── Retrieval hyperparameters ──────────────────────────
TOP_K    = 5
CRAG_TAU = 0.5

# ── BM25 stopwords ─────────────────────────────────────
BM25_STOPWORDS = {
    'the','is','are','was','were','be','following',
    'which','what','used','true','false','correct',
    'incorrect','all','none','except','not','given',
    'regarding','about','of','in','for','most',
    'least','best','common','commonest','a','an',
}

# ── Cleaning hyperparameters ───────────────────────────
MIN_EXP_LENGTH      = 20
SIMILARITY_LEAK_THR = 0.92

# ── Fine-tuning hyperparameters ────────────────────────
FT_LEARNING_RATE = 5e-6
FT_EPOCHS        = 1
FT_BATCH_SIZE    = 32
FT_MAX_PAIRS     = 60000
FT_NEG_RATIO     = 0.3

# ── Evaluation ─────────────────────────────────────────
PRECISION_K_VALUES = [1, 3, 5]
MRR_K              = 10


def ensure_dirs():
    """Create all required directories if they do not exist."""
    for d in [DATA_DIR, RESULTS_DIR, MODELS_DIR, SRC_DIR]:
        os.makedirs(d, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print("Config loaded. All directories verified.")
    print(f"  BASE_DIR:    {BASE_DIR}")
    print(f"  DATA_DIR:    {DATA_DIR}")
    print(f"  RESULTS_DIR: {RESULTS_DIR}")
    print(f"  MODELS_DIR:  {MODELS_DIR}")
    print(f"  FAISS_NPROBE:{FAISS_NPROBE}")
