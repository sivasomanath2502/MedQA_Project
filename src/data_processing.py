"""
data_processing.py
──────────────────
Handles all MedMCQA data loading, cleaning, filtering,
and splitting into knowledge base documents and QA pairs.
"""

import os
import re
import pickle
import numpy as np
from datasets import load_dataset
from collections import defaultdict

import sys
sys.path.append("/content/drive/MyDrive/MedQA_Project")
from src.config import (
    DATA_DIR, MIN_EXP_LENGTH,
    TRAIN_QA_PATH, VAL_QA_PATH, TEST_QA_PATH,
    FULL_KB_DOCS_PATH, CLEANED_KB_PATH,
    VAL_EVAL_MCQ_PATH, VAL_TUNE_MCQ_PATH,
    TEST_MCQ_PATH,
)


def clean_explanation(text: str) -> str:
    """
    Remove exam-style artefacts from explanation text.
    Handles answer prefixes, reference citations,
    leading symbols, and whitespace normalisation.
    """
    if not text or not isinstance(text, str):
        return ""

    answer_prefixes = [
        r'^Ans\.?\s*[\(\[]?\s*[a-dA-D]\s*[\)\]]?\s*i\.e\.?,?\s*',
        r'^Ans\.?\s*[\(\[]?\s*[a-dA-D]\s*[\)\]]?\s*',
        r'^Answer\s+is\s+[\(\[]?\s*[a-dA-D]\s*[\)\]]?\s*',
        r'^ANSWER\s*:?\s*[\(\[]?\s*[A-D]\s*[\)\]]?\s*',
        r'^The\s+answer\s+is\s+',
        r'^Correct\s+answer\s+is\s+[\(\[]?\s*[a-dA-D]\s*[\)\]]?\s*',
        r'^Option\s+[\(\[]?\s*[a-dA-D]\s*[\)\]]?\s*',
    ]
    for pattern in answer_prefixes:
        text = re.sub(pattern, '', text,
                       flags=re.IGNORECASE)

    ref_patterns = [
        r'Ref(?:erence)?\.?\s*:?\s*[-–]?\s*[A-Za-z].*?'
        r'\d+(?:st|nd|rd|th)?\s*(?:ed(?:ition)?\.?)?[^.]*[.,]?',
        r'\b[Pp]\.?\s*\d+\b',
        r'\b[Pp][Gg]\.?\s*\d+\b',
        r'\b[Pp]age\s+\d+\b',
        r'[Rr]efer\s+[A-Z][a-z]+\s+\d+[a-z]+',
    ]
    for pattern in ref_patterns:
        text = re.sub(pattern, ' ', text,
                       flags=re.IGNORECASE)

    text = re.sub(r'^[\*\#\•\-\–\—\s]+', '', text)
    text = re.sub(r'[\*\#]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def is_valid_explanation(exp,
                          min_length: int = MIN_EXP_LENGTH
                          ) -> bool:
    """Returns True if explanation is non-null and
    at least min_length characters after stripping."""
    if exp is None:
        return False
    if not isinstance(exp, str):
        return False
    if len(exp.strip()) < min_length:
        return False
    return True


def get_correct_answer_text(sample: dict) -> str:
    """Extract the correct option text from a sample."""
    options = [
        sample['opa'], sample['opb'],
        sample['opc'], sample['opd'],
    ]
    return options[int(sample['cop'])]


def load_medmcqa():
    """Download and return MedMCQA dataset splits."""
    print("Loading MedMCQA from HuggingFace...")
    print("First time: ~2-3 minutes (90MB download)")
    dataset = load_dataset("openlifescienceai/medmcqa")
    print(f"  Train:      {len(dataset['train']):,}")
    print(f"  Validation: {len(dataset['validation']):,}")
    print(f"  Test:       {len(dataset['test']):,}")
    return dataset


def build_project_dataset(split_data,
                            split_name: str):
    """
    Transform a raw MedMCQA split into clean
    project format. Returns knowledge_docs and qa_pairs.
    """
    knowledge_docs = []
    qa_pairs       = []
    skipped        = 0

    for sample in split_data:
        correct_answer  = get_correct_answer_text(sample)
        has_explanation = is_valid_explanation(
            sample['exp'])

        qa_pairs.append({
            "id":              sample['id'],
            "question":        sample['question'],
            "answer":          correct_answer,
            "subject":         sample['subject_name'],
            "topic":           sample['topic_name'],
            "has_explanation": has_explanation,
        })

        if has_explanation:
            knowledge_docs.append({
                "id":      sample['id'],
                "text":    sample['exp'].strip(),
                "subject": sample['subject_name'],
                "topic":   sample['topic_name'],
            })
        else:
            skipped += 1

    print(f"{split_name}:")
    print(f"  Total:          {len(split_data):,}")
    print(f"  Knowledge docs: {len(knowledge_docs):,}")
    print(f"  QA pairs:       {len(qa_pairs):,}")
    print(f"  No explanation: {skipped:,}")
    return knowledge_docs, qa_pairs


def build_mcq_dataset_from_raw(split_data,
                                 split_name: str):
    """
    Build full MCQ dataset (with all option fields).
    Used for evaluation where opa/opb/opc/opd/cop needed.
    """
    mcq_samples = []
    skipped     = 0

    for sample in split_data:
        if not all([sample['opa'], sample['opb'],
                    sample['opc'], sample['opd']]):
            skipped += 1
            continue

        mcq_samples.append({
            "id":       sample['id'],
            "question": sample['question'],
            "opa":      sample['opa'],
            "opb":      sample['opb'],
            "opc":      sample['opc'],
            "opd":      sample['opd'],
            "cop":      int(sample['cop']),
            "answer":   get_correct_answer_text(sample),
            "subject":  sample['subject_name'],
            "topic":    sample['topic_name'],
        })

    print(f"{split_name} MCQ: {len(mcq_samples):,} "
          f"({skipped} skipped)")
    return mcq_samples


def clean_knowledge_base(kb_docs: list) -> tuple:
    """Apply clean_explanation() to all KB documents."""
    cleaned_docs = []
    stats = {"total": 0, "changed": 0,
              "became_empty": 0}

    for doc in kb_docs:
        original = doc['text']
        cleaned  = clean_explanation(original)
        stats["total"] += 1

        if not cleaned:
            stats["became_empty"] += 1
            cleaned = original

        if cleaned != original:
            stats["changed"] += 1

        cleaned_docs.append({
            "id":      doc['id'],
            "text":    cleaned,
            "subject": doc['subject'],
            "topic":   doc['topic'],
        })

    print(f"Cleaning complete:")
    print(f"  Total:        {stats['total']:,}")
    print(f"  Changed:      {stats['changed']:,} "
          f"({stats['changed']/stats['total']*100:.1f}%)")
    print(f"  Became empty: {stats['became_empty']:,}")
    return cleaned_docs, stats


def split_validation_set(val_mcq: list,
                           seed: int = 42):
    """Split validation MCQ set 50/50 into tune/eval."""
    np.random.seed(seed)
    indices     = np.random.permutation(len(val_mcq))
    split_point = len(val_mcq) // 2
    val_tune    = [val_mcq[i]
                    for i in indices[:split_point]]
    val_eval    = [val_mcq[i]
                    for i in indices[split_point:]]
    print(f"Validation split (seed={seed}):")
    print(f"  Tune: {len(val_tune):,}")
    print(f"  Eval: {len(val_eval):,}")
    return val_tune, val_eval


def save_pkl(obj, path: str, label: str = ""):
    """Save object as pickle."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    size_mb = os.path.getsize(path) / (1024 ** 2)
    name    = label or os.path.basename(path)
    print(f"  Saved {name}: {size_mb:.1f} MB")


def load_pkl(path: str, label: str = ""):
    """Load pickle file."""
    with open(path, "rb") as f:
        obj = pickle.load(f)
    name = label or os.path.basename(path)
    print(f"  Loaded {name}")
    return obj


def subject_distribution(docs_or_qa: list,
                           key: str = "subject") -> dict:
    """Count samples per subject."""
    counts = defaultdict(int)
    for item in docs_or_qa:
        counts[item[key]] += 1
    return dict(sorted(counts.items(),
                        key=lambda x: x[1],
                        reverse=True))
