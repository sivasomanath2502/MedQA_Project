"""
evaluation.py
─────────────
All evaluation metrics in one place:
  Accuracy, Per-subject accuracy, Ablation delta,
  Retrieval P@K, MRR, Faithfulness, ECE,
  Failure type analysis.
"""

import json
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm

import sys
sys.path.append("/content/drive/MyDrive/MedQA_Project")
from src.config import (
    PRECISION_K_VALUES, MRR_K, RESULTS_DIR)


def evaluate_pipeline(qa_pairs: list,
                       pipeline_fn,
                       desc: str = "Evaluating"
                       ) -> dict:
    """
    Evaluate any MCQ pipeline on a list of QA pairs.
    pipeline_fn(question, options) → result dict.
    """
    correct          = 0
    retrieval_counts = defaultdict(int)
    subject_correct  = defaultdict(int)
    subject_total    = defaultdict(int)
    results          = []

    for qa in tqdm(qa_pairs, desc=desc):
        options = [qa['opa'], qa['opb'],
                   qa['opc'], qa['opd']]

        result     = pipeline_fn(qa['question'], options)
        is_correct = (result['predicted_idx'] == qa['cop'])

        if is_correct:
            correct += 1

        retrieval_counts[
            result.get('retrieval_type',
                        'unknown')] += 1
        subject_correct[qa['subject']] += int(is_correct)
        subject_total[qa['subject']]   += 1

        results.append({
            "question":       qa['question'],
            "correct_idx":    qa['cop'],
            "correct_option": options[qa['cop']],
            "predicted_idx":  result['predicted_idx'],
            "predicted_opt":  result['predicted_option'],
            "is_correct":     is_correct,
            "confidence":     result.get('confidence', 0.0),
            "retrieval_type": result.get(
                'retrieval_type', ''),
            "source_subject": result.get(
                'source_subject', ''),
            "source_passage": result.get(
                'source_passage', ''),
            "gate_score":     result.get(
                'gate_score', None),
            "subject":        qa['subject'],
        })

    n        = len(qa_pairs)
    accuracy = correct / n * 100

    subject_accuracy = {
        s: subject_correct[s] /
           subject_total[s] * 100
        for s in subject_total
    }

    return {
        "accuracy":         accuracy,
        "correct":          correct,
        "total":            n,
        "retrieval_counts": dict(retrieval_counts),
        "subject_accuracy": subject_accuracy,
        "results":          results,
    }


def compute_retrieval_precision_at_k(
        questions: list,
        kb_id_lookup: dict,
        retrieve_fn,
        k_values: list = None) -> dict:
    """
    Compute Precision@K for retrieval component.
    Relevance = retrieved doc has same ID as question's doc.
    """
    if k_values is None:
        k_values = PRECISION_K_VALUES

    matched = [q for q in questions
                if q['id'] in kb_id_lookup]
    print(f"Precision@K: {len(matched):,} matched questions")

    precision_sums = defaultdict(float)

    for qa in tqdm(matched, desc="Precision@K"):
        passages = retrieve_fn(qa['question'])
        retrieved_ids = [
            kb_id_lookup.get(
                p['doc_idx'], {}).get('id', '')
            for p in passages
        ] if passages else []

        correct_id = qa['id']
        for k in k_values:
            top_k_ids = retrieved_ids[:k]
            precision_sums[k] += \
                int(correct_id in top_k_ids) / k

    return {
        k: precision_sums[k] / len(matched)
        for k in k_values
    }


def compute_mrr(questions: list,
                 kb_id_lookup: dict,
                 retrieve_fn,
                 k: int = MRR_K) -> float:
    """
    Compute Mean Reciprocal Rank for retriever.
    MRR = mean of (1 / rank_of_first_relevant_doc).
    """
    matched = [q for q in questions
                if q['id'] in kb_id_lookup]
    print(f"MRR: {len(matched):,} matched questions")

    rr_sum = 0.0
    for qa in tqdm(matched, desc="MRR"):
        passages = retrieve_fn(qa['question'])
        correct_idx = kb_id_lookup[qa['id']][0]

        for rank, p in enumerate(
                passages[:k], start=1):
            if p.get('doc_idx') == correct_idx:
                rr_sum += 1.0 / rank
                break

    return rr_sum / len(matched) if matched else 0.0


def compute_faithfulness(eval_results: list) -> dict:
    """
    Compute faithfulness: fraction of predictions
    where predicted option text appears in retrieved
    passage (case-insensitive substring match).
    """
    faithful_all     = 0
    faithful_correct = 0
    faithful_wrong   = 0
    n_correct        = 0
    n_wrong          = 0

    for r in eval_results:
        predicted   = r.get('predicted_opt', '').lower()
        passage     = r.get('source_passage', '').lower()
        is_faithful = (predicted in passage) \
                       if passage else False

        if r['is_correct']:
            n_correct += 1
            if is_faithful:
                faithful_correct += 1
        else:
            n_wrong += 1
            if is_faithful:
                faithful_wrong += 1

        if is_faithful:
            faithful_all += 1

    n = len(eval_results)
    return {
        "overall_faithfulness":   faithful_all / n
                                   if n else 0,
        "correct_faithfulness":   faithful_correct /
                                   n_correct
                                   if n_correct else 0,
        "incorrect_faithfulness": faithful_wrong /
                                   n_wrong
                                   if n_wrong else 0,
        "n_evaluated":            n,
    }


def compute_ece(eval_results: list,
                 n_bins: int = 10) -> dict:
    """
    Compute Expected Calibration Error.
    ECE measures how well confidence predicts accuracy.
    Lower is better. 0 = perfectly calibrated.
    """
    confidences = np.array(
        [r['confidence'] for r in eval_results])
    correctness = np.array(
        [r['is_correct'] for r in eval_results],
        dtype=float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece      = 0.0
    bin_data = []

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i+1]
        mask   = (confidences >= lo) & (confidences < hi)

        if mask.sum() == 0:
            continue

        bin_acc  = correctness[mask].mean()
        bin_conf = confidences[mask].mean()
        bin_n    = mask.sum()

        ece += (bin_n / len(eval_results)) * \
               abs(bin_acc - bin_conf)

        bin_data.append({
            "bin":        f"[{lo:.1f},{hi:.1f})",
            "count":      int(bin_n),
            "accuracy":   float(bin_acc),
            "confidence": float(bin_conf),
            "gap":        float(abs(bin_acc - bin_conf)),
        })

    return {
        "ece":      float(ece),
        "n_bins":   n_bins,
        "bin_data": bin_data,
    }


def analyse_failure_types(eval_results: list,
                            kb_id_lookup: dict) -> dict:
    """
    Categorise errors into four failure types.
    Type 1: wrong doc retrieved
    Type 2: right doc, wrong option
    Type 3: knowledge gap (forced fallback)
    Type 4: elimination / hard reasoning
    """
    wrong = [r for r in eval_results
              if not r['is_correct']]

    type1 = type2 = type3 = type4 = 0

    for r in wrong:
        rt         = r.get('retrieval_type', '')
        subj_match = (r.get('source_subject', '')
                      == r['subject'])

        if rt in ('faiss_forced', 'bm25_primary'):
            type3 += 1
        elif rt == 'bm25_fallback':
            type1 += 1
        elif rt == 'faiss' and not subj_match:
            type1 += 1
        elif rt == 'faiss' and subj_match:
            type2 += 1
        else:
            type4 += 1

    n = len(wrong)
    return {
        "total_errors":   n,
        "type1_count":    type1,
        "type2_count":    type2,
        "type3_count":    type3,
        "type4_count":    type4,
        "type1_fraction": type1/n if n else 0,
        "type2_fraction": type2/n if n else 0,
        "type3_fraction": type3/n if n else 0,
        "type4_fraction": type4/n if n else 0,
    }


def build_ablation_table(results_dict: dict) -> list:
    """Build ablation table from dict of system results."""
    rows = []
    prev = None
    for system, result in results_dict.items():
        acc   = result['accuracy']
        delta = acc - prev if prev is not None else None
        rows.append({
            "system":   system,
            "accuracy": acc,
            "delta":    delta,
        })
        prev = acc
    return rows


def print_ablation_table(rows: list):
    """Print formatted ablation table."""
    print(f"\n{'System':<45} "
          f"{'Accuracy':>9} {'Delta':>8}")
    print("-" * 65)
    for row in rows:
        delta_str = f"{row['delta']:+.2f}%" \
                     if row['delta'] is not None \
                     else "  ---  "
        print(f"{row['system']:<45} "
              f"{row['accuracy']:>8.2f}%  "
              f"{delta_str:>8}")


def save_results(results: dict,
                  path: str,
                  label: str = ""):
    """Save evaluation results as JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not serialisable: {type(obj)}")

    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=convert)

    size_kb = os.path.getsize(path) / 1024
    name    = label or os.path.basename(path)
    print(f"  Saved {name}: {size_kb:.1f} KB")
