# 🧠 Medical MCQA with RAG and CRAG

## 📌 Project Overview

This project builds a **Medical Multiple Choice Question Answering (MCQA)** system over the MedMCQA dataset — 194,000 questions from AIIMS and NEET PG entrance examinations.

The system retrieves relevant medical passages from a **156,555-document knowledge base** and selects the correct option from four candidates.

A **Corrective RAG (CRAG)** layer detects retrieval failures and triggers keyword-based fallback before the answer model is invoked.

All components use **pretrained models** with no domain-specific fine-tuning from scratch.

**Final system:** Medical CE + CRAG v3 — **39.67% accuracy**  
**Encoder-decoder comparison:** Flan-T5-large — **48.76%**

---

## 🚀 Key Results

| System | Accuracy | Delta |
|--------|--------|-------|
| Random chance | 25.00% | — |
| No-RAG (CE only) | 28.97% | +3.97% |
| RAG baseline | 35.42% | +6.45% |
| CRAG v1 (gate) | 35.04% | −0.38% |
| CRAG v2 (+ BM25 fallback) | 39.29% | +4.25% |
| CRAG v3 (+ reranking) | 40.01% | +0.72% |
| **Medical CE — Final system** | **39.67%** | −0.34% |
| Flan-T5-large (comparison) | 48.76% | +9.09% |

---

## 🧪 Fine-Tuning Experiments

| Attempt | Strategy | Accuracy | Outcome |
|--------|----------|----------|--------|
| Attempt 1 | Random negatives | 37.52% | Failed |
| Attempt 2 | Hard negatives + mixed data | 33.27% | Failed |

---

## 📊 Retrieval Metrics

| Metric | Value |
|--------|------|
| Recall@3 | 0.973 |
| Precision@1 | 0.1603 |
| Precision@3 | 0.0867 |
| Precision@5 | 0.0654 |
| MRR | 0.2283 |
| Query latency | 4.34 ms |

## 📂 Repository Structure

```
MedQA_Project/
├── src/
│   ├── config.py
│   ├── data_processing.py
│   ├── knowledge_base.py
│   ├── retriever.py
│   ├── crag.py
│   ├── mcq_pipeline.py
│   ├── evaluation.py
│   └── models.py
├── experiments/
│   ├── 01_data_processing.ipynb
│   ├── 02_build_index.ipynb
│   ├── 03_baseline_rag.ipynb
│   ├── 04_crag_ablation.ipynb
│   ├── 05_encoder_comparison.ipynb
│   ├── 06_finetuning_attempt.ipynb
│   ├── 07_flan_t5_comparison.ipynb
│   └── 08_final_evaluation.ipynb
├── results/
├── data/
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Pipeline

```
Question
↓
[Stage 1] Query Encoding — all-MiniLM-L6-v2 (384-dim)
↓
[Stage 2] FAISS Dense Retrieval — top-5 passages
↓
[Stage 3] CRAG Gate — cross-encoder score vs τ=0.5
├── PASS → use FAISS passages
└── FAIL → BM25 fallback with query expansion
↓
[Stage 4] Passage Reranking — cross-encoder
↓
[Stage 5] Option Scoring — Medical CE
↓
Predicted Answer (A/B/C/D)
```

---

## 🤖 Models

| Model | Role | Params | Why chosen |
|------|------|--------|-----------|
| all-MiniLM-L6-v2 | Query encoding, FAISS retrieval | 22M | Fine-tuned on 1B sentence pairs for semantic similarity |
| ms-marco-MiniLM-L-6-v2 | CRAG gate, reranking | 22M | Trained on MS-MARCO relevance scoring |
| ms-marco-MiniLM-L-12-v2 | Final option scoring | 22M | Same task, deeper model |
| BM25Okapi | Keyword fallback retrieval | — | Complementary retrieval |
| Flan-T5-large | Encoder-decoder comparison | 780M | Joint reasoning across options |

---

## 🔍 Key Findings

### 1. RAG contributes +6.45%
Retrieved context improves answer selection over model prior knowledge.

### 2. CRAG contributes cumulatively
- Gate detects bad retrievals  
- BM25 fallback recovers them  
- Reranking improves ordering  

### 3. Task alignment beats domain alignment
BioBERT (medical domain) performs worse than MS-MARCO (general web)  
because **task match > domain match**

### 4. Encoder-only ceiling (~40%)
Cannot compare answer options jointly.

### 5. Encoder-decoder breaks the ceiling
Flan-T5 achieves **+9.09% improvement**

### 6. Fine-tuning failed
Both attempts reduced accuracy — pretrained models performed better.

---

## 🧪 Reproducing Results

### Install dependencies

```bash
pip install faiss-cpu sentence-transformers rank-bm25 \
            transformers torch datasets
```
## ▶️ Run Notebooks (Execution Order)

Run the notebooks in the following order:

```
01_data_processing.ipynb     → Load and clean MedMCQA dataset
02_build_index.ipynb        → Build FAISS + BM25 index
03_baseline_rag.ipynb       → Evaluate baseline RAG system
04_crag_ablation.ipynb      → Test CRAG components (gate, fallback, reranking)
05_encoder_comparison.ipynb → Compare encoder models
06_finetuning_attempt.ipynb → Fine-tuning experiments
07_flan_t5_comparison.ipynb → Encoder-decoder comparison
08_final_evaluation.ipynb   → Final metrics and summary
```
