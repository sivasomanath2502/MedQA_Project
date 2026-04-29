"""
models.py
─────────
Centralised model loading.
All models loaded once and reused across experiments.
"""

import torch
from sentence_transformers import (
    SentenceTransformer, CrossEncoder)
from transformers import (
    T5ForConditionalGeneration, T5Tokenizer)

import sys
sys.path.append("/content/drive/MyDrive/MedQA_Project")
from src.config import (
    RETRIEVER_MODEL, CE_BASE_MODEL,
    CE_MEDICAL_MODEL, FLAN_T5_MODEL,
)


def load_retriever(model_name: str = RETRIEVER_MODEL
                   ) -> SentenceTransformer:
    """
    Load sentence-transformer retriever.
    all-MiniLM-L6-v2: 22M params, 384-dim output,
    fine-tuned on 1B sentence pairs for retrieval.
    """
    print(f"Loading retriever: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"  Parameters: ~22M  |  Output dim: 384")
    return model


def load_cross_encoder(model_name: str = CE_BASE_MODEL
                        ) -> CrossEncoder:
    """
    Load cross-encoder for CRAG gate, reranking,
    and option scoring.

    ms-marco-MiniLM-L-6-v2 (base, 22M):
      Trained on MS-MARCO passage relevance.
      Task-aligned for relevance scoring.

    ms-marco-MiniLM-L-12-v2 (medical CE, 22M):
      Same training, 12 layers vs 6.
      More capacity. Used as final system.
    """
    print(f"Loading cross-encoder: {model_name}")
    model = CrossEncoder(model_name, max_length=512)
    print(f"  Max length: 512 tokens")
    return model


def load_flan_t5(model_name: str = FLAN_T5_MODEL,
                  device: str = None) -> tuple:
    """
    Load Flan-T5-large for encoder-decoder comparison.
    NOT the final system — included to demonstrate
    encoder-only architectural limitation.

    780M params, ~3GB VRAM on float16.
    Instruction fine-tuned on 1800+ NLP tasks.
    Encodes question + all 4 options jointly.
    """
    if device is None:
        device = "cuda" \
                  if torch.cuda.is_available() \
                  else "cpu"

    print(f"Loading Flan-T5-large: {model_name}")
    print(f"  Device: {device}")
    print(f"  Note: comparison only, not final system")

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model     = T5ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    ).to(device)
    model.eval()

    if device == "cuda":
        vram = torch.cuda.memory_allocated() / 1e9
        print(f"  VRAM used: {vram:.1f} GB")

    return tokenizer, model, device


def get_device() -> torch.device:
    """Return available device."""
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: "
              f"{torch.cuda.get_device_name(0)}")
        total = torch.cuda.get_device_properties(
            0).total_memory
        print(f"  VRAM: {total/1e9:.1f} GB")
    return device
