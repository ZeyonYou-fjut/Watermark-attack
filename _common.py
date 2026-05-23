"""Shared config & helpers for ACSAC watermark-attack experiments."""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# --- Path setup -------------------------------------------------------------
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# HuggingFace mirror (optional, for users behind the GFW)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ----- Model registry -------------------------------------------------------
MODELS = {
    "qwen2-7b": "Qwen/Qwen2-7B-Instruct",
    "llama3-8b": "TsinghuaC3I/Llama-3-8B-UltraMedical",
    "mistral-7b": "BioMistral/BioMistral-7B",
}

PRIMARY_MODEL = MODELS["qwen2-7b"]

# ----- Default attack parameters -------------------------------------------
DEFAULT_BOOST = 50
DEFAULT_ACTIVATION = 0.7
DEFAULT_N_TOKENS = 2000

DEFAULT_PROMPT = (
    "Write a detailed essay about the impact of artificial intelligence on "
    "modern society, covering its effects on employment, education, healthcare, "
    "and ethical considerations."
)

# ----- Default watermark parameters ----------------------------------------
DEFAULT_KGW_PARAMS = {"delta": 2.0, "gamma": 0.25}
DEFAULT_UNIGRAM_PARAMS = {"delta": 2.0, "gamma": 0.25}
DEFAULT_DIPMARK_PARAMS = {"alpha": 1.0, "gamma": 0.25}

DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_K = 50
ATTACK_SEED = 42
WATERMARK_KEY = 42

# ----- Target / suppress word lists ----------------------------------------
TARGET_WORDS = [
    "the", "is", "of", "and", "to", "in", "that", "it", "for", "was",
    "on", "are", "with", "as", "this", "have", "from", "be", "or", "an",
    "will", "can", "has", "but", "not", "you", "all", "they", "their", "which",
    "would", "there", "been", "one", "our", "more", "when", "about", "into", "could",
]


# ----- Model loading --------------------------------------------------------

def load_model(model_id: str, device: str = "cuda"):
    """Load model + tokenizer, freeing GPU memory beforehand.
    
    Uses device_map=None + .to(device) for predictable VRAM usage and faster inference.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load_model] {model_id} -> {device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    vram = torch.cuda.memory_allocated() // 1024 // 1024 if torch.cuda.is_available() else 0
    print(f"[load_model] done. VRAM: {vram} MB", flush=True)
    return model, tokenizer


def free_model(model=None, tokenizer=None):
    """Release model & tokenizer; flush CUDA caches."""
    try:
        if model is not None:
            del model
    except Exception:
        pass
    try:
        if tokenizer is not None:
            del tokenizer
    except Exception:
        pass
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ----- Save helpers ---------------------------------------------------------

def save_results(data: dict, filename: str) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / filename
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    print(f"[save_results] {path}", flush=True)
    return path


def now_iso() -> str:
    return datetime.now().isoformat(timespec='seconds')


# ----- Target token id construction -----------------------------------------

def get_target_ids(tokenizer, target_words: Optional[List[str]] = None,
                   n_targets: int = 40) -> List[int]:
    """Map common high-frequency words → token ids (with leading-space variant)."""
    words = target_words if target_words is not None else TARGET_WORDS
    words = words[:n_targets]
    ids = set()
    for w in words:
        for variant in (w, " " + w, w.capitalize(), " " + w.capitalize()):
            try:
                toks = tokenizer.encode(variant, add_special_tokens=False)
            except Exception:
                continue
            if toks:
                ids.add(int(toks[0]))
    return sorted(ids)


# ----- Lift computation -----------------------------------------------------

def compute_lift(generated_ids: List[int], target_ids: List[int],
                 baseline_ids: Optional[List[int]] = None) -> float:
    """Lift = (target frequency under attack) / (target frequency under baseline)."""
    target_set = set(target_ids)
    n = max(1, len(generated_ids))
    rate_attack = sum(1 for t in generated_ids if t in target_set) / n
    if baseline_ids is None:
        baseline_rate = 0.05  # rough natural prior
    else:
        nb = max(1, len(baseline_ids))
        baseline_rate = sum(1 for t in baseline_ids if t in target_set) / nb
    if baseline_rate <= 0:
        baseline_rate = 1e-6
    return rate_attack / baseline_rate
