"""Output-based stealth metrics (KL, KS, surprise) with calibrated thresholds.

Threshold design:
1. KL: use chi-square approximation -- chi2.ppf(0.95, K-1) / (2*n) as the
   lower bound, and ensure >= 10*null_kl to guard against reference-sample
   estimation error.
2. KS: dual condition -- ks_pval < 0.05 AND ks_stat > 0.05, to rule out
   spurious significance under large samples.
3. Surprise: surprise_sigma > 2.0, maintaining the 2-sigma criterion.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import torch
from scipy.stats import chi2, entropy, ks_2samp


# ----------------- Helpers (rank/surprise extraction) -----------------------

def compute_token_ranks_surprises(token_ids: List[int], model, tokenizer,
                                  context_prompt: str = "",
                                  temperature: float = 0.7,
                                  top_k: int = 50) -> Dict:
    """Compute per-token rank and surprise for a token sequence under the model.

    Uses KV cache to avoid O(N²) memory blowup when replaying long sequences
    (e.g. paraphrase outputs in Exp-D). Each step only feeds the new token and
    reuses past_key_values, mirroring the generate_with_attack path.

    Used when only token_ids are available (e.g. for already-generated text).
    """
    device = next(model.parameters()).device
    if context_prompt:
        ctx_ids = tokenizer.encode(context_prompt, return_tensors="pt").to(device)
    else:
        # use BOS / first token as context
        bos = tokenizer.bos_token_id or tokenizer.eos_token_id or 0
        ctx_ids = torch.tensor([[bos]], device=device)

    ranks: List[int] = []
    surprises: List[float] = []
    input_ids = ctx_ids
    past_key_values = None
    with torch.no_grad():
        for tid in token_ids:
            outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :].float()
            if temperature and temperature > 0:
                logits = logits / temperature
            if top_k and top_k > 0:
                top_k_vals, _ = logits.topk(top_k, dim=-1)
                threshold = top_k_vals[:, -1:]
                logits = torch.where(logits < threshold,
                                     torch.full_like(logits, float("-inf")),
                                     logits)
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            tp = float(probs[tid].item()) if 0 <= tid < probs.numel() else 1e-30
            rank = int((probs >= probs[tid]).sum().item()) if 0 <= tid < probs.numel() else probs.numel()
            ranks.append(rank)
            surprises.append(-math.log(tp + 1e-30))
            # Next iteration only needs the new token (KV cache holds the rest)
            input_ids = torch.tensor([[tid]], device=device)
    return {'ranks': ranks, 'surprises': surprises}


# ----------------- Statistical tests ----------------------------------------

def compute_kl_divergence(attack_ranks: List[int], clean_ranks: List[int],
                          n_bins: int = 50) -> float:
    if not attack_ranks or not clean_ranks:
        return 0.0
    max_rank = max(max(attack_ranks, default=1), max(clean_ranks, default=1)) + 1
    bins = np.linspace(0, min(max_rank, n_bins), n_bins + 1)
    hist_attack, _ = np.histogram(attack_ranks, bins=bins, density=True)
    hist_clean, _ = np.histogram(clean_ranks, bins=bins, density=True)
    eps = 1e-10
    hist_attack = hist_attack + eps
    hist_clean = hist_clean + eps
    hist_attack = hist_attack / hist_attack.sum()
    hist_clean = hist_clean / hist_clean.sum()
    return float(entropy(hist_attack, hist_clean))


def compute_ks_test(attack_ranks: List[int], clean_ranks: List[int]) -> Dict:
    if len(attack_ranks) < 2 or len(clean_ranks) < 2:
        return {'ks_stat': 0.0, 'ks_pval': 1.0}
    stat, pval = ks_2samp(attack_ranks, clean_ranks)
    return {'ks_stat': float(stat), 'ks_pval': float(pval)}


def compute_surprise_gap(attack_surprises: List[float],
                         clean_surprises: List[float]) -> Dict:
    mean_attack = float(np.mean(attack_surprises)) if attack_surprises else 0.0
    mean_clean = float(np.mean(clean_surprises)) if clean_surprises else 0.0
    gap = abs(mean_attack - mean_clean)
    sigma = gap / (np.std(clean_surprises) + 1e-10) if clean_surprises else 0.0
    return {
        'surprise_gap': float(gap),
        'surprise_sigma': float(sigma),
        'mean_surprise_attack': mean_attack,
        'mean_surprise_clean': mean_clean,
    }


# ----------------- Main entrypoint ------------------------------------------

def evaluate_stealth(attack_token_ids=None, clean_token_ids=None,
                     model=None, tokenizer=None,
                     n_tokens: int = 2000,
                     attack_ranks: Optional[List[int]] = None,
                     attack_surprises: Optional[List[float]] = None,
                     clean_ranks: Optional[List[int]] = None,
                     clean_surprises: Optional[List[float]] = None,
                     null_kl: Optional[float] = None,
                     n_bins: int = 50,
                     context_prompt: str = "") -> Dict:
    """Full stealth evaluation under three calibrated detectors.

    Two calling conventions:
    1. Pass pre-computed (attack_ranks, attack_surprises, clean_ranks, clean_surprises).
    2. Pass attack_token_ids / clean_token_ids + model + tokenizer; this function
       will compute ranks/surprises by replaying token-by-token under the model.

    Returns dict with kl_divergence, ks_stat, ks_pval, surprise_sigma,
    detectors_triggered (0–3), is_stealthy, kl_threshold.
    """
    # Auto-compute ranks/surprises if not provided
    if attack_ranks is None or attack_surprises is None:
        if attack_token_ids is None or model is None or tokenizer is None:
            raise ValueError("Need either ranks+surprises or token_ids+model+tokenizer.")
        a = compute_token_ranks_surprises(attack_token_ids, model, tokenizer,
                                          context_prompt=context_prompt)
        attack_ranks = a['ranks']
        attack_surprises = a['surprises']
    if clean_ranks is None or clean_surprises is None:
        if clean_token_ids is None or model is None or tokenizer is None:
            raise ValueError("Need either ranks+surprises or token_ids+model+tokenizer.")
        c = compute_token_ranks_surprises(clean_token_ids, model, tokenizer,
                                          context_prompt=context_prompt)
        clean_ranks = c['ranks']
        clean_surprises = c['surprises']

    kl = compute_kl_divergence(attack_ranks, clean_ranks, n_bins=n_bins)
    ks = compute_ks_test(attack_ranks, clean_ranks)
    surprise = compute_surprise_gap(attack_surprises, clean_surprises)

    alpha = 0.05
    min_effect_size = 0.15  # midpoint of small-effect range for large-sample KS
    n_eff = max(int(n_tokens) if n_tokens else 2000, 1)
    kl_threshold_chi2 = chi2.ppf(1 - alpha, n_bins - 1) / (2 * n_eff)

    # Auto-compute null_kl via split-half if not provided
    if null_kl is None and clean_ranks and len(clean_ranks) >= 20:
        mid = len(clean_ranks) // 2
        null_kl = compute_kl_divergence(clean_ranks[:mid], clean_ranks[mid:], n_bins=n_bins)

    if null_kl is not None and null_kl > 0:
        kl_threshold = max(kl_threshold_chi2, 10 * null_kl)
    else:
        kl_threshold = max(kl_threshold_chi2, 0.1)

    triggered = 0
    detectors = {}
    detectors['rank_ks'] = bool(
        (ks['ks_pval'] < alpha) and (ks['ks_stat'] > min_effect_size)
    )
    if detectors['rank_ks']:
        triggered += 1
    detectors['kl'] = bool(kl > kl_threshold)
    if detectors['kl']:
        triggered += 1
    detectors['surprise'] = bool(surprise['surprise_sigma'] > 2.0)
    if detectors['surprise']:
        triggered += 1

    return {
        'kl_divergence': float(kl),
        **ks,
        **surprise,
        'null_kl': null_kl,
        'kl_threshold': float(kl_threshold),
        'n_tokens': n_eff,
        'detectors_triggered': int(triggered),
        'detector_details': detectors,
        'is_stealthy': bool(triggered == 0),
    }
