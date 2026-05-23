"""KGW green-list watermark (Kirchenbauer et al., 2023).

Unified interface:
- KGWWatermark class: apply_watermark(logits, prev_token_id),
  get_green_list(prev_token_id), detect(token_ids, context_ids=None)
- generate_watermarked(model, tokenizer, prompt, n_tokens, **wm_params)
- detect(token_ids, tokenizer, **wm_params)
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import torch


class KGWWatermark:
    """KGW green-list watermark."""

    def __init__(self, vocab_size: int, key: int = 42,
                 delta: float = 2.0, gamma: float = 0.25):
        self.vocab_size = vocab_size
        self.key = key
        self.delta = delta
        self.gamma = gamma
        self.green_list_size = int(vocab_size * gamma)

    def get_green_list(self, prev_token_id: int) -> torch.Tensor:
        seed = self.key * 15485863 + int(prev_token_id)
        rng = torch.Generator()
        rng.manual_seed(seed)
        perm = torch.randperm(self.vocab_size, generator=rng)
        green_mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        green_mask[perm[:self.green_list_size]] = True
        return green_mask

    def apply_watermark(self, logits: torch.Tensor, prev_token_id: int) -> torch.Tensor:
        squeeze = False
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
            squeeze = True
        green_mask = self.get_green_list(prev_token_id).to(logits.device)
        watermarked = logits.clone()
        watermarked[:, green_mask] += self.delta
        return watermarked.squeeze(0) if squeeze else watermarked

    def detect(self, token_ids: List[int], context_ids: List[int] = None) -> dict:
        if context_ids is None:
            if len(token_ids) < 2:
                return {'z_score': 0.0, 'p_value': 1.0,
                        'green_fraction': 0.0, 'is_watermarked': False,
                        'n_green': 0, 'n_total': 0}
            context_ids = token_ids[:-1]
            token_ids = token_ids[1:]

        n_green = 0
        n_total = len(token_ids)
        for token, prev_token in zip(token_ids, context_ids):
            green_mask = self.get_green_list(prev_token)
            if green_mask[token]:
                n_green += 1

        green_fraction = n_green / n_total if n_total > 0 else 0
        expected = self.gamma * n_total
        std = np.sqrt(n_total * self.gamma * (1 - self.gamma))
        z_score = (n_green - expected) / std if std > 0 else 0
        from scipy.stats import norm
        p_value = float(1 - norm.cdf(z_score))
        return {
            'z_score': float(z_score),
            'p_value': p_value,
            'green_fraction': float(green_fraction),
            'n_green': int(n_green),
            'n_total': int(n_total),
            'is_watermarked': bool(z_score > 4.0),
        }


def generate_watermarked(model, tokenizer, prompt: str, n_tokens: int = 2000,
                         delta: float = 2.0, gamma: float = 0.25,
                         key: int = 42, temperature: float = 0.7,
                         top_k: int = 50) -> Tuple[List[int], float, str]:
    """Generate text with KGW watermark embedded.

    Returns: (token_ids, z_score, text)
    """
    device = next(model.parameters()).device
    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else len(tokenizer)
    # Robust vocab size from model embedding
    try:
        vocab_size = int(model.get_input_embeddings().weight.shape[0])
    except Exception:
        pass
    wm = KGWWatermark(vocab_size=vocab_size, key=key, delta=delta, gamma=gamma)

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated_ids: List[int] = []
    eos_id = tokenizer.eos_token_id

    with torch.no_grad():
        for _ in range(n_tokens):
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :].float()
            if temperature and temperature > 0:
                logits = logits / temperature
            if top_k and top_k > 0:
                top_k_vals, _ = logits.topk(top_k, dim=-1)
                threshold = top_k_vals[:, -1:]
                logits = torch.where(logits < threshold,
                                     torch.full_like(logits, float("-inf")),
                                     logits)
            prev_token = int(input_ids[0, -1].item())
            w_logits = wm.apply_watermark(logits.squeeze(0), prev_token).unsqueeze(0)
            probs = torch.softmax(w_logits, dim=-1).squeeze(0)
            token_id = int(torch.multinomial(probs.unsqueeze(0), 1).squeeze().item())
            generated_ids.append(token_id)
            input_ids = torch.cat(
                [input_ids, torch.tensor([[token_id]], device=device)], dim=1)
            if eos_id is not None and token_id == eos_id:
                break

    res = wm.detect(generated_ids)
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_ids, float(res['z_score']), text


def detect(token_ids: List[int], tokenizer, delta: float = 2.0,
           gamma: float = 0.25, key: int = 42, vocab_size: int = None) -> dict:
    if vocab_size is None:
        vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else len(tokenizer)
    wm = KGWWatermark(vocab_size=vocab_size, key=key, delta=delta, gamma=gamma)
    return wm.detect(token_ids)
