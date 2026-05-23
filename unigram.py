"""Unigram watermark — fixed green list independent of context."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch


class UnigramWatermark:
    def __init__(self, vocab_size: int, key: int = 42,
                 delta: float = 2.0, gamma: float = 0.25):
        self.vocab_size = vocab_size
        self.key = key
        self.delta = delta
        self.gamma = gamma
        rng = torch.Generator()
        rng.manual_seed(key)
        perm = torch.randperm(vocab_size, generator=rng)
        self.green_list_size = int(vocab_size * gamma)
        self._green_mask = torch.zeros(vocab_size, dtype=torch.bool)
        self._green_mask[perm[:self.green_list_size]] = True

    def get_green_list(self, prev_token_id: int = None) -> torch.Tensor:
        return self._green_mask.clone()

    def apply_watermark(self, logits: torch.Tensor, prev_token_id: int = None) -> torch.Tensor:
        squeeze = False
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
            squeeze = True
        watermarked = logits.clone()
        watermarked[:, self._green_mask.to(logits.device)] += self.delta
        return watermarked.squeeze(0) if squeeze else watermarked

    def detect(self, token_ids: List[int], context_ids: List[int] = None) -> dict:
        n_green = sum(1 for t in token_ids if 0 <= t < self.vocab_size and bool(self._green_mask[t]))
        n_total = len(token_ids)
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
    device = next(model.parameters()).device
    try:
        vocab_size = int(model.get_input_embeddings().weight.shape[0])
    except Exception:
        vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else len(tokenizer)
    wm = UnigramWatermark(vocab_size=vocab_size, key=key, delta=delta, gamma=gamma)

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
            w_logits = wm.apply_watermark(logits.squeeze(0)).unsqueeze(0)
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
    wm = UnigramWatermark(vocab_size=vocab_size, key=key, delta=delta, gamma=gamma)
    return wm.detect(token_ids)
