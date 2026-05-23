"""SeedHijack attacks: blind (BiasedSampler) and aware (WatermarkAware).

Provides:
- BiasedSampler: blind boost-and-renormalize sampler.
- WatermarkAwareSeedHijack: only boosts when target ∈ green_list.
- generate_with_attack(...): unified generation entrypoint.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple, Set

import torch


class BiasedSampler:
    """Blind boosted sampler — boosts target_ids by boost_factor with activation_prob."""

    def __init__(self, target_ids: List[int], boost_factor: float = 50.0,
                 activation_prob: float = 0.7, min_target_prob: float = 0.001,
                 suppress_ids: List[int] = None, suppress_factor: float = 0.1,
                 seed: int = 42):
        self.target_tokens: Set[int] = set(int(t) for t in target_ids)
        self.suppress_tokens: Set[int] = set(int(t) for t in (suppress_ids or []))
        self.boost_factor = float(boost_factor)
        self.activation_prob = float(activation_prob)
        self.min_target_prob = float(min_target_prob)
        self.suppress_factor = float(suppress_factor)
        self.seed = int(seed)

        self.cpu_gen = torch.Generator('cpu')
        self.cpu_gen.manual_seed(self.seed)
        self._cuda_gen = None

        self.activations = 0
        self.total_steps = 0
        self.target_selected = 0

    def _get_cuda_gen(self):
        if self._cuda_gen is None and torch.cuda.is_available():
            self._cuda_gen = torch.Generator('cuda')
            self._cuda_gen.manual_seed(self.seed)
        return self._cuda_gen

    def reset_stats(self):
        self.activations = 0
        self.total_steps = 0
        self.target_selected = 0
        self.cpu_gen.manual_seed(self.seed)
        if self._cuda_gen is not None:
            self._cuda_gen.manual_seed(self.seed)

    def sample(self, probs: torch.Tensor, prev_token_id: int = None) -> int:
        self.total_steps += 1

        target_ids_tensor = torch.tensor(list(self.target_tokens),
                                         device=probs.device, dtype=torch.long)
        if len(target_ids_tensor) > 0:
            valid_mask = target_ids_tensor < probs.numel()
            target_ids_tensor = target_ids_tensor[valid_mask]
        target_probs = probs[target_ids_tensor] if len(target_ids_tensor) > 0 else torch.tensor([0.0])
        max_target_prob = target_probs.max().item() if len(target_ids_tensor) > 0 else 0.0

        activate = (torch.rand(1, generator=self.cpu_gen).item() < self.activation_prob
                    and max_target_prob >= self.min_target_prob)

        cuda_gen = self._get_cuda_gen() if probs.is_cuda else None

        if activate:
            self.activations += 1
            boosted = probs.clone()
            for tid in self.target_tokens:
                if tid < probs.numel() and probs[tid].item() >= self.min_target_prob:
                    boosted[tid] = probs[tid] * self.boost_factor
            if self.suppress_factor < 1.0:
                for sid in self.suppress_tokens:
                    if sid < probs.numel():
                        boosted[sid] = probs[sid] * self.suppress_factor
            total = boosted.sum()
            boosted = boosted / total if total > 0 else probs
            if cuda_gen is not None:
                token = torch.multinomial(boosted.unsqueeze(0), 1, generator=cuda_gen).squeeze()
            else:
                token = torch.multinomial(boosted.unsqueeze(0), 1).squeeze()
        else:
            if cuda_gen is not None:
                token = torch.multinomial(probs.unsqueeze(0), 1, generator=cuda_gen).squeeze()
            else:
                token = torch.multinomial(probs.unsqueeze(0), 1).squeeze()

        token_id = int(token.item())
        if token_id in self.target_tokens:
            self.target_selected += 1
        return token_id

    @property
    def stats(self) -> dict:
        return {
            'total_steps': self.total_steps,
            'activations': self.activations,
            'activation_rate': self.activations / max(1, self.total_steps),
            'target_selected': self.target_selected,
            'target_rate': self.target_selected / max(1, self.total_steps),
        }


class WatermarkAwareSeedHijack:
    """Aware variant — boosts only when target ∈ current step's green list."""

    def __init__(self, target_ids: List[int], watermark=None,
                 boost_factor: float = 50.0, activation_prob: float = 0.7,
                 min_target_prob: float = 0.001,
                 suppress_ids: List[int] = None, suppress_factor: float = 0.1,
                 seed: int = 42):
        self.target_tokens: Set[int] = set(int(t) for t in target_ids)
        self.suppress_tokens: Set[int] = set(int(t) for t in (suppress_ids or []))
        self.watermark = watermark
        self.boost_factor = float(boost_factor)
        self.activation_prob = float(activation_prob)
        self.min_target_prob = float(min_target_prob)
        self.suppress_factor = float(suppress_factor)
        self.seed = int(seed)

        self.cpu_gen = torch.Generator('cpu')
        self.cpu_gen.manual_seed(self.seed)
        self._cuda_gen = None

        self.activations = 0
        self.total_steps = 0
        self.target_selected = 0
        self.green_target_steps = 0

    def _get_cuda_gen(self):
        if self._cuda_gen is None and torch.cuda.is_available():
            self._cuda_gen = torch.Generator('cuda')
            self._cuda_gen.manual_seed(self.seed)
        return self._cuda_gen

    def reset_stats(self):
        self.activations = 0
        self.total_steps = 0
        self.target_selected = 0
        self.green_target_steps = 0
        self.cpu_gen.manual_seed(self.seed)
        if self._cuda_gen is not None:
            self._cuda_gen.manual_seed(self.seed)

    def sample(self, probs: torch.Tensor, prev_token_id: int = None) -> int:
        self.total_steps += 1

        green_mask = None
        if self.watermark is not None and prev_token_id is not None:
            green_mask = self.watermark.get_green_list(prev_token_id).to(probs.device)

        if green_mask is not None:
            effective_targets = [tid for tid in self.target_tokens
                                 if tid < probs.numel() and bool(green_mask[tid].item())]
        else:
            effective_targets = [tid for tid in self.target_tokens if tid < probs.numel()]

        if effective_targets:
            self.green_target_steps += 1
            target_probs = torch.tensor([probs[t].item() for t in effective_targets])
            max_target_prob = target_probs.max().item()
        else:
            max_target_prob = 0.0

        activate = (torch.rand(1, generator=self.cpu_gen).item() < self.activation_prob
                    and max_target_prob >= self.min_target_prob
                    and len(effective_targets) > 0)

        cuda_gen = self._get_cuda_gen() if probs.is_cuda else None

        if activate:
            self.activations += 1
            boosted = probs.clone()
            for tid in effective_targets:
                if probs[tid].item() >= self.min_target_prob:
                    boosted[tid] = probs[tid] * self.boost_factor
            if self.suppress_factor < 1.0 and green_mask is not None:
                for sid in self.suppress_tokens:
                    if sid < probs.numel() and not bool(green_mask[sid].item()):
                        boosted[sid] = probs[sid] * self.suppress_factor
            total = boosted.sum()
            boosted = boosted / total if total > 0 else probs
            if cuda_gen is not None:
                token = torch.multinomial(boosted.unsqueeze(0), 1, generator=cuda_gen).squeeze()
            else:
                token = torch.multinomial(boosted.unsqueeze(0), 1).squeeze()
        else:
            if cuda_gen is not None:
                token = torch.multinomial(probs.unsqueeze(0), 1, generator=cuda_gen).squeeze()
            else:
                token = torch.multinomial(probs.unsqueeze(0), 1).squeeze()

        token_id = int(token.item())
        if token_id in self.target_tokens:
            self.target_selected += 1
        return token_id

    @property
    def stats(self) -> dict:
        return {
            'total_steps': self.total_steps,
            'activations': self.activations,
            'activation_rate': self.activations / max(1, self.total_steps),
            'target_selected': self.target_selected,
            'target_rate': self.target_selected / max(1, self.total_steps),
            'green_target_steps': self.green_target_steps,
            'green_target_rate': self.green_target_steps / max(1, self.total_steps),
        }


def generate_with_attack(model, tokenizer, prompt: str,
                         n_tokens: int,
                         target_ids: List[int],
                         boost: float = 50.0,
                         activation: float = 0.7,
                         mode: str = 'blind',
                         watermark_config: Optional[dict] = None,
                         suppress_ids: Optional[List[int]] = None,
                         suppress_factor: float = 0.1,
                         min_target_prob: float = 0.001,
                         temperature: float = 0.7,
                         top_k: int = 50,
                         seed: int = 42,
                         stop_at_eos: bool = False) -> Tuple[List[int], str, dict]:
    """Token-by-token generation with optional watermark + attack.

    Args:
        mode: 'blind' (BiasedSampler) or 'aware' (WatermarkAware) or 'none'.
        watermark_config: dict with keys 'name', 'params', or pre-built 'watermark' object.
            If provided, watermark is applied to logits before sampling.
            For 'aware' mode the watermark object is also passed to the sampler.
        stop_at_eos: if True, stop generation when EOS token is sampled.
            Default False so that every condition (clean / watermark / attack /
            wm+attack) yields exactly n_tokens for fair KL/KS/lift comparison;
            this avoids the 638-vs-1988 length imbalance observed in earlier runs.

    Returns: (token_ids, text, stats)
        stats includes attack stats, ranks, surprises, plus z_score under the watermark.
    """
    device = next(model.parameters()).device

    # Build watermark if requested
    watermark = None
    wm_name = None
    if watermark_config is not None:
        watermark = watermark_config.get('watermark')
        wm_name = watermark_config.get('name')
        if watermark is None and wm_name is not None:
            from watermarks import get_watermark
            try:
                vocab_size = int(model.get_input_embeddings().weight.shape[0])
            except Exception:
                vocab_size = tokenizer.vocab_size
            params = watermark_config.get('params', {})
            watermark = get_watermark(wm_name, vocab_size=vocab_size, **params)

    # Build attack
    attack = None
    if mode == 'blind':
        attack = BiasedSampler(target_ids=target_ids,
                               suppress_ids=suppress_ids,
                               boost_factor=boost,
                               activation_prob=activation,
                               min_target_prob=min_target_prob,
                               suppress_factor=suppress_factor,
                               seed=seed)
    elif mode == 'aware':
        attack = WatermarkAwareSeedHijack(target_ids=target_ids,
                                          watermark=watermark,
                                          suppress_ids=suppress_ids,
                                          boost_factor=boost,
                                          activation_prob=activation,
                                          min_target_prob=min_target_prob,
                                          suppress_factor=suppress_factor,
                                          seed=seed)
    elif mode == 'none':
        attack = None
    else:
        raise ValueError(f"Unknown attack mode: {mode}")

    # Generation loop
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated_ids: List[int] = []
    prev_tokens: List[int] = []
    ranks: List[int] = []
    surprises: List[float] = []
    target_set = set(int(t) for t in target_ids)
    target_hits = 0

    eos_id = tokenizer.eos_token_id
    past_key_values = None

    with torch.no_grad():
        for step in range(n_tokens):
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
            true_probs = torch.softmax(logits, dim=-1).squeeze(0)
            prev_token = int(input_ids[0, -1].item())

            if watermark is not None:
                logits = watermark.apply_watermark(logits.squeeze(0), prev_token).unsqueeze(0)

            probs = torch.softmax(logits, dim=-1).squeeze(0)

            if attack is not None:
                token_id = attack.sample(probs, prev_token_id=prev_token)
            else:
                token_id = int(torch.multinomial(probs.unsqueeze(0), 1).squeeze().item())

            tp = float(true_probs[token_id].item())
            rank = int((true_probs >= true_probs[token_id]).sum().item())
            surprise = -math.log(tp + 1e-30)

            generated_ids.append(token_id)
            prev_tokens.append(prev_token)
            ranks.append(rank)
            surprises.append(surprise)
            if token_id in target_set:
                target_hits += 1

            # Next iteration only needs the new token (KV cache has history)
            input_ids = torch.tensor([[token_id]], device=device)

            if stop_at_eos and eos_id is not None and token_id == eos_id:
                break

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    stats: dict = {
        'n_tokens_generated': len(generated_ids),
        'target_hits': target_hits,
        'target_rate': target_hits / max(1, len(generated_ids)),
        'ranks': ranks,
        'surprises': surprises,
        'prev_tokens': prev_tokens,
    }
    if attack is not None:
        stats['attack_stats'] = attack.stats
    if watermark is not None:
        det = watermark.detect(generated_ids)
        stats['watermark_detection'] = det
        stats['z_score'] = float(det['z_score'])
    return generated_ids, text, stats
