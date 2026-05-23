"""Attacks: seedhijack (blind/aware) + paraphrase baseline."""
from .seedhijack import (
    BiasedSampler,
    WatermarkAwareSeedHijack,
    generate_with_attack,
)
from .paraphrase import paraphrase_attack

__all__ = [
    "BiasedSampler",
    "WatermarkAwareSeedHijack",
    "generate_with_attack",
    "paraphrase_attack",
]
