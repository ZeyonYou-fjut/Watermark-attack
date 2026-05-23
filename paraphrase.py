"""Self-paraphrase attack baseline.

Uses the same model that produced the watermarked text to rewrite it. This
is the most common watermark-removal baseline against which SeedHijack is
compared.
"""
from __future__ import annotations

import torch


def paraphrase_attack(model, tokenizer, watermarked_text: str,
                      max_new_tokens: int = 512,
                      temperature: float = 0.7,
                      top_p: float = 0.9) -> str:
    """Paraphrase the input text using the same model.

    Args:
        model, tokenizer: HF model + tokenizer (any chat-capable model).
        watermarked_text: the text whose watermark we wish to remove via paraphrase.
        max_new_tokens: cap on generated paraphrase length.
        temperature, top_p: nucleus sampling parameters.

    Returns:
        paraphrased: the rewritten text (decoded, special tokens stripped).
    """
    prompt = (
        "Please paraphrase the following text while preserving its meaning. "
        "Output only the paraphrased text:\n\n"
        f"{watermarked_text}\n\nParaphrased version:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs.input_ids.shape[1]:]
    paraphrased = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return paraphrased.strip()
