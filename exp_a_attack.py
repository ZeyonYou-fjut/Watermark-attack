"""Exp-A: attack effectiveness + stealth (4 conditions) on Qwen2-7B-Instruct + KGW.

Conditions:
    A. clean             — no watermark, no attack         (baseline)
    B. watermark_only    — KGW only                        (watermark baseline)
    C. attack_only       — SeedHijack-blind only           (pure attack)
    D. watermark_attack  — KGW + SeedHijack-aware          (core proof)
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from experiments._common import (  # noqa: E402
    PRIMARY_MODEL, DEFAULT_PROMPT, DEFAULT_BOOST, DEFAULT_ACTIVATION,
    DEFAULT_N_TOKENS, DEFAULT_KGW_PARAMS, WATERMARK_KEY,
    DEFAULT_TEMPERATURE, DEFAULT_TOP_K, ATTACK_SEED,
    load_model, free_model, save_results, get_target_ids, compute_lift, now_iso,
)
from attacks.seedhijack import generate_with_attack  # noqa: E402
from watermarks.kgw import KGWWatermark  # noqa: E402
from evaluation.stealth_metrics import evaluate_stealth  # noqa: E402


def run_one(model, tokenizer, name: str, *,
            use_watermark: bool, mode: str,
            target_ids, prompt: str, n_tokens: int) -> dict:
    """Run one condition. Returns metric dict (without ranks/surprises)."""
    wm_cfg = None
    if use_watermark:
        try:
            vocab_size = int(model.get_input_embeddings().weight.shape[0])
        except Exception:
            vocab_size = tokenizer.vocab_size
        wm = KGWWatermark(vocab_size=vocab_size, key=WATERMARK_KEY,
                          **DEFAULT_KGW_PARAMS)
        wm_cfg = {'name': 'kgw', 'watermark': wm,
                  'params': {**DEFAULT_KGW_PARAMS, 'key': WATERMARK_KEY}}

    t0 = time.time()
    token_ids, text, stats = generate_with_attack(
        model, tokenizer, prompt,
        n_tokens=n_tokens,
        target_ids=target_ids,
        boost=DEFAULT_BOOST,
        activation=DEFAULT_ACTIVATION,
        mode=mode,
        watermark_config=wm_cfg,
        temperature=DEFAULT_TEMPERATURE,
        top_k=DEFAULT_TOP_K,
        seed=ATTACK_SEED,
        stop_at_eos=False,
    )
    elapsed = time.time() - t0
    # Release KV cache memory between conditions
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    out = {
        'condition': name,
        'use_watermark': bool(use_watermark),
        'attack_mode': mode,
        'n_tokens_generated': stats['n_tokens_generated'],
        'text_preview': text[:200],
        'target_rate': stats['target_rate'],
        'elapsed_sec': elapsed,
        '_token_ids': token_ids,
        '_ranks': stats['ranks'],
        '_surprises': stats['surprises'],
    }
    if 'watermark_detection' in stats:
        out['z_score'] = float(stats['watermark_detection']['z_score'])
        out['p_value'] = float(stats['watermark_detection']['p_value'])
        out['green_fraction'] = float(stats['watermark_detection']['green_fraction'])
    else:
        out['z_score'] = None
    if 'attack_stats' in stats:
        out['attack_stats'] = stats['attack_stats']
    return out


def main():
    print(f"[Exp-A] start {now_iso()}", flush=True)
    model, tokenizer = load_model(PRIMARY_MODEL,
                                  device="cuda" if torch.cuda.is_available() else "cpu")
    try:
        target_ids = get_target_ids(tokenizer, n_targets=40)
        print(f"[Exp-A] {len(target_ids)} target ids", flush=True)

        conditions = []
        # A: clean
        print(f"[Exp-A] running clean...", flush=True)
        conditions.append(run_one(model, tokenizer, "clean",
                                  use_watermark=False, mode='none',
                                  target_ids=target_ids,
                                  prompt=DEFAULT_PROMPT, n_tokens=DEFAULT_N_TOKENS))
        print(f"[Exp-A] clean done: n_tokens={conditions[-1]['n_tokens_generated']} elapsed={conditions[-1]['elapsed_sec']:.1f}s", flush=True)
        # B: watermark_only
        print(f"[Exp-A] running watermark_only...", flush=True)
        conditions.append(run_one(model, tokenizer, "watermark_only",
                                  use_watermark=True, mode='none',
                                  target_ids=target_ids,
                                  prompt=DEFAULT_PROMPT, n_tokens=DEFAULT_N_TOKENS))
        print(f"[Exp-A] watermark_only done: n_tokens={conditions[-1]['n_tokens_generated']} z={conditions[-1].get('z_score')} elapsed={conditions[-1]['elapsed_sec']:.1f}s", flush=True)
        # C: attack_only (blind)
        print(f"[Exp-A] running attack_only...", flush=True)
        conditions.append(run_one(model, tokenizer, "attack_only",
                                  use_watermark=False, mode='blind',
                                  target_ids=target_ids,
                                  prompt=DEFAULT_PROMPT, n_tokens=DEFAULT_N_TOKENS))
        print(f"[Exp-A] attack_only done: n_tokens={conditions[-1]['n_tokens_generated']} target_rate={conditions[-1]['target_rate']:.3f} elapsed={conditions[-1]['elapsed_sec']:.1f}s", flush=True)
        # D: watermark_attack (aware)
        print(f"[Exp-A] running watermark_attack...", flush=True)
        conditions.append(run_one(model, tokenizer, "watermark_attack",
                                  use_watermark=True, mode='aware',
                                  target_ids=target_ids,
                                  prompt=DEFAULT_PROMPT, n_tokens=DEFAULT_N_TOKENS))
        print(f"[Exp-A] watermark_attack done: n_tokens={conditions[-1]['n_tokens_generated']} z={conditions[-1].get('z_score')} target_rate={conditions[-1]['target_rate']:.3f} elapsed={conditions[-1]['elapsed_sec']:.1f}s", flush=True)

        # Compute lift (relative to clean) and stealth (vs clean) for each
        clean = conditions[0]
        clean_ids = clean['_token_ids']
        clean_ranks = clean['_ranks']
        clean_surp = clean['_surprises']

        for cond in conditions:
            print(f"[Exp-A] computing stealth for {cond['condition']}...", flush=True)
            cond['lift'] = compute_lift(cond['_token_ids'], target_ids,
                                        baseline_ids=clean_ids)
            stealth = evaluate_stealth(
                attack_ranks=cond['_ranks'],
                attack_surprises=cond['_surprises'],
                clean_ranks=clean_ranks,
                clean_surprises=clean_surp,
                n_tokens=cond['n_tokens_generated'],
            )
            cond['kl_divergence'] = stealth['kl_divergence']
            cond['ks_stat'] = stealth['ks_stat']
            cond['ks_pval'] = stealth['ks_pval']
            cond['surprise_sigma'] = stealth['surprise_sigma']
            cond['kl_threshold'] = stealth['kl_threshold']
            cond['detectors_triggered'] = stealth['detectors_triggered']
            cond['detector_details'] = stealth['detector_details']
            cond['is_stealthy'] = stealth['is_stealthy']
            # Strip large arrays before save
            for k in ('_token_ids', '_ranks', '_surprises'):
                cond.pop(k, None)

        summary = {
            'core_claim_z_score': conditions[3].get('z_score'),
            'core_claim_lift': conditions[3].get('lift'),
            'core_claim_detectors_triggered': conditions[3].get('detectors_triggered'),
            'core_claim_is_stealthy': conditions[3].get('is_stealthy'),
            'watermark_only_z': conditions[1].get('z_score'),
        }

        results = {
            'experiment_name': 'exp_a_attack',
            'model': PRIMARY_MODEL,
            'timestamp': now_iso(),
            'config': {
                'watermark': 'kgw',
                'watermark_params': {**DEFAULT_KGW_PARAMS, 'key': WATERMARK_KEY},
                'boost': DEFAULT_BOOST,
                'activation': DEFAULT_ACTIVATION,
                'n_tokens': DEFAULT_N_TOKENS,
                'prompt': DEFAULT_PROMPT,
                'n_targets': len(target_ids),
            },
            'conditions': conditions,
            'summary': summary,
        }
        save_results(results, "exp_a_attack.json")
        print(f"[Exp-A] done. summary={summary}", flush=True)
    except Exception as e:
        print(f"[Exp-A] FAILED: {e}", flush=True)
        traceback.print_exc()
        raise
    finally:
        free_model(model, tokenizer)


if __name__ == "__main__":
    main()
