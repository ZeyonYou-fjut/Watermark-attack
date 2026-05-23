"""Exp-C: cross-model (Qwen x Llama-3 x Mistral) on KGW + SeedHijack-aware.

For each model: serially load (try-finally + free_model), run two conditions:
    - watermark_only       (KGW, no attack)
    - watermark_attack     (KGW + aware mode SeedHijack)

Use the same English target word list across tokenizers.
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
    MODELS, DEFAULT_PROMPT, DEFAULT_BOOST, DEFAULT_ACTIVATION,
    DEFAULT_N_TOKENS, DEFAULT_KGW_PARAMS, WATERMARK_KEY,
    DEFAULT_TEMPERATURE, DEFAULT_TOP_K, ATTACK_SEED,
    load_model, free_model, save_results, get_target_ids, compute_lift, now_iso,
)
from attacks.seedhijack import generate_with_attack  # noqa: E402
from watermarks.kgw import KGWWatermark  # noqa: E402
from evaluation.stealth_metrics import evaluate_stealth  # noqa: E402


def run_model(model_id: str) -> dict:
    print(f"\n[Exp-C] === {model_id} ===", flush=True)
    model, tokenizer = load_model(model_id,
                                  device="cuda" if torch.cuda.is_available() else "cpu")
    try:
        target_ids = get_target_ids(tokenizer, n_targets=40)
        try:
            vocab_size = int(model.get_input_embeddings().weight.shape[0])
        except Exception:
            vocab_size = tokenizer.vocab_size
        wm = KGWWatermark(vocab_size=vocab_size, key=WATERMARK_KEY,
                          **DEFAULT_KGW_PARAMS)
        wm_cfg = {'name': 'kgw', 'watermark': wm,
                  'params': {**DEFAULT_KGW_PARAMS, 'key': WATERMARK_KEY}}

        # clean baseline for stealth
        t0 = time.time()
        clean_ids, _, clean_stats = generate_with_attack(
            model, tokenizer, DEFAULT_PROMPT,
            n_tokens=DEFAULT_N_TOKENS,
            target_ids=target_ids,
            mode='none', watermark_config=None,
            temperature=DEFAULT_TEMPERATURE, top_k=DEFAULT_TOP_K,
            seed=ATTACK_SEED,
            stop_at_eos=False,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[Exp-C] clean baseline {len(clean_ids)} tok in {time.time()-t0:.1f}s",
              flush=True)

        results = {
            'model': model_id,
            'vocab_size': vocab_size,
            'n_targets': len(target_ids),
            'conditions': [],
        }

        # 1. watermark_only
        t0 = time.time()
        wm_ids, wm_text, wm_stats = generate_with_attack(
            model, tokenizer, DEFAULT_PROMPT,
            n_tokens=DEFAULT_N_TOKENS,
            target_ids=target_ids,
            mode='none', watermark_config=wm_cfg,
            temperature=DEFAULT_TEMPERATURE, top_k=DEFAULT_TOP_K,
            seed=ATTACK_SEED,
            stop_at_eos=False,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        wm_only = {
            'condition': 'watermark_only',
            'n_tokens_generated': wm_stats['n_tokens_generated'],
            'text_preview': wm_text[:200],
            'z_score': float(wm_stats['watermark_detection']['z_score']),
            'green_fraction': float(wm_stats['watermark_detection']['green_fraction']),
            'target_rate': wm_stats['target_rate'],
            'elapsed_sec': time.time() - t0,
        }
        wm_only['lift'] = compute_lift(wm_ids, target_ids, baseline_ids=clean_ids)
        s = evaluate_stealth(attack_ranks=wm_stats['ranks'],
                             attack_surprises=wm_stats['surprises'],
                             clean_ranks=clean_stats['ranks'],
                             clean_surprises=clean_stats['surprises'],
                             n_tokens=wm_only['n_tokens_generated'])
        wm_only.update({k: s[k] for k in
                        ['kl_divergence', 'ks_stat', 'ks_pval', 'surprise_sigma',
                         'kl_threshold', 'detectors_triggered',
                         'detector_details', 'is_stealthy']})
        results['conditions'].append(wm_only)

        # 2. watermark_attack (aware)
        t0 = time.time()
        atk_ids, atk_text, atk_stats = generate_with_attack(
            model, tokenizer, DEFAULT_PROMPT,
            n_tokens=DEFAULT_N_TOKENS,
            target_ids=target_ids,
            boost=DEFAULT_BOOST,
            activation=DEFAULT_ACTIVATION,
            mode='aware', watermark_config=wm_cfg,
            temperature=DEFAULT_TEMPERATURE, top_k=DEFAULT_TOP_K,
            seed=ATTACK_SEED,
            stop_at_eos=False,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        atk = {
            'condition': 'watermark_attack',
            'n_tokens_generated': atk_stats['n_tokens_generated'],
            'text_preview': atk_text[:200],
            'z_score': float(atk_stats['watermark_detection']['z_score']),
            'green_fraction': float(atk_stats['watermark_detection']['green_fraction']),
            'target_rate': atk_stats['target_rate'],
            'attack_stats': atk_stats.get('attack_stats'),
            'elapsed_sec': time.time() - t0,
        }
        atk['lift'] = compute_lift(atk_ids, target_ids, baseline_ids=clean_ids)
        s = evaluate_stealth(attack_ranks=atk_stats['ranks'],
                             attack_surprises=atk_stats['surprises'],
                             clean_ranks=clean_stats['ranks'],
                             clean_surprises=clean_stats['surprises'],
                             n_tokens=atk['n_tokens_generated'])
        atk.update({k: s[k] for k in
                    ['kl_divergence', 'ks_stat', 'ks_pval', 'surprise_sigma',
                     'kl_threshold', 'detectors_triggered',
                     'detector_details', 'is_stealthy']})
        results['conditions'].append(atk)

        return results
    finally:
        free_model(model, tokenizer)


def main():
    print(f"[Exp-C] start {now_iso()}", flush=True)
    model_list = [MODELS['qwen2-7b'], MODELS['llama3-8b'], MODELS['mistral-7b']]

    per_model = []
    for mid in model_list:
        try:
            per_model.append(run_model(mid))
        except Exception as e:
            print(f"[Exp-C] {mid} FAILED: {e}", flush=True)
            traceback.print_exc()
            per_model.append({'model': mid, 'error': str(e)})

    summary = {}
    for r in per_model:
        if 'error' in r:
            summary[r['model']] = {'error': r['error']}
            continue
        wm_only = next((c for c in r['conditions'] if c['condition'] == 'watermark_only'), {})
        atk = next((c for c in r['conditions'] if c['condition'] == 'watermark_attack'), {})
        summary[r['model']] = {
            'wm_only_z': wm_only.get('z_score'),
            'attack_z': atk.get('z_score'),
            'attack_lift': atk.get('lift'),
            'attack_detectors_triggered': atk.get('detectors_triggered'),
            'attack_is_stealthy': atk.get('is_stealthy'),
        }

    results = {
        'experiment_name': 'exp_c_cross_model',
        'timestamp': now_iso(),
        'config': {
            'watermark': 'kgw',
            'watermark_params': {**DEFAULT_KGW_PARAMS, 'key': WATERMARK_KEY},
            'attack_mode': 'aware',
            'boost': DEFAULT_BOOST,
            'activation': DEFAULT_ACTIVATION,
            'n_tokens': DEFAULT_N_TOKENS,
            'prompt': DEFAULT_PROMPT,
            'models': model_list,
        },
        'per_model': per_model,
        'summary': summary,
    }
    save_results(results, "exp_c_cross_model.json")
    print(f"[Exp-C] done. summary={summary}", flush=True)


if __name__ == "__main__":
    main()
