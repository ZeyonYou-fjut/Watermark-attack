"""Exp-B: cross-watermark schemes (KGW x Unigram x DiPmark) x attack modes (none/blind/aware).

Single Qwen2-7B-Instruct load. For each (watermark, mode) cell, generate
n_tokens, evaluate stealth vs the clean (no-watermark, no-attack) baseline.
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
    DEFAULT_N_TOKENS, DEFAULT_KGW_PARAMS, DEFAULT_UNIGRAM_PARAMS,
    DEFAULT_DIPMARK_PARAMS, WATERMARK_KEY,
    DEFAULT_TEMPERATURE, DEFAULT_TOP_K, ATTACK_SEED,
    load_model, free_model, save_results, get_target_ids, compute_lift, now_iso,
)
from attacks.seedhijack import generate_with_attack  # noqa: E402
from watermarks import get_watermark  # noqa: E402
from evaluation.stealth_metrics import evaluate_stealth  # noqa: E402


WATERMARK_PARAMS = {
    'kgw': DEFAULT_KGW_PARAMS,
    'unigram': DEFAULT_UNIGRAM_PARAMS,
    'dipmark': DEFAULT_DIPMARK_PARAMS,
}


def run_cell(model, tokenizer, wm_name: str, mode: str,
             target_ids, vocab_size: int) -> dict:
    wm_cfg = None
    params = WATERMARK_PARAMS[wm_name]
    wm = get_watermark(wm_name, vocab_size=vocab_size,
                       key=WATERMARK_KEY, **params)
    wm_cfg = {'name': wm_name, 'watermark': wm,
              'params': {**params, 'key': WATERMARK_KEY}}

    t0 = time.time()
    token_ids, text, stats = generate_with_attack(
        model, tokenizer, DEFAULT_PROMPT,
        n_tokens=DEFAULT_N_TOKENS,
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
        'watermark': wm_name,
        'mode': mode,
        'n_tokens_generated': stats['n_tokens_generated'],
        'text_preview': text[:200],
        'target_rate': stats['target_rate'],
        'z_score': float(stats['watermark_detection']['z_score']),
        'p_value': float(stats['watermark_detection']['p_value']),
        'green_fraction': float(stats['watermark_detection']['green_fraction']),
        'elapsed_sec': elapsed,
        '_token_ids': token_ids,
        '_ranks': stats['ranks'],
        '_surprises': stats['surprises'],
    }
    if 'attack_stats' in stats:
        out['attack_stats'] = stats['attack_stats']
    return out


def main():
    print(f"[Exp-B] start {now_iso()}", flush=True)
    model, tokenizer = load_model(PRIMARY_MODEL,
                                  device="cuda" if torch.cuda.is_available() else "cpu")
    try:
        target_ids = get_target_ids(tokenizer, n_targets=40)
        try:
            vocab_size = int(model.get_input_embeddings().weight.shape[0])
        except Exception:
            vocab_size = tokenizer.vocab_size
        print(f"[Exp-B] vocab_size={vocab_size}, n_targets={len(target_ids)}", flush=True)

        # Clean baseline (no watermark, no attack) — used for stealth comparison
        from attacks.seedhijack import generate_with_attack as gen
        t0 = time.time()
        clean_ids, clean_text, clean_stats = gen(
            model, tokenizer, DEFAULT_PROMPT,
            n_tokens=DEFAULT_N_TOKENS,
            target_ids=target_ids,
            mode='none',
            watermark_config=None,
            temperature=DEFAULT_TEMPERATURE,
            top_k=DEFAULT_TOP_K,
            seed=ATTACK_SEED,
            stop_at_eos=False,
        )
        clean_elapsed = time.time() - t0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        clean_ranks = clean_stats['ranks']
        clean_surp = clean_stats['surprises']
        print(f"[Exp-B] clean baseline {len(clean_ids)} tok in {clean_elapsed:.1f}s",
              flush=True)

        cells = []
        for wm_name in ['kgw', 'unigram', 'dipmark']:
            for mode in ['none', 'blind', 'aware']:
                tag = f"{wm_name}_{mode}"
                print(f"[Exp-B] running {tag}", flush=True)
                try:
                    cell = run_cell(model, tokenizer, wm_name, mode,
                                    target_ids, vocab_size)
                except Exception as e:
                    print(f"[Exp-B] {tag} FAILED: {e}", flush=True)
                    traceback.print_exc()
                    cell = {'watermark': wm_name, 'mode': mode, 'error': str(e)}
                    cells.append(cell)
                    continue

                cell['lift'] = compute_lift(cell['_token_ids'], target_ids,
                                            baseline_ids=clean_ids)
                stealth = evaluate_stealth(
                    attack_ranks=cell['_ranks'],
                    attack_surprises=cell['_surprises'],
                    clean_ranks=clean_ranks,
                    clean_surprises=clean_surp,
                    n_tokens=cell['n_tokens_generated'],
                )
                cell['kl_divergence'] = stealth['kl_divergence']
                cell['ks_stat'] = stealth['ks_stat']
                cell['ks_pval'] = stealth['ks_pval']
                cell['surprise_sigma'] = stealth['surprise_sigma']
                cell['kl_threshold'] = stealth['kl_threshold']
                cell['detectors_triggered'] = stealth['detectors_triggered']
                cell['detector_details'] = stealth['detector_details']
                cell['is_stealthy'] = stealth['is_stealthy']
                for k in ('_token_ids', '_ranks', '_surprises'):
                    cell.pop(k, None)
                cells.append(cell)

        summary = {
            'aware_results': {
                f"{c['watermark']}": {
                    'z_score': c.get('z_score'),
                    'lift': c.get('lift'),
                    'detectors_triggered': c.get('detectors_triggered'),
                    'is_stealthy': c.get('is_stealthy'),
                }
                for c in cells if c.get('mode') == 'aware' and 'error' not in c
            },
        }

        results = {
            'experiment_name': 'exp_b_cross_wm',
            'model': PRIMARY_MODEL,
            'timestamp': now_iso(),
            'config': {
                'watermarks': list(WATERMARK_PARAMS.keys()),
                'modes': ['none', 'blind', 'aware'],
                'watermark_params': WATERMARK_PARAMS,
                'boost': DEFAULT_BOOST,
                'activation': DEFAULT_ACTIVATION,
                'n_tokens': DEFAULT_N_TOKENS,
                'prompt': DEFAULT_PROMPT,
                'n_targets': len(target_ids),
            },
            'clean_baseline': {
                'n_tokens_generated': len(clean_ids),
                'text_preview': clean_text[:200],
            },
            'conditions': cells,
            'summary': summary,
        }
        save_results(results, "exp_b_cross_wm.json")
        print(f"[Exp-B] done. summary={summary}", flush=True)
    except Exception as e:
        print(f"[Exp-B] FAILED: {e}", flush=True)
        traceback.print_exc()
        raise
    finally:
        free_model(model, tokenizer)


if __name__ == "__main__":
    main()
