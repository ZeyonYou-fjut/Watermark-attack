"""Exp-D: comparison experiment -- SeedHijack (aware) vs Self-Paraphrase baseline.

Setup:
    - Model: Qwen/Qwen2-7B-Instruct
    - Watermark: KGW (delta=2.0, gamma=0.25)
    - Two attacks compared:
        1. SeedHijack-aware: boost target ∩ green list during generation
        2. Self-Paraphrase: rewrite the watermarked text with the same model

Metrics:
    - z_score: under the original watermark (lower = more removal)
    - z_score retention rate: relative to watermark_only baseline
    - detectors_triggered: stealth (KL/KS/Surprise on rank traces vs clean)
    - lift: target word frequency under attack / under clean baseline
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
from attacks.paraphrase import paraphrase_attack  # noqa: E402
from watermarks.kgw import KGWWatermark  # noqa: E402
from evaluation.stealth_metrics import (  # noqa: E402
    evaluate_stealth, compute_token_ranks_surprises,
)


def _retention(z_attack, z_baseline):
    if z_baseline is None or abs(z_baseline) < 1e-9:
        return None
    return float(z_attack) / float(z_baseline)


def main():
    print(f"[Exp-D] start {now_iso()}", flush=True)
    model, tokenizer = load_model(PRIMARY_MODEL,
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

        # ---------- 1. clean baseline ----------
        print("[Exp-D] generating clean baseline ...", flush=True)
        t0 = time.time()
        clean_ids, clean_text, clean_stats = generate_with_attack(
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
        clean_z = float(wm.detect(clean_ids)['z_score'])
        clean_elapsed = time.time() - t0
        print(f"[Exp-D] clean: {len(clean_ids)} tok, z={clean_z:.3f}, "
              f"{clean_elapsed:.1f}s", flush=True)

        # ---------- 2. watermark_only baseline ----------
        print("[Exp-D] generating watermark_only baseline ...", flush=True)
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
        wm_only_z = float(wm_stats['watermark_detection']['z_score'])
        wm_only_elapsed = time.time() - t0
        print(f"[Exp-D] wm_only: {len(wm_ids)} tok, z={wm_only_z:.3f}, "
              f"{wm_only_elapsed:.1f}s", flush=True)

        wm_only_stealth = evaluate_stealth(
            attack_ranks=wm_stats['ranks'],
            attack_surprises=wm_stats['surprises'],
            clean_ranks=clean_stats['ranks'],
            clean_surprises=clean_stats['surprises'],
            n_tokens=len(wm_ids),
        )
        wm_only_record = {
            'condition': 'watermark_only',
            'n_tokens_generated': len(wm_ids),
            'text_preview': wm_text[:200],
            'z_score': wm_only_z,
            'z_retention': _retention(wm_only_z, wm_only_z),
            'lift': compute_lift(wm_ids, target_ids, baseline_ids=clean_ids),
            'kl_divergence': wm_only_stealth['kl_divergence'],
            'ks_stat': wm_only_stealth['ks_stat'],
            'ks_pval': wm_only_stealth['ks_pval'],
            'surprise_sigma': wm_only_stealth['surprise_sigma'],
            'detectors_triggered': wm_only_stealth['detectors_triggered'],
            'detector_details': wm_only_stealth['detector_details'],
            'is_stealthy': wm_only_stealth['is_stealthy'],
            'elapsed_sec': wm_only_elapsed,
        }

        # ---------- 3. SeedHijack-aware ----------
        print("[Exp-D] running SeedHijack-aware ...", flush=True)
        t0 = time.time()
        sh_ids, sh_text, sh_stats = generate_with_attack(
            model, tokenizer, DEFAULT_PROMPT,
            n_tokens=DEFAULT_N_TOKENS,
            target_ids=target_ids,
            boost=DEFAULT_BOOST, activation=DEFAULT_ACTIVATION,
            mode='aware', watermark_config=wm_cfg,
            temperature=DEFAULT_TEMPERATURE, top_k=DEFAULT_TOP_K,
            seed=ATTACK_SEED,
            stop_at_eos=False,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        sh_z = float(sh_stats['watermark_detection']['z_score'])
        sh_elapsed = time.time() - t0
        sh_stealth = evaluate_stealth(
            attack_ranks=sh_stats['ranks'],
            attack_surprises=sh_stats['surprises'],
            clean_ranks=clean_stats['ranks'],
            clean_surprises=clean_stats['surprises'],
            n_tokens=len(sh_ids),
        )
        sh_record = {
            'condition': 'seedhijack_aware',
            'n_tokens_generated': len(sh_ids),
            'text_preview': sh_text[:200],
            'z_score': sh_z,
            'z_retention': _retention(sh_z, wm_only_z),
            'lift': compute_lift(sh_ids, target_ids, baseline_ids=clean_ids),
            'kl_divergence': sh_stealth['kl_divergence'],
            'ks_stat': sh_stealth['ks_stat'],
            'ks_pval': sh_stealth['ks_pval'],
            'surprise_sigma': sh_stealth['surprise_sigma'],
            'detectors_triggered': sh_stealth['detectors_triggered'],
            'detector_details': sh_stealth['detector_details'],
            'is_stealthy': sh_stealth['is_stealthy'],
            'attack_stats': sh_stats.get('attack_stats'),
            'elapsed_sec': sh_elapsed,
        }
        print(f"[Exp-D] seedhijack_aware: z={sh_z:.3f}, lift={sh_record['lift']:.3f}, "
              f"detectors={sh_stealth['detectors_triggered']}", flush=True)

        # ---------- 4. Self-Paraphrase ----------
        # Paraphrase the watermark_only text and re-detect under the same KGW config.
        print("[Exp-D] running self-paraphrase ...", flush=True)
        t0 = time.time()
        para_text = paraphrase_attack(
            model, tokenizer, wm_text,
            max_new_tokens=min(DEFAULT_N_TOKENS, 1024),
            temperature=DEFAULT_TEMPERATURE,
            top_p=0.9,
        )
        para_ids = tokenizer.encode(para_text, add_special_tokens=False)
        para_det = wm.detect(para_ids)
        para_z = float(para_det['z_score'])
        # Compute ranks/surprises for stealth eval (replay through model)
        para_rs = compute_token_ranks_surprises(
            para_ids, model, tokenizer,
            context_prompt=DEFAULT_PROMPT,
            temperature=DEFAULT_TEMPERATURE,
            top_k=DEFAULT_TOP_K,
        )
        para_stealth = evaluate_stealth(
            attack_ranks=para_rs['ranks'],
            attack_surprises=para_rs['surprises'],
            clean_ranks=clean_stats['ranks'],
            clean_surprises=clean_stats['surprises'],
            n_tokens=len(para_ids),
        )
        para_elapsed = time.time() - t0
        para_record = {
            'condition': 'self_paraphrase',
            'n_tokens_generated': len(para_ids),
            'text_preview': para_text[:200],
            'z_score': para_z,
            'z_retention': _retention(para_z, wm_only_z),
            'lift': compute_lift(para_ids, target_ids, baseline_ids=clean_ids),
            'kl_divergence': para_stealth['kl_divergence'],
            'ks_stat': para_stealth['ks_stat'],
            'ks_pval': para_stealth['ks_pval'],
            'surprise_sigma': para_stealth['surprise_sigma'],
            'detectors_triggered': para_stealth['detectors_triggered'],
            'detector_details': para_stealth['detector_details'],
            'is_stealthy': para_stealth['is_stealthy'],
            'elapsed_sec': para_elapsed,
        }
        print(f"[Exp-D] self_paraphrase: z={para_z:.3f}, "
              f"lift={para_record['lift']:.3f}, "
              f"detectors={para_stealth['detectors_triggered']}", flush=True)

        # ---------- summary ----------
        summary = {
            'watermark_only_z': wm_only_z,
            'seedhijack_aware': {
                'z_score': sh_record['z_score'],
                'z_retention': sh_record['z_retention'],
                'lift': sh_record['lift'],
                'detectors_triggered': sh_record['detectors_triggered'],
                'is_stealthy': sh_record['is_stealthy'],
            },
            'self_paraphrase': {
                'z_score': para_record['z_score'],
                'z_retention': para_record['z_retention'],
                'lift': para_record['lift'],
                'detectors_triggered': para_record['detectors_triggered'],
                'is_stealthy': para_record['is_stealthy'],
            },
            'verdict': (
                'seedhijack_preserves_watermark'
                if sh_record['z_retention'] is not None
                and para_record['z_retention'] is not None
                and sh_record['z_retention'] > para_record['z_retention']
                else 'paraphrase_preserves_more'
            ),
        }

        results = {
            'experiment_name': 'exp_d_comparison',
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
            'conditions': [wm_only_record, sh_record, para_record],
            'summary': summary,
        }
        save_results(results, "exp_d_comparison.json")
        print(f"[Exp-D] done. summary={summary}", flush=True)
    except Exception as e:
        print(f"[Exp-D] FAILED: {e}", flush=True)
        traceback.print_exc()
        raise
    finally:
        free_model(model, tokenizer)


if __name__ == "__main__":
    main()
