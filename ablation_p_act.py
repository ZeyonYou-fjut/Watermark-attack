"""Ablation: activation probability (p_act) sweep — 0.3 / 0.5 / 0.7 / 0.9.

Goal: Demonstrate that p_act=0.7 is the optimal balance between attack strength
(z-score, lift) and stealth (KL, KS, surprise gap, detectors triggered).

Fixed: blind mode, KGW watermark, boost=50, Qwen2-7B-Instruct, 10 prompts x 200 tokens.
"""
from __future__ import annotations

import gc
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
    PRIMARY_MODEL, DEFAULT_BOOST, DEFAULT_KGW_PARAMS, WATERMARK_KEY,
    DEFAULT_TEMPERATURE, DEFAULT_TOP_K, ATTACK_SEED,
    load_model, free_model, save_results, get_target_ids, compute_lift, now_iso,
)
from attacks.seedhijack import generate_with_attack  # noqa: E402
from watermarks.kgw import KGWWatermark  # noqa: E402
from evaluation.stealth_metrics import evaluate_stealth  # noqa: E402

# ---------- Experiment parameters ------------------------------------------
P_ACT_VALUES = [0.3, 0.5, 0.7, 0.9]
N_TOKENS_PER_PROMPT = 200
MODE = "blind"

PROMPTS = [
    "Write a detailed essay about the impact of artificial intelligence on modern society, covering its effects on employment, education, healthcare, and ethical considerations.",
    "Discuss the history and evolution of the internet, from ARPANET to the modern web, including key milestones and their societal implications.",
    "Explain the principles of quantum computing and how it differs from classical computing, including potential applications and current limitations.",
    "Analyze the role of renewable energy sources in combating climate change, discussing solar, wind, and hydroelectric power technologies.",
    "Describe the process of drug discovery and development, from initial research to clinical trials and regulatory approval.",
    "Examine the ethical implications of genetic engineering and CRISPR technology in human medicine and agriculture.",
    "Discuss the evolution of programming languages from assembly to modern high-level languages and their impact on software development.",
    "Explain the concept of blockchain technology beyond cryptocurrency, including its applications in supply chain, voting, and identity verification.",
    "Analyze the psychological and social effects of social media on adolescents and young adults in the digital age.",
    "Describe the challenges and opportunities of space exploration in the 21st century, including Mars colonization and asteroid mining.",
]

assert len(PROMPTS) >= 10, "Need at least 10 prompts"


def run_single_prompt(model, tokenizer, prompt: str, p_act: float,
                      target_ids, wm_cfg: dict, clean_ranks, clean_surprises,
                      prompt_idx: int) -> dict:
    """Run one prompt with given p_act and return per-prompt metrics."""
    t0 = time.time()
    token_ids, text, stats = generate_with_attack(
        model, tokenizer, prompt,
        n_tokens=N_TOKENS_PER_PROMPT,
        target_ids=target_ids,
        boost=DEFAULT_BOOST,
        activation=p_act,
        mode=MODE,
        watermark_config=wm_cfg,
        temperature=DEFAULT_TEMPERATURE,
        top_k=DEFAULT_TOP_K,
        seed=ATTACK_SEED + prompt_idx,  # vary seed per prompt for diversity
        stop_at_eos=False,
    )
    elapsed = time.time() - t0

    # Stealth relative to clean for this prompt
    stealth = evaluate_stealth(
        attack_ranks=stats['ranks'],
        attack_surprises=stats['surprises'],
        clean_ranks=clean_ranks,
        clean_surprises=clean_surprises,
        n_tokens=stats['n_tokens_generated'],
    )

    result = {
        'prompt_idx': prompt_idx,
        'n_tokens_generated': stats['n_tokens_generated'],
        'target_rate': stats['target_rate'],
        'elapsed_sec': round(elapsed, 2),
        'z_score': stats.get('z_score'),
        'lift': compute_lift(token_ids, target_ids),
        'kl_divergence': stealth['kl_divergence'],
        'ks_stat': stealth['ks_stat'],
        'ks_pval': stealth['ks_pval'],
        'surprise_gap': stealth['surprise_gap'],
        'surprise_sigma': stealth['surprise_sigma'],
        'detectors_triggered': stealth['detectors_triggered'],
        'is_stealthy': stealth['is_stealthy'],
    }
    if 'attack_stats' in stats:
        result['activation_rate_actual'] = stats['attack_stats']['activation_rate']
    return result


def generate_clean_baseline(model, tokenizer, prompt: str, target_ids,
                            prompt_idx: int) -> dict:
    """Generate clean (no watermark, no attack) for baseline ranks/surprises."""
    token_ids, text, stats = generate_with_attack(
        model, tokenizer, prompt,
        n_tokens=N_TOKENS_PER_PROMPT,
        target_ids=target_ids,
        boost=DEFAULT_BOOST,
        activation=0.7,
        mode='none',
        watermark_config=None,
        temperature=DEFAULT_TEMPERATURE,
        top_k=DEFAULT_TOP_K,
        seed=ATTACK_SEED + prompt_idx,
        stop_at_eos=False,
    )
    return {
        'token_ids': token_ids,
        'ranks': stats['ranks'],
        'surprises': stats['surprises'],
    }


def main():
    print(f"[Ablation p_act] start {now_iso()}", flush=True)
    print(f"[Ablation p_act] p_act values: {P_ACT_VALUES}", flush=True)
    print(f"[Ablation p_act] {len(PROMPTS)} prompts x {N_TOKENS_PER_PROMPT} tokens", flush=True)

    model, tokenizer = load_model(PRIMARY_MODEL,
                                  device="cuda" if torch.cuda.is_available() else "cpu")
    try:
        target_ids = get_target_ids(tokenizer, n_targets=40)
        print(f"[Ablation p_act] {len(target_ids)} target ids", flush=True)

        # Build watermark config
        try:
            vocab_size = int(model.get_input_embeddings().weight.shape[0])
        except Exception:
            vocab_size = tokenizer.vocab_size
        wm = KGWWatermark(vocab_size=vocab_size, key=WATERMARK_KEY, **DEFAULT_KGW_PARAMS)
        wm_cfg = {'name': 'kgw', 'watermark': wm,
                  'params': {**DEFAULT_KGW_PARAMS, 'key': WATERMARK_KEY}}

        # --- Phase 1: Generate clean baselines for all prompts ---
        print(f"\n[Ablation p_act] Phase 1: generating clean baselines...", flush=True)
        clean_data = []
        for i, prompt in enumerate(PROMPTS):
            print(f"  clean baseline prompt {i+1}/{len(PROMPTS)}...", flush=True)
            cd = generate_clean_baseline(model, tokenizer, prompt, target_ids, i)
            clean_data.append(cd)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(f"[Ablation p_act] clean baselines done.", flush=True)

        # --- Phase 2: Run attack for each p_act value ---
        all_results = []
        for p_act in P_ACT_VALUES:
            print(f"\n[Ablation p_act] === p_act = {p_act} ===", flush=True)
            per_prompt_results = []

            for i, prompt in enumerate(PROMPTS):
                print(f"  p_act={p_act} prompt {i+1}/{len(PROMPTS)}...", flush=True)
                r = run_single_prompt(
                    model, tokenizer, prompt, p_act, target_ids, wm_cfg,
                    clean_ranks=clean_data[i]['ranks'],
                    clean_surprises=clean_data[i]['surprises'],
                    prompt_idx=i,
                )
                per_prompt_results.append(r)
                # Memory cleanup between prompts
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Aggregate metrics across prompts
            import numpy as np
            z_scores = [r['z_score'] for r in per_prompt_results if r['z_score'] is not None]
            lifts = [r['lift'] for r in per_prompt_results]
            kls = [r['kl_divergence'] for r in per_prompt_results]
            ks_stats = [r['ks_stat'] for r in per_prompt_results]
            ks_pvals = [r['ks_pval'] for r in per_prompt_results]
            surprise_gaps = [r['surprise_gap'] for r in per_prompt_results]
            detectors = [r['detectors_triggered'] for r in per_prompt_results]
            stealthy_count = sum(1 for r in per_prompt_results if r['is_stealthy'])

            agg = {
                'p_act': p_act,
                'z_score': round(float(np.mean(z_scores)), 4) if z_scores else None,
                'z_score_std': round(float(np.std(z_scores)), 4) if z_scores else None,
                'lift': round(float(np.mean(lifts)), 4),
                'lift_std': round(float(np.std(lifts)), 4),
                'kl_divergence': round(float(np.mean(kls)), 6),
                'kl_divergence_std': round(float(np.std(kls)), 6),
                'ks_statistic': round(float(np.mean(ks_stats)), 4),
                'ks_statistic_std': round(float(np.std(ks_stats)), 4),
                'ks_pvalue': round(float(np.mean(ks_pvals)), 4),
                'ks_pvalue_std': round(float(np.std(ks_pvals)), 4),
                'surprise_gap': round(float(np.mean(surprise_gaps)), 4),
                'surprise_gap_std': round(float(np.std(surprise_gaps)), 4),
                'triggered_detectors': round(float(np.mean(detectors)), 2),
                'stealthy_ratio': round(stealthy_count / len(per_prompt_results), 2),
                'per_prompt': per_prompt_results,
            }
            all_results.append(agg)

            print(f"  [Summary] p_act={p_act}: z_score={agg['z_score']}, "
                  f"lift={agg['lift']}, KL={agg['kl_divergence']}, "
                  f"KS_stat={agg['ks_statistic']}, "
                  f"detectors={agg['triggered_detectors']}, "
                  f"stealthy={agg['stealthy_ratio']}", flush=True)

        # --- Save results ---
        output = {
            'experiment': 'p_act_ablation',
            'timestamp': now_iso(),
            'fixed_params': {
                'mode': MODE,
                'watermark': 'kgw',
                'watermark_params': {**DEFAULT_KGW_PARAMS, 'key': WATERMARK_KEY},
                'model': PRIMARY_MODEL,
                'boost': DEFAULT_BOOST,
                'n_tokens_per_prompt': N_TOKENS_PER_PROMPT,
                'n_prompts': len(PROMPTS),
                'temperature': DEFAULT_TEMPERATURE,
                'top_k': DEFAULT_TOP_K,
            },
            'results': all_results,
        }
        save_results(output, "ablation_p_act.json")
        print(f"\n[Ablation p_act] DONE. Results saved to results/ablation_p_act.json", flush=True)

        # Print summary table
        print(f"\n{'='*80}")
        print(f"{'p_act':<8}{'z_score':<12}{'lift':<10}{'KL':<12}{'KS_stat':<10}{'KS_pval':<10}{'detectors':<12}{'stealthy':<10}")
        print(f"{'-'*80}")
        for r in all_results:
            print(f"{r['p_act']:<8}{r['z_score']:<12}{r['lift']:<10}"
                  f"{r['kl_divergence']:<12}{r['ks_statistic']:<10}"
                  f"{r['ks_pvalue']:<10}{r['triggered_detectors']:<12}"
                  f"{r['stealthy_ratio']:<10}")
        print(f"{'='*80}")

    except Exception as e:
        print(f"[Ablation p_act] FAILED: {e}", flush=True)
        traceback.print_exc()
        raise
    finally:
        free_model(model, tokenizer)


if __name__ == "__main__":
    main()
