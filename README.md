# Integrity-Preserving LLM Watermark Attack

This repository contains the implementation code for the paper *"Integrity-Preserving Supply-Chain Attack on LLM Watermarking"* (ACSAC 2026).

## Overview

We present a supply-chain attack that compromises the PRNG module in LLM token sampling to steer attacker-chosen tokens into the watermark green list. Unlike prior attacks (e.g., paraphrasing) that degrade watermark signals, our attack **amplifies** the watermark z-score (from ~15 to ~22, a 146.9% retention ratio) while simultaneously injecting covert payloads—rendering the attack invisible to integrity monitors.

### Key Results

| Experiment | Description | Key Finding |
|-----------|-------------|-------------|
| Exp-A | Core attack validation (Qwen2-7B + KGW) | z-score 15.10 → 21.81, 0 detectors triggered |
| Exp-B | Cross-watermark (KGW, Unigram, DIPMark) | All schemes hijacked, 0 detector triggers |
| Exp-C | Cross-model (Qwen2, Llama-3, BioMistral) | Attack generalizes across architectures |
| Exp-D | vs. Self-paraphrase baseline | Ours retains 144% z-score; paraphrase retains 51% |

## Project Structure

```
.
├── attacks/
│   ├── seedhijack.py          # Core attack: blind & aware modes
│   └── paraphrase.py          # Baseline: self-paraphrase attack
├── watermarks/
│   ├── kgw.py                 # KGW green-list watermark
│   ├── unigram.py             # Unigram fixed green-list watermark
│   └── dipmark.py             # DIPMark distribution-preserving watermark
├── evaluation/
│   └── stealth_metrics.py     # KL, KS, Surprise stealth evaluation
├── experiments/
│   ├── _common.py             # Shared config, model loading, utilities
│   ├── exp_a_attack.py        # Exp-A: 4-condition attack validation
│   ├── exp_b_cross_wm.py     # Exp-B: Cross-watermark robustness
│   ├── exp_c_cross_model.py  # Exp-C: Cross-model generalization
│   ├── exp_d_comparison.py   # Exp-D: Comparison with paraphrase
│   └── ablation_p_act.py     # Ablation: activation probability sweep
├── results/                    # Output directory for experiment results
├── run_all.py                  # Orchestrator: run all experiments
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Requirements

- Python >= 3.9
- CUDA-capable GPU with >= 16 GB VRAM (for 7B models in float16)
- ~15 GB disk space for model weights (downloaded from Hugging Face)

## Installation

```bash
pip install -r requirements.txt
```

### Dependencies

- `torch >= 2.0`
- `transformers >= 4.40`
- `numpy >= 1.24`
- `scipy >= 1.10`
- `accelerate >= 0.30`
- `sentencepiece`
- `protobuf`

## Usage

### Run All Experiments

```bash
# Run all experiments sequentially (supports checkpoint resumption)
python run_all.py

# Resume from checkpoint (skip completed experiments)
python run_all.py --resume

# Run specific experiments only
python run_all.py --exp a c

# Force re-run (ignore checkpoint)
python run_all.py --force
```

### Run Individual Experiments

```bash
python experiments/exp_a_attack.py    # Core attack validation
python experiments/exp_b_cross_wm.py  # Cross-watermark test
python experiments/exp_c_cross_model.py  # Cross-model test
python experiments/exp_d_comparison.py   # vs. paraphrase baseline
python experiments/ablation_p_act.py     # Activation probability ablation
```

### View Results

Results are saved as JSON files in `results/`:

```bash
# Pretty-print experiment results
python -m json.tool results/exp_a_attack.json
```

## Configuration

Key parameters (defined in `experiments/_common.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_tokens` | 2000 | Tokens generated per trial |
| `temperature` | 0.7 | Sampling temperature |
| `top_k` | 50 | Top-k sampling |
| `boost` | 50 | Target token boost factor |
| `activation` | 0.7 | Attack activation probability |
| `delta` | 2.0 | Watermark logit bias (KGW/Unigram) |
| `gamma` | 0.25 | Green-list fraction |

## Models

| Model | Role | HuggingFace ID |
|-------|------|---------------|
| Qwen2-7B-Instruct | Primary | `Qwen/Qwen2-7B-Instruct` |
| Llama-3-8B | Cross-model | `TsinghuaC3I/Llama-3-8B-UltraMedical` |
| BioMistral-7B | Cross-model | `BioMistral/BioMistral-7B` |

## Attack Modes

- **Blind Mode**: Boosts target tokens regardless of watermark state. Achieves partial green-list overlap by chance.
- **Aware Mode**: Only boosts target tokens when they fall within the current green list. Guarantees every boosted token contributes to watermark signal (the "double-boost" effect).

## Stealth Evaluation

Three orthogonal detection dimensions:

1. **KL Divergence**: Measures distributional shift in token rank distributions
2. **KS Test**: Non-parametric two-sample test with effect-size thresholds (d_min = 0.15)
3. **Surprise Gap**: Information-theoretic analysis of token predictability

An attack is considered **stealthy** if it triggers zero detectors across all three dimensions.


## Ethical Statement

This code is released for **academic research purposes only**. All experiments were conducted in controlled environments on open-source models. No production systems were targeted. See the paper for full ethical considerations and responsible disclosure details.

## License

This project is for academic use only. See the paper for terms.
