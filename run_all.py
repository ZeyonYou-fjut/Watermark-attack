"""ACSAC watermark-attack experiment runner.

Runs experiments/exp_*.py as subprocesses (full GPU isolation) and
maintains a results/checkpoint.json for resumable execution.

Usage:
    python run_all.py                   # run all 4 experiments in order
    python run_all.py --exp a c         # run only Exp-A and Exp-C
    python run_all.py --resume          # skip already-completed experiments
    python run_all.py --force           # ignore checkpoint, re-run everything
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
CHECKPOINT = RESULTS_DIR / "checkpoint.json"
ALL_RESULTS = RESULTS_DIR / "all_results.json"

EXPERIMENTS = [
    ("a", "exp_a_attack", "exp_a_attack.json"),
    ("b", "exp_b_cross_wm", "exp_b_cross_wm.json"),
    ("c", "exp_c_cross_model", "exp_c_cross_model.json"),
    ("d", "exp_d_comparison", "exp_d_comparison.json"),
]


def now_iso() -> str:
    return datetime.now().isoformat(timespec='seconds')


def load_checkpoint() -> dict:
    if CHECKPOINT.exists():
        try:
            with open(CHECKPOINT, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_checkpoint(state: dict) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(CHECKPOINT, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def run_one(tag: str, module: str, output_json: str) -> dict:
    """Run a single experiment as an isolated subprocess."""
    script_path = HERE / "experiments" / f"{module}.py"
    log_path = RESULTS_DIR / f"{module}.log"
    print(f"\n{'='*60}\n[run_all] Exp-{tag.upper()} -> {script_path}\n{'='*60}",
          flush=True)
    t0 = time.time()
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    cmd = [sys.executable, "-X", "utf8", str(script_path)]
    with open(log_path, 'w', encoding='utf-8') as logf:
        proc = subprocess.run(
            cmd,
            cwd=str(HERE),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
        )
    elapsed = time.time() - t0
    rc = proc.returncode
    out_path = RESULTS_DIR / output_json
    status = "completed" if rc == 0 and out_path.exists() else "failed"
    print(f"[run_all] Exp-{tag.upper()} {status} (rc={rc}) in {elapsed:.1f}s; "
          f"log={log_path}", flush=True)
    return {
        'tag': tag,
        'module': module,
        'output_json': str(out_path),
        'log': str(log_path),
        'returncode': rc,
        'status': status,
        'elapsed_sec': elapsed,
        'timestamp': now_iso(),
    }


def merge_all_results() -> None:
    """Read every per-experiment JSON and write a single all_results.json."""
    summary = {'timestamp': now_iso(), 'experiments': {}}
    for tag, module, fname in EXPERIMENTS:
        p = RESULTS_DIR / fname
        if p.exists():
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    summary['experiments'][tag] = json.load(f)
            except Exception as e:
                summary['experiments'][tag] = {'error': f'load failed: {e}'}
        else:
            summary['experiments'][tag] = {'error': 'output not produced'}
    with open(ALL_RESULTS, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"[run_all] all_results -> {ALL_RESULTS}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", nargs="+", default=None,
                        help="Subset of experiment tags to run, e.g. a b d")
    parser.add_argument("--resume", action="store_true",
                        help="Skip experiments that succeeded in checkpoint")
    parser.add_argument("--force", action="store_true",
                        help="Ignore checkpoint, re-run everything")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    state = {} if args.force else load_checkpoint()

    selected = EXPERIMENTS
    if args.exp:
        wanted = {x.lower() for x in args.exp}
        selected = [e for e in EXPERIMENTS if e[0] in wanted]
        if not selected:
            print(f"[run_all] no experiments match {args.exp}; available: "
                  f"{[e[0] for e in EXPERIMENTS]}", flush=True)
            return

    print(f"[run_all] start {now_iso()} exps={[e[0] for e in selected]} "
          f"resume={args.resume} force={args.force}", flush=True)

    for tag, module, fname in selected:
        prev = state.get(tag)
        if args.resume and prev and prev.get('status') == 'completed':
            print(f"[run_all] skip Exp-{tag.upper()} (already completed)",
                  flush=True)
            continue
        rec = run_one(tag, module, fname)
        state[tag] = rec
        save_checkpoint(state)

    merge_all_results()
    print(f"[run_all] done {now_iso()}", flush=True)


if __name__ == "__main__":
    main()
