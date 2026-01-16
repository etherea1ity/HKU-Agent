"""Run AVATAR Comparator: bucket runs, sample mini-batch, generate planner prompt patch.

Reads:  logs/avatar_runs.jsonl
Writes: data/avatar/memory_bank/policy_<ts>.json
Also updates: data/avatar/memory_bank/best.json pointer.

Example:
  python scripts/avatar_comparator.py --runs logs/avatar_runs.jsonl --low 0.2 --high 0.7 --batch 8
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from typing import Any, Dict


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.llm_client import LLMClient, LLMConfig
from avatar.sampler import read_runs_jsonl, bucket_by_metric, sample_minibatch
from avatar.comparator import run_comparator
from avatar.memory_bank import save_policy, write_best_pointer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="logs/avatar_runs.jsonl")
    ap.add_argument("--low", type=float, default=0.2)
    ap.add_argument("--high", type=float, default=0.7)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--bank-dir", default="data/avatar/memory_bank")

    args = ap.parse_args()

    runs = read_runs_jsonl(args.runs)
    buckets = bucket_by_metric(runs, low=float(args.low), high=float(args.high))

    if not buckets.positive or not buckets.negative:
        raise RuntimeError(
            f"Not enough runs to compare. pos={len(buckets.positive)} neg={len(buckets.negative)} "
            f"(adjust thresholds --low/--high or collect more runs)."
        )

    pos_batch, neg_batch = sample_minibatch(
        positive=buckets.positive,
        negative=buckets.negative,
        batch_size=int(args.batch),
        seed=int(args.seed),
    )

    llm = LLMClient(LLMConfig())
    out, usage = run_comparator(llm, positive=pos_batch, negative=neg_batch)
    if not out:
        raise RuntimeError("Comparator failed to produce a valid JSON patch.")

    policy = {
        "planner_system_patch": out.planner_system_patch,
        "rationale": out.rationale,
        "signals": out.signals,
        "thresholds": {"low": float(args.low), "high": float(args.high)},
        "batch": {"pos": len(pos_batch), "neg": len(neg_batch)},
    }

    metrics = {
        "runs": len(runs),
        "pos": len(buckets.positive),
        "neg": len(buckets.negative),
        "neutral": len(buckets.neutral),
        "usage": usage,
    }

    saved = save_policy(policy, bank_dir=args.bank_dir, metrics=metrics)
    pointer = write_best_pointer(saved, bank_dir=args.bank_dir)

    print(f"Saved policy: {saved}")
    print(f"Updated best pointer: {pointer}")


if __name__ == "__main__":
    main()
