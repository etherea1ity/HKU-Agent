from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


def read_runs_jsonl(path: str) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            runs.append(json.loads(line))
    return runs


@dataclass(frozen=True)
class BucketedRuns:
    positive: List[Dict[str, Any]]
    negative: List[Dict[str, Any]]
    neutral: List[Dict[str, Any]]


def bucket_by_metric(
    runs: List[Dict[str, Any]],
    *,
    low: float,
    high: float,
    metric_key: str = "metric",
) -> BucketedRuns:
    """Bucket runs by AVATAR thresholds.

    - metric >= high => positive
    - metric <= low  => negative
    - else           => neutral

    Notes:
    - Caller should ensure 0 <= low <= high <= 1.
    """

    pos: List[Dict[str, Any]] = []
    neg: List[Dict[str, Any]] = []
    neu: List[Dict[str, Any]] = []

    for r in runs:
        try:
            m = float(r.get(metric_key, 0.0))
        except Exception:
            m = 0.0

        if m >= high:
            pos.append(r)
        elif m <= low:
            neg.append(r)
        else:
            neu.append(r)

    return BucketedRuns(positive=pos, negative=neg, neutral=neu)


def sample_minibatch(
    *,
    positive: List[Dict[str, Any]],
    negative: List[Dict[str, Any]],
    batch_size: int,
    seed: int = 7,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Sample b/2 positive and b/2 negative runs."""

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    half = batch_size // 2
    if half == 0:
        half = 1

    rng = random.Random(seed)

    pos = list(positive)
    neg = list(negative)
    rng.shuffle(pos)
    rng.shuffle(neg)

    return pos[:half], neg[:half]
