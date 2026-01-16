from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional, Set


_CIT_RE = re.compile(r"\[(\d{1,3})\]")
_CTX_BLOCK_RE = re.compile(r"^\[(\d{1,3})\]\s", re.MULTILINE)


def parse_answer_citations(answer: str) -> Set[int]:
    if not answer:
        return set()
    return {int(m.group(1)) for m in _CIT_RE.finditer(answer)}


def parse_context_block_ids(context: str) -> Set[int]:
    if not context:
        return set()
    return {int(m.group(1)) for m in _CTX_BLOCK_RE.finditer(context)}


def citations_ok(answer: str, context: str, *, require_at_least_one: bool = True) -> bool:
    used = parse_answer_citations(answer)
    avail = parse_context_block_ids(context)

    if require_at_least_one and not used:
        return False
    # every citation used must exist
    return used.issubset(avail)


@dataclass(frozen=True)
class MetricResult:
    metric: float
    citations_ok: bool
    correctness: Optional[float] = None
    groundedness: Optional[float] = None


def combine_metrics(
    *,
    citations_ok_flag: bool,
    correctness: Optional[float],
    groundedness: Optional[float],
    w_correct: float = 0.7,
    w_ground: float = 0.3,
) -> MetricResult:
    """Combine judge scores into a single scalar metric.

    If citations are invalid/missing, metric is forced to 0 (strict).
    """

    if not citations_ok_flag:
        return MetricResult(metric=0.0, citations_ok=False, correctness=correctness, groundedness=groundedness)

    c = float(correctness) if correctness is not None else 0.0
    g = float(groundedness) if groundedness is not None else 0.0

    metric = max(0.0, min(1.0, w_correct * c + w_ground * g))
    return MetricResult(metric=metric, citations_ok=True, correctness=correctness, groundedness=groundedness)
