from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_BANK_DIR = "data/avatar/memory_bank"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_policy(
    policy: Dict[str, Any],
    *,
    bank_dir: str = DEFAULT_BANK_DIR,
    name: Optional[str] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> str:
    """Save a comparator-produced policy into the memory bank.

    Returns the saved file path.
    """

    ensure_dir(bank_dir)

    ts = _utc_now().replace(":", "-")
    base = name or f"policy_{ts}"
    path = os.path.join(bank_dir, f"{base}.json")

    rec = {
        "saved_at": _utc_now(),
        "metrics": metrics or {},
        "policy": policy,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)

    return path


def list_policies(bank_dir: str = DEFAULT_BANK_DIR) -> List[str]:
    if not os.path.isdir(bank_dir):
        return []
    files = [os.path.join(bank_dir, f) for f in os.listdir(bank_dir) if f.endswith(".json")]
    files.sort()
    return files


def load_policy(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_best_pointer(best_path: str, *, bank_dir: str = DEFAULT_BANK_DIR) -> str:
    ensure_dir(bank_dir)
    out = os.path.join(bank_dir, "best.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"best": best_path, "updated_at": _utc_now()}, f, ensure_ascii=False, indent=2)
    return out


def resolve_best_policy_path(bank_dir: str = DEFAULT_BANK_DIR) -> Optional[str]:
    pointer = os.path.join(bank_dir, "best.json")
    if os.path.exists(pointer):
        try:
            obj = load_policy(pointer)
            p = obj.get("best")
            if isinstance(p, str) and os.path.exists(p):
                return p
        except Exception:
            pass

    # fallback: newest .json that looks like a policy record (excluding best.json)
    files = [p for p in list_policies(bank_dir) if not p.endswith("best.json")]
    if not files:
        return None
    return files[-1]


def extract_system_patch(record: Dict[str, Any]) -> Optional[str]:
    """Return system prompt patch text from a memory bank record."""
    policy = record.get("policy") if isinstance(record, dict) else None
    if not isinstance(policy, dict):
        return None

    patch = policy.get("planner_system_patch")
    if isinstance(patch, str) and patch.strip():
        return patch.strip()
    return None
