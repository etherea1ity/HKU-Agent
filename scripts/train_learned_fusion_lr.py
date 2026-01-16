"""Train Logistic Regression (LR) for learned fusion.

Reads JSONL produced by scripts/build_learned_fusion_dataset.py and trains a simple
logistic regression classifier that predicts whether a candidate chunk is relevant.

Outputs:
- rag/models/learned_fusion_lr.joblib
- rag/models/learned_fusion_lr_meta.json

The saved artifact includes the sklearn Pipeline and feature names.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _vectorize(rows: List[Dict[str, Any]], feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    x = np.zeros((len(rows), len(feature_names)), dtype=np.float32)
    y = np.zeros((len(rows),), dtype=np.int64)

    for i, r in enumerate(rows):
        feats = r.get("features") or {}
        for j, name in enumerate(feature_names):
            x[i, j] = float(feats.get(name, 0.0))
        y[i] = int(r.get("label", 0))

    return x, y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/learned_fusion")
    ap.add_argument("--train", default=None)
    ap.add_argument("--test", default=None)

    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--max-iter", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)

    ap.add_argument("--model-out", default="rag/models/learned_fusion_lr.joblib")
    ap.add_argument("--meta-out", default="rag/models/learned_fusion_lr_meta.json")

    args = ap.parse_args()

    train_path = args.train or os.path.join(args.data_dir, "train.jsonl")
    test_path = args.test or os.path.join(args.data_dir, "test.jsonl")

    train_rows = _read_jsonl(train_path)
    test_rows = _read_jsonl(test_path)

    if not train_rows:
        raise RuntimeError(f"Empty train set: {train_path}")
    if not test_rows:
        raise RuntimeError(f"Empty test set: {test_path}")

    # Must match features produced in build_learned_fusion_dataset.py
    feature_names = [
        "bm25_score",
        "bm25_rank",
        "bm25_present",
        "bm25_rrf",
        "semantic_score",
        "semantic_rank",
        "semantic_present",
        "semantic_rrf",
        "best_rank",
        "best_rrf",
    ]

    x_train, y_train = _vectorize(train_rows, feature_names)
    x_test, y_test = _vectorize(test_rows, feature_names)

    pipeline: Pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    C=float(args.C),
                    max_iter=int(args.max_iter),
                    random_state=int(args.seed),
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )

    pipeline.fit(x_train, y_train)

    prob_test = pipeline.predict_proba(x_test)[:, 1]
    pred_test = (prob_test >= 0.5).astype(np.int64)

    metrics: Dict[str, Any] = {
        "train_size": int(len(train_rows)),
        "test_size": int(len(test_rows)),
        "pos_rate_train": float(np.mean(y_train)),
        "pos_rate_test": float(np.mean(y_test)),
        "roc_auc": float(roc_auc_score(y_test, prob_test)) if len(np.unique(y_test)) > 1 else None,
        "pr_auc": float(average_precision_score(y_test, prob_test)) if len(np.unique(y_test)) > 1 else None,
        "accuracy": float(accuracy_score(y_test, pred_test)),
        "f1": float(f1_score(y_test, pred_test, zero_division=0)),
        "log_loss": float(log_loss(y_test, prob_test, labels=[0, 1])),
        "confusion_matrix": confusion_matrix(y_test, pred_test).tolist(),
        "classification_report": classification_report(y_test, pred_test, zero_division=0),
    }

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)

    artifact = {
        "pipeline": pipeline,
        "feature_names": feature_names,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "train_path": train_path,
        "test_path": test_path,
        "params": {"C": float(args.C), "max_iter": int(args.max_iter), "seed": int(args.seed)},
    }

    joblib.dump(artifact, args.model_out)

    with open(args.meta_out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_out": args.model_out,
                "feature_names": feature_names,
                "trained_at": artifact["trained_at"],
                "metrics": metrics,
                "params": artifact["params"],
                "train_path": train_path,
                "test_path": test_path,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Loaded train: {train_path} ({len(train_rows)} rows)")
    print(f"Loaded test:  {test_path} ({len(test_rows)} rows)")
    print("--- Metrics ---")
    for k, v in metrics.items():
        if k == "classification_report":
            continue
        print(f"{k}: {v}")
    print("--- Classification report ---")
    print(metrics["classification_report"])
    print(f"Saved model: {args.model_out}")
    print(f"Saved meta:  {args.meta_out}")


if __name__ == "__main__":
    main()
