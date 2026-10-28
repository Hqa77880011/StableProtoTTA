"""Post-hoc order statistics; labels never enter the adapter."""

import numpy as np


def order_metrics(rows, correct, counts, rare_classes):
    """Pool accepted counts within an order and report class/exposure averages."""
    correct, counts = np.asarray(correct), np.asarray(counts)
    present = counts > 0
    per_class = np.divide(correct, counts, out=np.zeros_like(correct, dtype=float), where=present)
    rare = np.isin(np.arange(len(counts)), rare_classes) & present
    common = ~np.isin(np.arange(len(counts)), rare_classes) & present
    accepted = sum(r["accepted"] for r in rows)
    return {
        "accuracy": float(np.mean([r["accuracy"] for r in rows])),
        "micro_accuracy": float(100 * correct.sum() / counts.sum()),
        "macro_accuracy": float(100 * per_class[present].mean()),
        "rare_accuracy": float(100 * per_class[rare].mean()) if rare.any() else None,
        "common_accuracy": float(100 * per_class[common].mean()) if common.any() else None,
        "accepted_ratio": accepted / sum(r["samples"] for r in rows),
        "accepted_precision": sum(r["accepted_correct"] for r in rows) / accepted if accepted else None,
    }
