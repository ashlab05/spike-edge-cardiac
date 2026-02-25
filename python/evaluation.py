"""
evaluation.py
Compute and display classification metrics for SNN vs baseline comparison.
"""

import numpy as np

try:
    from sklearn.metrics import (
        confusion_matrix,
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        classification_report,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def compute_metrics(y_true: list, y_pred: list) -> dict:
    """
    Compute accuracy, precision, recall, F1, confusion matrix.
    y_true, y_pred: lists of ints (0/1).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    accuracy  = (tp + tn) / (tp + tn + fp + fn + 1e-9)
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)

    return {
        "accuracy":  round(accuracy,  4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def print_report(metrics: dict, label: str = "Model"):
    bar = "─" * 40
    print(f"\n{bar}")
    print(f"  {label}")
    print(bar)
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1-Score  : {metrics['f1']:.4f}")
    print(f"  TP={metrics['tp']}  TN={metrics['tn']}  FP={metrics['fp']}  FN={metrics['fn']}")
    cm = metrics["confusion_matrix"]
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"    FN={cm[1][0]}  TP={cm[1][1]}")
    print(bar)


if __name__ == "__main__":
    # Demo with synthetic data
    import random
    random.seed(42)
    n = 200
    y_true = [random.choice([0, 0, 0, 1]) for _ in range(n)]

    # Simulate SNN (slightly better than random)
    y_snn  = [1 if (t == 1 and random.random() > 0.15)
                or (t == 0 and random.random() < 0.07)
                else t for t in y_true]

    # Simulate baseline (more false positives)
    y_base = [1 if (t == 1 and random.random() > 0.25)
                or (t == 0 and random.random() < 0.18)
                else t for t in y_true]

    from baseline_threshold import detect_batch
    metrics_snn  = compute_metrics(y_true, y_snn)
    metrics_base = compute_metrics(y_true, y_base)

    print_report(metrics_snn,  label="SNN (LIF-based)")
    print_report(metrics_base, label="Threshold Baseline")
