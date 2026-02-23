"""Metric tracking for training and evaluation."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class MetricTracker:
    """
    Computes standard classification/regression metrics.

    Example:
        tracker = MetricTracker(task="multiclass", num_classes=10)
        metrics = tracker.compute(predictions, labels)
        # {"accuracy": 0.94, "f1_macro": 0.93, ...}
    """

    def __init__(self, task: str = "classification", average: str = "macro") -> None:
        self.task = task
        self.average = average
        self._history: Dict[str, List[float]] = defaultdict(list)

    def compute(self, preds: List, labels: List) -> Dict[str, float]:
        if self.task == "classification":
            return self._classification_metrics(preds, labels)
        else:
            return self._regression_metrics(preds, labels)

    def _classification_metrics(self, preds, labels) -> Dict[str, float]:
        p = np.array(preds)
        y = np.array(labels)
        metrics = {
            "accuracy": round(float(accuracy_score(y, p)), 4),
            f"f1_{self.average}": round(
                float(f1_score(y, p, average=self.average, zero_division=0)), 4
            ),
            f"precision_{self.average}": round(
                float(precision_score(y, p, average=self.average, zero_division=0)), 4
            ),
            f"recall_{self.average}": round(
                float(recall_score(y, p, average=self.average, zero_division=0)), 4
            ),
        }
        for k, v in metrics.items():
            self._history[k].append(v)
        return metrics

    def _regression_metrics(self, preds, labels) -> Dict[str, float]:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        p = np.array(preds, dtype=float)
        y = np.array(labels, dtype=float)
        metrics = {
            "mae": round(float(mean_absolute_error(y, p)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y, p))), 4),
            "r2": round(float(r2_score(y, p)), 4),
        }
        for k, v in metrics.items():
            self._history[k].append(v)
        return metrics

    def best(self, metric: str, mode: str = "max") -> Optional[float]:
        vals = self._history.get(metric, [])
        if not vals:
            return None
        return max(vals) if mode == "max" else min(vals)

    def history(self) -> Dict[str, List[float]]:
        return dict(self._history)
