"""Business-facing summaries for PD model outputs."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score


def threshold_table(y_true, prob, thresholds=None) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.arange(0.10, 0.71, 0.05)
    rows = []
    y = np.asarray(y_true)
    p = np.asarray(prob)
    for t in thresholds:
        pred = (p >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        rows.append({
            "threshold": round(float(t), 2),
            "flag_rate": float(pred.mean()),
            "precision_default": precision_score(y, pred, zero_division=0),
            "recall_default": recall_score(y, pred, zero_division=0),
            "f1_default": f1_score(y, pred, zero_division=0),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })
    return pd.DataFrame(rows)


def risk_band_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    out = predictions.copy()
    summary = (
        out.groupby("risk_band", observed=False)
        .agg(customers=("actual", "size"), average_pd=("pd", "mean"), observed_default_rate=("actual", "mean"))
        .reset_index()
    )
    total = max(len(out), 1)
    summary["portfolio_share"] = summary["customers"] / total
    return summary
