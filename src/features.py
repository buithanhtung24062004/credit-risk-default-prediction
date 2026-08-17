"""Feature engineering for credit-risk modeling."""
from __future__ import annotations

import numpy as np
import pandas as pd

PAY_COLS = ["pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6"]
BILL_COLS = [f"bill_amt{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"pay_amt{i}" for i in range(1, 7)]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create interpretable behavioral features without using the target."""
    out = df.copy()

    out["max_delay"] = out[PAY_COLS].max(axis=1)
    out["months_delayed"] = (out[PAY_COLS] > 0).sum(axis=1)
    out["avg_bill"] = out[BILL_COLS].mean(axis=1)
    out["avg_payment"] = out[PAY_AMT_COLS].mean(axis=1)

    denom = out["limit_bal"].replace(0, np.nan)
    out["utilization_proxy"] = (out["avg_bill"] / denom).replace([np.inf, -np.inf], np.nan).fillna(0)

    bill_denom = out["avg_bill"].abs().clip(lower=1)
    out["payment_bill_ratio"] = out["avg_payment"] / bill_denom

    return out
