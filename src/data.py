"""Data loading and light cleaning for the UCI credit-default dataset."""
from __future__ import annotations

import pandas as pd

TARGET = "default"

RENAME_MAP = {
    "X1": "limit_bal",
    "X2": "sex",
    "X3": "education",
    "X4": "marriage",
    "X5": "age",
    "X6": "pay_0",
    "X7": "pay_2",
    "X8": "pay_3",
    "X9": "pay_4",
    "X10": "pay_5",
    "X11": "pay_6",
    "X12": "bill_amt1",
    "X13": "bill_amt2",
    "X14": "bill_amt3",
    "X15": "bill_amt4",
    "X16": "bill_amt5",
    "X17": "bill_amt6",
    "X18": "pay_amt1",
    "X19": "pay_amt2",
    "X20": "pay_amt3",
    "X21": "pay_amt4",
    "X22": "pay_amt5",
    "X23": "pay_amt6",
}


def load_uci_data() -> pd.DataFrame:
    """Fetch dataset 350 from UCI and return a clean modeling frame."""
    from ucimlrepo import fetch_ucirepo

    ds = fetch_ucirepo(id=350)
    X = ds.data.features.copy()
    y = ds.data.targets.copy()

    X = X.rename(columns=RENAME_MAP)
    target_col = y.columns[0]
    y = y.rename(columns={target_col: TARGET})

    df = pd.concat([X, y], axis=1)
    df[TARGET] = df[TARGET].astype(int)
    return df


def basic_quality_report(df: pd.DataFrame) -> dict:
    """Return a compact data-quality summary."""
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "default_rate": float(df[TARGET].mean()),
    }
