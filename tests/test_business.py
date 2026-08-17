import pandas as pd
from src.business import threshold_table, risk_band_summary


def test_threshold_table_has_expected_metrics():
    y = [0, 0, 1, 1]
    p = [0.1, 0.4, 0.6, 0.9]
    out = threshold_table(y, p, thresholds=[0.5])
    assert out.loc[0, "recall_default"] == 1.0
    assert out.loc[0, "precision_default"] == 1.0


def test_risk_band_summary_counts_all_rows():
    pred = pd.DataFrame({
        "actual": [0, 1, 0],
        "pd": [0.05, 0.30, 0.60],
        "risk_band": pd.Categorical(["Low", "High", "Very High"],
            categories=["Low", "Medium", "High", "Very High"], ordered=True),
    })
    out = risk_band_summary(pred)
    assert out["customers"].sum() == 3
