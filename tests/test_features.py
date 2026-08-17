import pandas as pd
from src.features import add_features


def test_feature_engineering_creates_expected_columns():
    row = {
        "limit_bal": 100000,
        "pay_0": 0, "pay_2": 1, "pay_3": 0, "pay_4": 2, "pay_5": 0, "pay_6": 0,
        **{f"bill_amt{i}": 50000 for i in range(1, 7)},
        **{f"pay_amt{i}": 5000 for i in range(1, 7)},
    }
    df = pd.DataFrame([row])
    out = add_features(df)
    assert out.loc[0, "max_delay"] == 2
    assert out.loc[0, "months_delayed"] == 2
    assert out.loc[0, "avg_bill"] == 50000
    assert out.loc[0, "avg_payment"] == 5000
    assert out.loc[0, "utilization_proxy"] == 0.5
    assert out.loc[0, "payment_bill_ratio"] == 0.1
