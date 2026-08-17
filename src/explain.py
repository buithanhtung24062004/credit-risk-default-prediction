"""Model explainability helpers using native coefficients/importances."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _feature_names(pipe):
    prep = pipe.named_steps["preprocessor"]
    return prep.get_feature_names_out()


def global_importance(pipe, top_n: int = 25) -> pd.DataFrame:
    """Return model-native global importance; intended for portfolio interpretation, not causality."""
    model = pipe.named_steps["model"]
    names = _feature_names(pipe)
    if hasattr(model, "coef_"):
        values = model.coef_[0]
        df = pd.DataFrame({"feature": names, "importance": values})
        df["abs_importance"] = df["importance"].abs()
        return df.sort_values("abs_importance", ascending=False).head(top_n)
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        df = pd.DataFrame({"feature": names, "importance": values})
        df["abs_importance"] = np.abs(df["importance"])
        return df.sort_values("abs_importance", ascending=False).head(top_n)
    return pd.DataFrame(columns=["feature", "importance", "abs_importance"])
