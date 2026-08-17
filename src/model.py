"""Model training and evaluation utilities."""
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score, confusion_matrix, brier_score_loss
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL = ["sex", "education", "marriage"]


@dataclass
class ModelResult:
    name: str
    model: Pipeline
    metrics: dict
    predictions: pd.DataFrame


def _preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cats = [c for c in CATEGORICAL if c in X.columns]
    nums = [c for c in X.columns if c not in cats]
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", num_pipe, nums),
        ("cat", cat_pipe, cats),
    ])


def _metrics(y_true, prob, threshold: float = 0.5) -> dict:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "roc_auc": roc_auc_score(y_true, prob),
        "average_precision": average_precision_score(y_true, prob),
        "precision_default": precision_score(y_true, pred, zero_division=0),
        "recall_default": recall_score(y_true, pred, zero_division=0),
        "f1_default": f1_score(y_true, pred, zero_division=0),
        "brier_score": brier_score_loss(y_true, prob),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "threshold": threshold,
    }


def fit_models(X_train, X_test, y_train, y_test, threshold: float = 0.5):
    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=5, class_weight="balanced_subsample",
            random_state=42, n_jobs=-1
        ),
    }

    results = {}
    for name, estimator in models.items():
        pipe = Pipeline([
            ("preprocessor", _preprocessor(X_train)),
            ("model", estimator),
        ])
        pipe.fit(X_train, y_train)
        prob = pipe.predict_proba(X_test)[:, 1]
        pred_df = pd.DataFrame({"actual": y_test.to_numpy(), "pd": prob}, index=y_test.index)
        results[name] = ModelResult(name, pipe, _metrics(y_test, prob, threshold), pred_df)
    return results


def assign_risk_band(prob: pd.Series) -> pd.Series:
    """Portfolio-friendly risk bands; illustrative, not regulatory grades."""
    return pd.cut(
        prob,
        bins=[-0.001, 0.10, 0.25, 0.50, 1.0],
        labels=["Low", "Medium", "High", "Very High"],
    )
