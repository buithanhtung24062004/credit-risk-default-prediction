from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import load_uci_data, basic_quality_report, TARGET
from src.features import add_features
from src.model import fit_models, assign_risk_band
from src.business import threshold_table, risk_band_summary
from src.explain import global_importance
from src.plots import save_evaluation_plots

RANDOM_STATE = 42


def main():
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    df = load_uci_data()
    print("Data quality:", basic_quality_report(df))

    df = add_features(df)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    results = fit_models(X_train, X_test, y_train, y_test)
    metrics = {}

    for name, result in results.items():
        metrics[name] = result.metrics
        pred = result.predictions.copy()
        pred["risk_band"] = assign_risk_band(pred["pd"])
        pred.to_csv(f"reports/{name}_test_predictions.csv", index=True)
        threshold_table(y_test, pred["pd"]).to_csv(f"reports/{name}_threshold_analysis.csv", index=False)
        risk_band_summary(pred).to_csv(f"reports/{name}_risk_band_summary.csv", index=False)
        global_importance(result.model).to_csv(f"reports/{name}_feature_importance.csv", index=False)
        save_evaluation_plots(name, result.model, X_test, y_test, pred["pd"])
        joblib.dump(result.model, f"models/{name}.joblib")

    pd.DataFrame(metrics).T.to_csv("reports/model_metrics.csv")
    with open("reports/model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nModel comparison:")
    print(pd.DataFrame(metrics).T.round(4))
    print("\nOutputs saved under reports/, reports/figures/, and models/.")


if __name__ == "__main__":
    main()
