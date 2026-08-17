"""Plot helpers."""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay


def save_evaluation_plots(name, model, X_test, y_test, prob, out_dir="reports/figures"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    RocCurveDisplay.from_predictions(y_test, prob)
    plt.title(f"ROC Curve — {name}")
    plt.tight_layout(); plt.savefig(out / f"{name}_roc.png", dpi=160); plt.close()

    PrecisionRecallDisplay.from_predictions(y_test, prob)
    plt.title(f"Precision–Recall Curve — {name}")
    plt.tight_layout(); plt.savefig(out / f"{name}_pr.png", dpi=160); plt.close()

    frac_pos, mean_pred = calibration_curve(y_test, prob, n_bins=10, strategy="quantile")
    plt.figure()
    plt.plot(mean_pred, frac_pos, marker="o", label=name)
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.xlabel("Mean predicted PD")
    plt.ylabel("Observed default rate")
    plt.title(f"Calibration — {name}")
    plt.legend(); plt.tight_layout(); plt.savefig(out / f"{name}_calibration.png", dpi=160); plt.close()
