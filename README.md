# Credit Risk Default Prediction

A portfolio-ready credit risk case study using the UCI **Default of Credit Card Clients** dataset. The project estimates probability of default (PD), compares an interpretable Logistic Regression baseline with a Random Forest challenger, evaluates ranking and calibration, analyzes operating thresholds, and translates model scores into risk bands for business interpretation.

## Why this project matters

Credit risk models are not useful because they produce a single accuracy score. They are useful when they help answer questions such as:

- Which borrowers are more likely to default?
- How well does the model rank risky borrowers above safer borrowers?
- Are predicted probabilities reasonably calibrated?
- What happens to false negatives and false positives when the decision threshold changes?
- How can model scores be summarized into practical risk segments?

This repository is designed as an end-to-end portfolio project rather than a standalone notebook.

## Dataset

**UCI Machine Learning Repository — Default of Credit Card Clients**

- 30,000 credit card clients
- Target: default payment in the following month
- Demographics, credit limit, repayment status, bill amounts, and payment amounts
- No missing values reported by UCI

The raw dataset is not committed to this repository. `run_analysis.py` retrieves it through `ucimlrepo`.

## Project structure

```text
.
├── .github/workflows/ci.yml
├── data/README.md
├── notebooks/01_credit_risk_walkthrough.ipynb
├── reports/figures/
├── src/
│   ├── business.py
│   ├── data.py
│   ├── explain.py
│   ├── features.py
│   ├── model.py
│   └── plots.py
├── tests/
├── GUIDE_VI.md
├── INTERVIEW_PREP_VI.md
├── MODEL_CARD.md
├── PORTFOLIO_CV.md
├── requirements.txt
└── run_analysis.py
```

## Modeling approach

### 1. Data preparation

The target is standardized to `default`. The analysis uses a stratified train/test split to preserve the class ratio.

### 2. Feature engineering

The project adds intuitive credit-risk features such as:

- recent repayment delinquency indicators
- count of delinquent months
- average bill amount relative to credit limit
- average payment amount relative to credit limit
- payment-to-bill style ratios

These features are deliberately simple and interpretable for a portfolio case study.

### 3. Models

**Logistic Regression** is the baseline because it is transparent and widely used as a benchmark in credit-risk work.

**Random Forest** is used as a nonlinear challenger to test whether interactions and nonlinear patterns improve discriminatory power.

### 4. Evaluation

The project reports:

- ROC-AUC
- Average Precision
- Precision
- Recall
- F1 score
- Brier Score
- Confusion matrix
- Calibration curve
- Threshold analysis
- Risk-band performance
- Feature importance / model coefficients

Accuracy is intentionally not treated as the main metric because default datasets are imbalanced and the business cost of missing a risky borrower can differ substantially from the cost of flagging a safe borrower.

## Run the project

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

Run the full pipeline:

```bash
python run_analysis.py
```

Outputs are written under `reports/` and fitted models under `models/`.

Expected outputs include:

```text
reports/model_metrics.csv
reports/logistic_regression_threshold_analysis.csv
reports/random_forest_threshold_analysis.csv
reports/logistic_regression_risk_band_summary.csv
reports/random_forest_risk_band_summary.csv
reports/logistic_regression_feature_importance.csv
reports/random_forest_feature_importance.csv
reports/figures/
models/
```

## Google Colab

After cloning the repository:

```python
!git clone https://github.com/buithanhtung24062004/credit-risk-default-prediction.git
%cd credit-risk-default-prediction
!pip install -r requirements.txt
!python run_analysis.py
```

Then inspect the model summary:

```python
import pandas as pd
pd.read_csv("reports/model_metrics.csv")
```

## Business interpretation

A probability model should not automatically be converted into an approve/reject decision at a threshold of 0.50. The appropriate cutoff depends on the objective and the relative costs of errors.

For example:

- a lower threshold may capture more future defaulters but flag more non-defaulters;
- a higher threshold may reduce false positives but miss more risky borrowers.

The threshold-analysis output makes this trade-off explicit rather than hiding it behind a single score.

Risk bands provide another useful portfolio view. Predicted PDs are grouped into ordered segments so that observed default rates can be compared across risk levels. A useful model should generally show increasing observed default rates as predicted risk rises.

## Model governance and limitations

This is an educational portfolio case study, not a production lending model.

Important limitations include:

- the dataset represents Taiwanese credit card clients from 2005;
- the train/test split is random rather than out-of-time;
- no reject-inference problem is addressed;
- no fairness, legal, regulatory, or adverse-action validation is performed;
- model stability and population drift are not assessed;
- illustrative risk bands and thresholds are not lending policies.

See `MODEL_CARD.md` for a fuller description.

## Tests and CI

Run unit tests with:

```bash
pytest -q
```

GitHub Actions runs the test suite automatically on pushes and pull requests.

## Portfolio materials

- `GUIDE_VI.md` — Vietnamese walkthrough of the project and key concepts
- `INTERVIEW_PREP_VI.md` — interview questions and suggested explanations
- `PORTFOLIO_CV.md` — CV / LinkedIn wording guidance
- `MODEL_CARD.md` — intended use, limitations, and governance notes

## Reproducibility note

Do not report AUC, recall, calibration, or other model results until `run_analysis.py` has been executed successfully on the real UCI dataset. This repository intentionally avoids fabricating performance numbers.
