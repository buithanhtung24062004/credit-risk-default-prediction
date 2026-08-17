# Credit Risk Default Prediction

End-to-end **Probability of Default (PD)** portfolio project using the UCI **Default of Credit Card Clients** dataset. The project compares an interpretable Logistic Regression baseline with a Random Forest challenger, evaluates discrimination and probability quality, studies business threshold trade-offs, and converts predicted PDs into portfolio risk bands.

## Executive results

The full pipeline was run on the official dataset of **30,000 customers**.

| Model | ROC-AUC | Average Precision | Precision | Recall | F1 | Brier Score |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7476 | 0.4979 | 0.4414 | **0.6014** | 0.5091 | 0.1934 |
| Random Forest | **0.7730** | **0.5509** | **0.5519** | 0.5207 | **0.5359** | **0.1489** |

**Model choice:** Random Forest is the stronger overall model because it improves ROC-AUC, Average Precision, precision, F1 and Brier Score. Logistic Regression has higher recall at the default 0.50 cutoff, which motivates explicit threshold analysis rather than choosing a model from AUC alone.

### A defensible operating point

At a **Random Forest threshold of 0.40**, the model achieves:

- Recall: **59.5%**
- Precision: **48.2%**
- Flag rate: **27.3%**
- True positives: **790**
- False positives: **850**

Compared with Logistic Regression at 0.50, this keeps nearly the same default recall (**59.5% vs 60.1%**) while improving precision (**48.2% vs 44.1%**) and reducing false positives from **1,010 to 850**.

The 0.40 threshold is an **illustrative portfolio decision point**, not a production lending policy. A real cutoff should be selected from explicit expected-loss economics, operational capacity, regulation and out-of-time validation.

### Risk-band separation

Random Forest PDs produce increasing observed default rates across risk segments:

| Risk band | Portfolio share | Avg predicted PD | Observed default rate |
|---|---:|---:|---:|
| Low | 12.55% | 6.71% | **5.18%** |
| Medium | 38.48% | 17.46% | **11.09%** |
| High | 28.10% | 34.23% | **20.23%** |
| Very High | 20.87% | 70.61% | **55.19%** |

The observed default rate in the Very High group is more than ten times the Low-risk group, showing useful portfolio segmentation.

See **[RESULTS.md](RESULTS.md)** for the full interpretation and the `reports/` folder for reproducible output tables.

## Why this project matters

A useful credit-risk model should answer more than “what is the accuracy?” This project focuses on questions that are closer to real risk work:

- How well does the model rank risky borrowers above safer borrowers?
- How reliable are predicted probabilities?
- What is the trade-off between missing defaulters and flagging non-defaulters?
- Which operating threshold is defensible for a stated objective?
- Do predicted PDs create meaningful portfolio risk segments?
- Which variables drive model decisions, and what are the interpretation limits?

## Dataset

**UCI Machine Learning Repository — Default of Credit Card Clients**

- 30,000 credit card clients
- Target: default payment in the following month
- Demographics, credit limit, repayment status, bill amounts and payment amounts
- No missing cells in the downloaded dataset
- Observed portfolio default rate: approximately **22.12%**

The raw dataset is not committed. `run_analysis.py` retrieves it through `ucimlrepo`.

## Project structure

```text
.
├── .github/workflows/ci.yml
├── data/README.md
├── notebooks/01_credit_risk_walkthrough.ipynb
├── reports/
│   ├── model_metrics.csv
│   ├── random_forest_threshold_analysis.csv
│   ├── random_forest_risk_band_summary.csv
│   └── random_forest_feature_importance.csv
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
├── RESULTS.md
├── requirements.txt
└── run_analysis.py
```

## Modeling approach

### 1. Data preparation

The target is standardized to `default`. The project uses a stratified train/test split to preserve the class ratio.

### 2. Feature engineering

Behavioral credit-risk features include:

- number of delinquent months
- maximum repayment delay
- recent repayment status
- utilization proxy
- average bill amount
- average payment amount
- payment-to-bill ratio

### 3. Models

**Logistic Regression** is the transparent baseline.  
**Random Forest** is the nonlinear challenger.

### 4. Evaluation

The pipeline reports:

- ROC-AUC
- Average Precision
- Precision / Recall / F1
- Brier Score
- confusion matrix
- calibration curves
- ROC and precision-recall curves
- threshold analysis
- risk-band performance
- feature importance / model coefficients

Accuracy is intentionally not the headline metric because defaults are imbalanced and error costs are asymmetric.

## Key feature signals

The highest Random Forest importance values are dominated by repayment behavior and delinquency history:

1. `months_delayed`
2. `pay_0`
3. `max_delay`
4. `utilization_proxy`
5. `payment_bill_ratio`

Feature importance measures model reliance; it is **not causal and not directional**.

## Run the project

```bash
pip install -r requirements.txt
python run_analysis.py
```

Outputs are written under `reports/`, `reports/figures/` and `models/`.

### Google Colab

```python
!git clone https://github.com/buithanhtung24062004/credit-risk-default-prediction.git
%cd credit-risk-default-prediction
!pip install -r requirements.txt
!python run_analysis.py
```

Inspect metrics with:

```python
import pandas as pd
pd.read_csv("reports/model_metrics.csv")
```

## Model governance and limitations

This is an educational portfolio case study, **not a production lending model**.

Important limitations:

- the data represents Taiwanese credit card clients from 2005;
- the split is random rather than out-of-time;
- no reject-inference analysis is performed;
- no production fairness, regulatory or adverse-action validation is performed;
- stability and population drift are not assessed;
- the illustrative risk bands and 0.40 operating threshold are not lending policies.

See `MODEL_CARD.md` for additional governance notes.

## Tests and CI

```bash
pytest -q
```

GitHub Actions runs tests automatically on pushes and pull requests.

## Portfolio materials

- `RESULTS.md` — real model results and business interpretation
- `GUIDE_VI.md` — Vietnamese end-to-end walkthrough
- `INTERVIEW_PREP_VI.md` — interview questions and explanation prompts
- `PORTFOLIO_CV.md` — CV and LinkedIn wording using the real results
- `MODEL_CARD.md` — intended use, limitations and governance notes

## Portfolio takeaway

The strongest story in this project is not simply that Random Forest achieved **0.773 ROC-AUC**. It is that model comparison, probability quality, threshold selection and portfolio segmentation were connected to a practical credit-risk decision framework.
