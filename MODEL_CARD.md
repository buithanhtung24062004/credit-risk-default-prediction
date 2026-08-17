# Model Card — Credit Default PD Case Study

## Intended use

Educational / portfolio analysis of one-month credit-card default risk using a historical public dataset. The output is an estimated **Probability of Default (PD)** for model comparison and portfolio segmentation.

## Not intended for

- real lending approval or rejection;
- pricing, limit assignment, regulatory capital, or collections decisions;
- use on populations outside the dataset without new validation;
- causal claims about borrower characteristics.

## Data

UCI *Default of Credit Card Clients*: 30,000 Taiwan credit-card clients, with credit limits, demographics, six months of repayment status, bill balances and prior payments. The target is default payment in the following month.

## Models

- Logistic Regression: interpretable baseline.
- Random Forest: nonlinear challenger.

## Evaluation philosophy

The project separates:

1. **Discrimination** — ROC-AUC and Average Precision.
2. **Operating performance** — precision/recall/F1 across candidate thresholds.
3. **Probability quality** — Brier Score and calibration curve.
4. **Portfolio interpretation** — observed default rates inside illustrative PD bands.

Accuracy is not used as the primary model-selection criterion because the target is imbalanced and error costs are asymmetric.

## Governance / limitations

The data are historical (2005), geographically specific, and do not provide a clean observation/performance-date structure for true out-of-time validation. Demographic features may be legally or ethically restricted in real lending. The risk bands in this repository are illustrative, not regulatory grades. A production implementation would require policy review, fairness testing, stability/drift monitoring, challenger governance, out-of-time validation, and explicit economic cost assumptions.
