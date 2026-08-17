# Portfolio & CV wording

## GitHub description

End-to-end credit risk PD modeling case study on 30,000 customers: behavioral feature engineering, Logistic Regression vs Random Forest, threshold trade-offs, calibration, portfolio risk bands, tests and model governance notes.

## CV bullets

- Built an end-to-end credit-default **Probability of Default (PD)** pipeline on **30,000 customer records** using Python, behavioral feature engineering, Logistic Regression and Random Forest; the best model achieved **0.773 test ROC-AUC** and **0.551 Average Precision**.
- Evaluated discrimination, default capture and probability quality using ROC-AUC, Average Precision, precision/recall, F1, Brier Score and calibration; selected an illustrative **0.40 Random Forest threshold** that retained **59.5% default recall** while reducing false positives from **1,010 to 850** versus the Logistic Regression baseline at 0.50.
- Translated customer-level PD estimates into portfolio risk bands with observed default rates rising from **5.2% in Low Risk to 55.2% in Very High Risk**, and documented explainability, limitations and model-governance considerations.

## Short project summary for CV

**Credit Risk Default Prediction | Python, scikit-learn, pandas**  
Developed a Probability of Default case study using the UCI Default of Credit Card Clients dataset. Compared an interpretable Logistic Regression baseline with a Random Forest challenger, evaluated ranking/calibration and threshold trade-offs, and converted PD estimates into business-facing risk segments. Best model test ROC-AUC: **0.773**.

## LinkedIn/project summary

Developed an end-to-end credit-risk portfolio case study using 30,000 customer records from the UCI Default of Credit Card Clients dataset. The project focuses on Probability of Default rather than binary classification alone, compares Logistic Regression with Random Forest, evaluates calibration and operating-threshold trade-offs, and translates model output into portfolio risk bands. Random Forest achieved a test ROC-AUC of 0.773 and Average Precision of 0.551; risk segmentation produced observed default rates from 5.2% in the Low band to 55.2% in the Very High band. The repository includes reproducible code, tests, CI, a model card and interview-ready documentation.

## Interview positioning

Do not claim the model is production-ready. Describe the 0.40 cutoff as an **illustrative operating point**, chosen because it provides roughly the same default recall as Logistic Regression at 0.50 while improving precision and reducing false positives. A real lending threshold would require explicit expected-loss economics, operational capacity, regulation, fairness review, stability testing and out-of-time validation.
