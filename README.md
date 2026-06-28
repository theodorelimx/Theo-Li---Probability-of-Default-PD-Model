# Probability of Default (PD) Model
**RAKBANK Management Associate Programme | Credit Risk Case Study 1**
**Author:** Theo Li | **Date:** June 2026

---

## Overview

This project builds a leakage-free, interpretable PD model to support RAKBANK's credit approval process. The model predicts the probability that a loan applicant will default, and translates that score into an Approve / Review / Decline lending policy.

**Results:**

| Metric | Score |
|---|---|
| AUC-ROC | 0.8895 |
| Gini | 0.7789 |
| KS Statistic | 0.6296 |
| Avg Precision | 0.5977 |

---

## Repository Structure

```
Theo-Li-Probability-of-Default-PD-Model/
├── RAKBANK_PD_Model_TheoLi.html    HTML report (open in browser)
├── RAKBANK_PD_Model_TheoLi.ipynb  Main notebook
├── case1_credit_approval.csv       Input dataset (50,000 applicants)
└── README.md
```

---

## How to Run

**Option 1: Google Colab (recommended, no installation needed)**

1. Go to colab.research.google.com
2. File > Upload notebook > select `RAKBANK_PD_Model_TheoLi.ipynb`
3. Upload `case1_credit_approval.csv` via the left sidebar file panel
4. In Cell 2, update the file path to: `pd.read_csv("/content/case1_credit_approval (2).csv")`
5. Runtime > Run all

**Option 2: Local**

```bash
pip install pandas numpy scikit-learn shap matplotlib seaborn
jupyter notebook RAKBANK_PD_Model_TheoLi.ipynb
```

---

## Methodology

### 1. Dataset Exploration

- 50,000 applicants with 8.1% default rate
- Class imbalance makes accuracy a misleading metric; AUC-ROC, Gini, and KS are used instead
- 17.5% of applicants have no bureau file (thin-file / first-time borrowers with 3.9% default rate vs 9.0% for those with records)
- EXT_SOURCE scores show the strongest separation between defaulters and non-defaulters

### 2. Feature Engineering

Features are organised into three MECE dimensions:

**Affordability** — can this person sustain the repayments?
- `ANNUITY_TO_INCOME` monthly repayment as share of annual income
- `CREDIT_TO_INCOME` loan as multiple of annual income
- `BUREAU_DEBT_TO_INCOME` total existing debt relative to income
- `LOAN_COST_RATIO` total repayments vs original loan principal

**Stability** — how predictable is their financial situation?
- `AGE_YEARS` proxy for financial maturity
- `EMPLOYMENT_YEARS` income reliability signal

**Credit History** — what does past behaviour tell us?
- `BUREAU_TOTAL_DPD` sum of all historical late payment events
- `EXT_SOURCE_MEAN` average of three external credit bureau scores
- `EXT_SOURCE_MIN` weakest external score (captures masked red flags)
- `BUREAU_UTILIZATION_SAFE` credit utilisation across bureau accounts

### 3. Leakage-Free Pipeline

- Stratified 80/20 train/validation split (preserves 8.1% default rate in both sets)
- All preprocessing (imputation, encoding, scaling) fitted on training data only
- sklearn Pipeline enforces this automatically

### 4. Model

- Logistic Regression as the baseline model
- `class_weight='balanced'` handles class imbalance (8.1% defaulters)
- Evaluated on AUC-ROC, Gini, KS Statistic, and Avg Precision

### 5. Explainability

- SHAP breaks each prediction into individual feature contributions
- Global summary plot shows drivers across all 10,000 validation applicants
- Individual waterfall plot explains a selected high-risk applicant

### 6. Risk Banding

Thresholds set at the 30th and 60th percentile of predicted scores:

| Band | Share | Actual Default Rate |
|---|---|---|
| Auto Approve | 30% | 0.8% |
| Manual Review | 30% | 1.7% |
| Decline | 40% | 18.4% |

The Decline band actual default rate is over 20 times higher than the Auto Approve band, validated on 10,000 held-out applicants.

---

## Key Findings

1. **EXT_SOURCE scores are the strongest predictor.** External bureau assessments consistently separate defaulters from non-defaulters across the validation set.
2. **ANNUITY_TO_INCOME is the top engineered feature.** Repayment burden relative to income has the highest SHAP magnitude and is the primary driver of high-risk decisions.
3. **Thin-file applicants are not high risk.** 17.5% of applicants have no bureau file but a 3.9% actual default rate vs 9.0% for those with records. Auto-declining them would eliminate a significant portion of creditworthy applicants from the pipeline.
4. **Logistic Regression performs well on this dataset.** The model achieves a Gini of 0.78, exceeding the 0.7 industry threshold, while remaining fully interpretable for regulatory and customer-facing explanations.

---
