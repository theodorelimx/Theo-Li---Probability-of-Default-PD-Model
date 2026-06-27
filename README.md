# Probability of Default (PD) Model
**RAKBANK Management Associate Programme — Credit Risk Case Study 1**  
Author: Theo Li | June 2026

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
rakbank-pd-model/
├── README.md
├── pd_model.ipynb       
├── data/
│   └── case1_credit_approval.csv.gz
└── report/
    └── RAKBANK_PD_Model_TheoLi.html  
```

---

## How to Run

**Install dependencies:**
```bash
pip install pandas numpy scikit-learn lightgbm shap matplotlib
```

**Run the model:**
```bash
python pd_model.py
```

**View the report:**  
Open `report/RAKBANK_PD_Model_TheoLi.html` in any browser.

---

## Methodology

### 1. Dataset Exploration
- 50,000 applicants with 8.1% default rate (class imbalance)
- 17.5% of applicants have no bureau file (thin-file / first-time borrowers)
- EXT_SOURCE scores show strongest separation between defaulters and non-defaulters

### 2. Feature Engineering
Features are organised into three MECE dimensions:

**Affordability** — can this person sustain the repayments?
- `ANNUITY_TO_INCOME` — monthly repayment as share of annual income
- `CREDIT_TO_INCOME` — loan as multiple of annual income
- `BUREAU_DEBT_TO_INCOME` — total existing debt relative to income
- `LOAN_COST_RATIO` — total repayments vs original loan principal

**Stability** — how predictable is their financial situation?
- `AGE_YEARS` — proxy for financial maturity
- `EMPLOYMENT_YEARS` — income reliability signal

**Credit History** — what does past behaviour tell us?
- `BUREAU_TOTAL_DPD` — sum of all historical late payment events
- `EXT_SOURCE_MEAN` — average of three external credit bureau scores
- `EXT_SOURCE_MIN` — weakest external score (captures masked red flags)
- `BUREAU_UTILIZATION_SAFE` — credit utilisation across bureau accounts

### 3. Leakage-Free Pipeline
- Stratified 80/20 train/validation split (preserves 8.1% default rate in both sets)
- All preprocessing (imputation, encoding, scaling) fitted on training data only
- sklearn Pipeline enforces this automatically

### 4. Model
- **Logistic Regression** — industry standard baseline for credit scoring
- `class_weight='balanced'` handles class imbalance
- Evaluated on AUC-ROC, Gini, KS Statistic, and Avg Precision (not accuracy)

### 5. Explainability
- SHAP (SHapley Additive exPlanations) decomposes each prediction into feature contributions
- Global summary plot shows drivers across all 10,000 validation applicants
- Individual waterfall plot explains the highest-risk applicant

### 6. Risk Banding
Scores translated into three decision tiers:

| Band | Threshold | Actual Default Rate | Share |
|---|---|---|---|
| Auto Approve | Bottom 30% of scores | 0.8% | 30% |
| Manual Review | Middle 30% of scores | 1.7% | 30% |
| Decline | Top 40% of scores | 18.4% | 40% |

The Decline band captures defaults at over 20 times the rate of the Auto Approve band, confirming meaningful risk separation.

---

## Key Findings

1. **EXT_SOURCE scores are the strongest predictor** — external bureau assessments consistently separate defaulters from non-defaulters
2. **Repayment burden (ANNUITY_TO_INCOME) is the top engineered feature** — highest SHAP magnitude across the validation set
3. **Thin-file applicants are not high risk** — 17.5% of applicants have no bureau file but a 3.9% default rate vs 9.0% for those with records
4. **Logistic Regression outperforms on this dataset** — EXT_SOURCE signals are near-linear, so the simpler model is both more accurate and more interpretable

---

## Dependencies

```
pandas >= 1.5
numpy >= 1.23
scikit-learn >= 1.2
shap >= 0.41
matplotlib >= 3.6
```
