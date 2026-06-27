"""
Probability of Default (PD) Model
RAKBANK Management Associate Programme — Case Study 1
Author: Theo Li
Date: June 2026
"""

# ─────────────────────────────────────────
# 1. IMPORTS
# ─────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
)

# ─────────────────────────────────────────
# 2. LOAD DATA
# ─────────────────────────────────────────
df = pd.read_csv("data/case1_credit_approval.csv.gz")
print(f"Dataset shape: {df.shape}")
print(f"Default rate: {df['TARGET'].mean():.1%}")

# ─────────────────────────────────────────
# 3. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────
print("\nTarget distribution:")
print(df["TARGET"].value_counts())

print("\nMissing values (top 10):")
print(df.isnull().sum().sort_values(ascending=False).head(10))

dr_no  = df[df["HAS_BUREAU_FILE"] == 0]["TARGET"].mean()
dr_has = df[df["HAS_BUREAU_FILE"] == 1]["TARGET"].mean()
print(f"\nDefault rate - No bureau file : {dr_no:.1%}")
print(f"Default rate - Has bureau file: {dr_has:.1%}")

# ─────────────────────────────────────────
# 4. FEATURE ENGINEERING
# Three dimensions: Affordability / Stability / Credit History
# ─────────────────────────────────────────
def engineer_features(df):
    d = df.copy()

    # Stability
    d["AGE_YEARS"]        = (-d["DAYS_BIRTH"]).clip(lower=0) / 365.25
    d["EMPLOYMENT_YEARS"] = (-d["DAYS_EMPLOYED"]).clip(lower=0) / 365.25

    # Affordability
    d["ANNUITY_TO_INCOME"]     = d["AMT_ANNUITY"]       / (d["AMT_INCOME_TOTAL"] + 1)
    d["CREDIT_TO_INCOME"]      = d["AMT_CREDIT"]        / (d["AMT_INCOME_TOTAL"] + 1)
    d["BUREAU_DEBT_TO_INCOME"] = d["BUREAU_TOTAL_DEBT"] / (d["AMT_INCOME_TOTAL"] + 1)
    d["LOAN_COST_RATIO"]       = (d["AMT_ANNUITY"] * d["TERM_MONTHS"]) / (d["AMT_CREDIT"] + 1)

    # Credit history
    dpd_cols = ["BUREAU_DPD_30_COUNT", "BUREAU_DPD_60_COUNT", "BUREAU_DPD_90_COUNT"]
    d["BUREAU_TOTAL_DPD"]        = d[dpd_cols].fillna(0).sum(axis=1)
    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    d["EXT_SOURCE_MEAN"]         = d[ext_cols].mean(axis=1)
    d["EXT_SOURCE_MIN"]          = d[ext_cols].min(axis=1)
    d["BUREAU_UTILIZATION_SAFE"] = d["BUREAU_UTILIZATION"].fillna(0)

    return d

df = engineer_features(df)

# ─────────────────────────────────────────
# 5. STRATIFIED TRAIN / VALIDATION SPLIT
# Stratified to preserve 8.1% default rate in both sets
# ─────────────────────────────────────────
DROP     = ["SK_ID_CURR", "TARGET", "APPLICANT_SCENARIO", "DAYS_BIRTH", "DAYS_EMPLOYED"]
FEATURES = [c for c in df.columns if c not in DROP]
X, y     = df[FEATURES], df["TARGET"]

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(sss.split(X, y))
X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

print(f"\nTrain: {X_train.shape} | Val: {X_val.shape}")
print(f"Train default rate: {y_train.mean():.1%} | Val default rate: {y_val.mean():.1%}")

# ─────────────────────────────────────────
# 6. PREPROCESSING PIPELINE
# fit() on training data only — prevents leakage into validation set
# ─────────────────────────────────────────
cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
])

preprocessor = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols),
])

# ─────────────────────────────────────────
# 7. LOGISTIC REGRESSION (BASELINE MODEL)
# class_weight='balanced' handles 8.1% class imbalance
# ─────────────────────────────────────────
lr_pipe = Pipeline([
    ("prep", preprocessor),
    ("clf",  LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        C=0.1,
        random_state=42,
    )),
])

lr_pipe.fit(X_train, y_train)
lr_prob = lr_pipe.predict_proba(X_val)[:, 1]

# ─────────────────────────────────────────
# 8. EVALUATION
# AUC / Gini / KS — not accuracy (misleading under imbalance)
# ─────────────────────────────────────────
lr_auc  = roc_auc_score(y_val, lr_prob)
lr_gini = 2 * lr_auc - 1
lr_ap   = average_precision_score(y_val, lr_prob)
fpr, tpr, _ = roc_curve(y_val, lr_prob)
lr_ks   = float(np.max(tpr - fpr))

print(f"\nModel Performance:")
print(f"  AUC-ROC       : {lr_auc:.4f}")
print(f"  Gini          : {lr_gini:.4f}")
print(f"  KS Statistic  : {lr_ks:.4f}")
print(f"  Avg Precision : {lr_ap:.4f}")

# ─────────────────────────────────────────
# 9. SHAP EXPLAINABILITY
# ─────────────────────────────────────────
clf           = lr_pipe.named_steps["clf"]
prep          = lr_pipe.named_steps["prep"]
all_names     = num_cols + cat_cols
X_val_prep    = prep.transform(X_val)
X_val_prep_df = pd.DataFrame(X_val_prep, columns=all_names)

explainer   = shap.LinearExplainer(clf, X_val_prep_df)
shap_values = explainer.shap_values(X_val_prep_df)

shap.summary_plot(shap_values, X_val_prep_df, max_display=15, show=False)
plt.title("SHAP Summary — Key Drivers of Default Risk", fontweight="bold")
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: shap_summary.png")

# ─────────────────────────────────────────
# 10. HIGH-RISK APPLICANT EXPLANATION
# ─────────────────────────────────────────
val_df            = X_val.copy().reset_index(drop=True)
val_df["PD_SCORE"]= lr_prob
val_df["ACTUAL"]  = y_val.values

hr_idx = val_df[val_df["ACTUAL"] == 1]["PD_SCORE"].idxmax()
hr     = val_df.loc[hr_idx]

print(f"\nHigh-Risk Applicant (PD: {hr['PD_SCORE']:.1%}):")
print(f"  Annuity / Income : {hr['ANNUITY_TO_INCOME']:.1%}")
print(f"  Credit / Income  : {hr['CREDIT_TO_INCOME']:.2f}x")
print(f"  EXT_SOURCE_MEAN  : {hr['EXT_SOURCE_MEAN']:.3f}")

shap_row    = shap_values[hr_idx]
shap_series = pd.Series(shap_row, index=all_names)
shap_top    = shap_series[shap_series.abs().nlargest(12).index].sort_values()

fig, ax = plt.subplots(figsize=(10, 6))
colors = ["#DC2626" if v > 0 else "#2563EB" for v in shap_top.values]
ax.barh(range(len(shap_top)), shap_top.values, color=colors, height=0.6)
ax.set_yticks(range(len(shap_top)))
ax.set_yticklabels(shap_top.index, fontsize=9)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("SHAP Value (positive = increases default risk)")
ax.set_title(f"High-Risk Applicant — PD: {hr['PD_SCORE']:.1%}", fontweight="bold")
plt.tight_layout()
plt.savefig("shap_waterfall.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: shap_waterfall.png")

# ─────────────────────────────────────────
# 11. RISK BANDING
# Thresholds at 30th / 60th percentile
# ─────────────────────────────────────────
t1 = float(np.percentile(lr_prob, 30))
t2 = float(np.percentile(lr_prob, 60))

val_df["RISK_BAND"] = val_df["PD_SCORE"].apply(
    lambda s: "Auto Approve" if s < t1 else ("Manual Review" if s < t2 else "Decline")
)

band_order = ["Auto Approve", "Manual Review", "Decline"]
summary = (
    val_df.groupby("RISK_BAND")
    .agg(Count=("PD_SCORE","count"), DefaultRate=("ACTUAL","mean"))
    .loc[band_order]
)
summary["PctPortfolio"] = summary["Count"] / summary["Count"].sum() * 100

print(f"\nRisk Band Summary (thresholds: {t1:.1%} / {t2:.1%}):")
print(summary.round(4).to_string())
