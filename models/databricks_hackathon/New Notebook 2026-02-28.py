# Databricks notebook source
import pandas as pd
import numpy as np
from scipy import stats

path = "/Workspace/Users/dlee23@uw.edu/databricks_hackathon/yield_anomaly_flags.csv"
df = pd.read_csv(path)
df["anomaly_flag"] = pd.to_numeric(df["anomaly_flag"], errors="coerce").fillna(0).astype(int)

print("Shape:", df.shape)
print("anomaly_flag counts:")
print(df["anomaly_flag"].value_counts().sort_index())
print("\nColumns:", list(df.columns))
df.head(10)

# COMMAND ----------

normal = df[df["anomaly_flag"] == 0]
anomaly = df[df["anomaly_flag"] == 1]

print(f"Normal (0): n = {len(normal)}")
print(f"Anomaly (1): n = {len(anomaly)}")
print(f"Anomaly 비율: {len(anomaly)/len(df)*100:.2f}%")

# COMMAND ----------

num_cols = ["yield_amount", "pred_yield", "residual", "residual_zscore"]
num_cols = [c for c in num_cols if c in df.columns]

summary = []
for col in num_cols:
    s0 = normal[col].dropna()
    s1 = anomaly[col].dropna()
    if len(s0) < 2 or len(s1) < 2:
        continue
    # Normality check (optional: use for choosing test)
    _, p_norm0 = stats.shapiro(s0.sample(min(5000, len(s0)))) if len(s0) > 3 else (0, 0)
    _, p_norm1 = stats.shapiro(s1.sample(min(5000, len(s1)))) if len(s1) > 3 else (0, 0)
    # t-test (difference in means)
    t_stat, p_ttest = stats.ttest_ind(s0, s1, equal_var=False)
    # Mann-Whitney (nonparametric)
    u_stat, p_mw = stats.mannwhitneyu(s0, s1, alternative="two-sided")
    summary.append({
        "variable": col,
        "mean_0": s0.mean(), "std_0": s0.std(), "median_0": s0.median(),
        "mean_1": s1.mean(), "std_1": s1.std(), "median_1": s1.median(),
        "mean_diff": s1.mean() - s0.mean(),
        "p_value_ttest": p_ttest,
        "p_value_mannwhitney": p_mw,
    })

sum_df = pd.DataFrame(summary)
display(sum_df)

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    ax = axes[i]
    sns.boxplot(data=df, x="anomaly_flag", y=col, ax=ax)
    ax.set_title(col)
    ax.set_xticklabels(["Normal (0)", "Anomaly (1)"])
plt.suptitle("Numeric variables: Normal (0) vs Anomaly (1)", fontsize=12, y=1.02)
plt.tight_layout()
plt.show()

# COMMAND ----------

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    ax = axes[i]
    for flag, label in [(0, "Normal"), (1, "Anomaly")]:
        sub = df[df["anomaly_flag"] == flag][col].dropna()
        ax.hist(sub, bins=30, alpha=0.5, label=label, density=True)
    ax.set_title(col)
    ax.legend()
plt.suptitle("Distribution comparison: Normal vs Anomaly", fontsize=12, y=1.02)
plt.tight_layout()
plt.show()

# COMMAND ----------

cat_cols = ["commodity", "state_abbr", "irrigation"]
cat_cols = [c for c in cat_cols if c in df.columns]

for col in cat_cols:
    print(f"\n=== {col} ===")
    tab = pd.crosstab(df[col], df["anomaly_flag"], margins=True)
    tab_pct = pd.crosstab(df[col], df["anomaly_flag"], normalize="index") * 100
    tab_pct.columns = ["% Normal", "% Anomaly"]
    display(tab.join(tab_pct))
    # Chi-square test
    crosstab = pd.crosstab(df[col], df["anomaly_flag"])
    chi2, p_chi, dof, _ = stats.chi2_contingency(crosstab)
    print(f"Chi-square test: chi2={chi2:.2f}, p={p_chi:.4f}")

# COMMAND ----------

if "year" in df.columns:
    year_agg = df.groupby("year").agg(
        n=("anomaly_flag", "count"),
        n_anomaly=("anomaly_flag", "sum"),
    ).reset_index()
    year_agg["anomaly_pct"] = (year_agg["n_anomaly"] / year_agg["n"] * 100).round(2)
    display(year_agg)
    plt.figure(figsize=(10, 4))
    plt.bar(year_agg["year"], year_agg["anomaly_pct"], color="coral", alpha=0.8)
    plt.xlabel("Year")
    plt.ylabel("Anomaly %")
    plt.title("Anomaly rate by year")
    plt.tight_layout()
    plt.show()

# COMMAND ----------

print("=" * 60)
print("Summary: anomaly_flag 0 (Normal) vs 1 (Anomaly) statistical differences and phenomena")
print("=" * 60)

for _, row in sum_df.iterrows():
    sig = "***" if row["p_value_ttest"] < 0.05 else ""
    print(f"\n[{row['variable']}] {sig}")
    print(f"  Normal(0): mean={row['mean_0']:.3f}, std={row['std_0']:.3f}")
    print(f"  Anomaly(1): mean={row['mean_1']:.3f}, std={row['std_1']:.3f}")
    print(f"  Difference (1-0): {row['mean_diff']:.3f}")
    print(f"  p-value (t-test): {row['p_value_ttest']:.4f}")

print("\n" + "-" * 60)
print("Interpretation:")
print("- anomaly_flag=1 indicates cases with large deviation between actual (yield_amount) and predicted (pred_yield) values (based on residual/residual_zscore).")
print("- Cases with large absolute residual or residual_zscore are flagged as 1.")
print("- The above statistics and visualizations show the distribution differences between Normal (0) and Anomaly (1) groups for yield and residual variables.")