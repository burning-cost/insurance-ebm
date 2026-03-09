# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-ebm: Full Workflow Demo
# MAGIC
# MAGIC This notebook demonstrates the complete insurance-ebm workflow on synthetic
# MAGIC motor insurance data. It covers:
# MAGIC
# MAGIC 1. Generating synthetic data
# MAGIC 2. Fitting frequency and severity EBM models
# MAGIC 3. Extracting relativity tables
# MAGIC 4. Actuarial diagnostics: Gini, double-lift, A/E by segment
# MAGIC 5. Post-fit monotonicity enforcement
# MAGIC 6. GLM comparison

# COMMAND ----------
# MAGIC %pip install insurance-ebm[excel] --quiet

# COMMAND ----------

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from insurance_ebm import InsuranceEBM, RelativitiesTable, MonotonicityEditor, GLMComparison
from insurance_ebm import gini, lorenz_curve, double_lift, calibration_table

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Synthetic motor insurance data
# MAGIC
# MAGIC Generating a portfolio of 20,000 policies with realistic rating factors
# MAGIC and claim experience. The true model has:
# MAGIC - Young driver loading (age < 30)
# MAGIC - Vehicle group relativities
# MAGIC - NCD discount (linear on log scale)
# MAGIC - Area loading (urban vs rural)

# COMMAND ----------

rng = np.random.default_rng(2024)
N = 20_000

driver_age = rng.integers(17, 80, size=N).astype(float)
vehicle_group = rng.choice(["G1", "G2", "G3", "G4", "G5"], size=N,
                           p=[0.25, 0.28, 0.22, 0.15, 0.10])
area = rng.choice(["A", "B", "C", "D", "E"], size=N,
                  p=[0.30, 0.25, 0.20, 0.15, 0.10])
ncd = rng.integers(0, 6, size=N).astype(float)
vehicle_age = rng.integers(0, 21, size=N).astype(float)
exposure = rng.uniform(0.1, 1.0, size=N)

# True log frequency
log_freq = (
    -3.2
    + 0.025 * np.maximum(0, 30 - driver_age)       # young driver loading
    + np.where(vehicle_group == "G5", 0.45, 0.0)
    + np.where(vehicle_group == "G4", 0.30, 0.0)
    + np.where(vehicle_group == "G3", 0.15, 0.0)
    + np.where(area == "A", 0.20, 0.0)              # urban loading
    + np.where(area == "B", 0.10, 0.0)
    - 0.12 * ncd                                    # NCD discount
)
freq = np.exp(log_freq)
claim_count = rng.poisson(freq * exposure)

# Severity (Gamma distributed, weakly related to vehicle group)
base_sev = 2500.0
sev_shape = 2.0
sev_mult = np.where(vehicle_group == "G5", 1.3,
           np.where(vehicle_group == "G4", 1.15, 1.0))
severity = rng.gamma(
    shape=sev_shape,
    scale=base_sev * sev_mult / sev_shape,
    size=N,
)
claim_amount = np.where(claim_count > 0, severity, 0.0)

# Build DataFrames
X = pl.DataFrame({
    "driver_age": driver_age,
    "vehicle_group": vehicle_group,
    "area": area,
    "ncd": ncd,
    "vehicle_age": vehicle_age,
})

print(f"Portfolio: {N:,} policies, {claim_count.sum():,} claims")
print(f"Overall frequency: {(claim_count / exposure).mean():.4f}")
print(X.head())

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Train/test split and model fitting

# COMMAND ----------

split = int(0.75 * N)
X_train, X_test = X[:split], X[split:]
y_train = claim_count[:split]
y_test = claim_count[split:]
exp_train = exposure[:split]
exp_test = exposure[split:]

# Fit Poisson frequency model
freq_model = InsuranceEBM(
    loss="poisson",
    interactions="3x",
    random_state=42,
    n_estimators=200,
    learning_rate=0.05,
    max_bins=256,
)
freq_model.fit(X_train, y_train, exposure=exp_train)
print("Frequency model fitted.")
print(f"Features: {freq_model.feature_names}")

# COMMAND ----------

# Evaluate on hold-out
preds_test = freq_model.predict(X_test, exposure=exp_test)
g = gini(y_test, preds_test, exposure=exp_test)
print(f"Hold-out Gini: {g:.4f}")
print(f"Hold-out score (neg deviance): {freq_model.score(X_test, y_test, exposure=exp_test):.4f}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Relativity tables

# COMMAND ----------

rt = RelativitiesTable(freq_model)

# Summary of all features
print("Feature leverage summary:")
print(rt.summary())

# COMMAND ----------

# Driver age shape
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
rt.plot("driver_age", ax=axes[0])
rt.plot("ncd", kind="line", ax=axes[1])
rt.plot("vehicle_group", ax=axes[2])
fig.suptitle("EBM Shape Functions — Relativities", fontsize=13)
plt.tight_layout()
plt.savefig("/tmp/ebm_relativities.png", dpi=120)
display(plt.gcf())

# COMMAND ----------

# Vehicle group table
print("Vehicle group relativities:")
print(rt.table("vehicle_group"))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Actuarial diagnostics

# COMMAND ----------

# Lorenz curve
fig, ax = plt.subplots(figsize=(7, 6))
lorenz_curve(y_test, preds_test, exposure=exp_test, plot=True, ax=ax)
ax.set_title(f"Lorenz Curve (Gini = {g:.3f})")
plt.tight_layout()
display(plt.gcf())

# COMMAND ----------

# Double-lift chart
dl = double_lift(y_test, preds_test, exposure=exp_test, n_bands=10)
print("Double-lift by predicted decile:")
print(dl)

fig, ax = plt.subplots(figsize=(10, 5))
x = dl["band"].to_list()
ax.bar([i - 0.2 for i in x], dl["actual"].to_list(), width=0.4, label="Actual", alpha=0.8, color="#1f77b4")
ax.bar([i + 0.2 for i in x], dl["predicted"].to_list(), width=0.4, label="Predicted", alpha=0.8, color="#ff7f0e")
ax.set_xlabel("Risk decile (1 = lowest)")
ax.set_ylabel("Mean claim frequency")
ax.set_title("Double-lift chart")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
display(plt.gcf())

# COMMAND ----------

# A/E by area
preds_all = freq_model.predict(X, exposure=exposure)
cal = calibration_table(claim_count, preds_all, X["area"].to_numpy(), exposure=exposure)
print("A/E by area:")
print(cal)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Monotonicity enforcement
# MAGIC
# MAGIC NCD should be monotonically decreasing (more years = lower risk).
# MAGIC Enforce this post-fit if the shape function is not already monotone.

# COMMAND ----------

me = MonotonicityEditor(freq_model)
print(f"NCD monotone (decrease) before enforcement: {me.check('ncd', direction='decrease')}")

scores_before = me.get_scores("ncd")
me.enforce("ncd", direction="decrease")
print(f"NCD monotone (decrease) after enforcement:  {me.check('ncd', direction='decrease')}")

fig = me.plot_before_after("ncd", scores_before=scores_before, direction="decrease")
display(fig)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. GLM comparison
# MAGIC
# MAGIC Suppose we have an existing GLM with known vehicle group relativities.
# MAGIC Compare how the EBM's shape function compares.

# COMMAND ----------

# Hypothetical GLM relativities for vehicle group
glm_rel = pl.DataFrame({
    "level": ["G1", "G2", "G3", "G4", "G5"],
    "relativity": [1.0, 1.03, 1.18, 1.28, 1.40],
})

cmp = GLMComparison(freq_model)
comparison = cmp.compare_shapes("vehicle_group", glm_relativities=glm_rel)
print("EBM vs GLM — vehicle_group:")
print(comparison)

fig, ax = plt.subplots(figsize=(8, 5))
cmp.plot_comparison("vehicle_group", glm_relativities=glm_rel, ax=ax)
plt.tight_layout()
display(plt.gcf())

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. Excel export

# COMMAND ----------

rt.export_excel("/tmp/frequency_relativities.xlsx")
print("Relativity tables exported to /tmp/frequency_relativities.xlsx")
print("Sheets:", freq_model.feature_names)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Metric | Value |
# MAGIC |--------|-------|
# MAGIC | Hold-out Gini | {:.3f} |
# MAGIC | Features | {n} |
# MAGIC | Interactions detected | 3x |
# MAGIC
# MAGIC The EBM captures the young driver loading, vehicle group, NCD gradient,
# MAGIC and area loadings as interpretable shape functions — the same outputs you'd
# MAGIC expect from a GLM review, but with the flexibility of a boosted model.
""".format(g, n=len(freq_model.feature_names))

print("Demo complete.")
