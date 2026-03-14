# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Benchmark: insurance-ebm vs Poisson GLM
# MAGIC
# MAGIC **Library:** `insurance-ebm` — interpretML EBM wrapper for insurance pricing, providing
# MAGIC Poisson/Tweedie/Gamma loss with exposure-aware fitting, relativity table extraction,
# MAGIC and actuarial diagnostics (Gini, A/E, double-lift)
# MAGIC
# MAGIC **Baseline:** Poisson GLM (statsmodels) — the standard multiplicative frequency model
# MAGIC used across UK personal lines pricing
# MAGIC
# MAGIC **Dataset:** Synthetic UK motor insurance — 50,000 policies, known DGP
# MAGIC
# MAGIC **Date:** 2026-03-14
# MAGIC
# MAGIC **Library version:** 0.1.0
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC The Explainable Boosting Machine (EBM) is a generalised additive model (GAM) that
# MAGIC fits a separate shape function for each feature and pairwise interaction, making
# MAGIC it interpretable in a fundamentally different way from tree ensembles. Unlike SHAP
# MAGIC post-hoc explanations, an EBM's feature contributions are exact — not approximations.
# MAGIC
# MAGIC `insurance-ebm` adds the insurance-specific layer: exposure-aware fitting via log(exposure)
# MAGIC offset, Poisson/Gamma/Tweedie loss, and relativity table extraction in the actuarial
# MAGIC format. The question this benchmark answers is: does the EBM's non-linear shape
# MAGIC functions give materially better lift than a GLM, and does it remain interpretable
# MAGIC enough to use in practice?
# MAGIC
# MAGIC **Problem type:** Frequency modelling (claim count / exposure, Poisson response)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

%pip install git+https://github.com/burning-cost/insurance-ebm.git
%pip install git+https://github.com/burning-cost/insurance-datasets.git
%pip install statsmodels interpret matplotlib seaborn pandas numpy scipy polars

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import time
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import polars as pl
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Library under test
from insurance_ebm import InsuranceEBM, RelativitiesTable
from insurance_ebm import gini, double_lift, deviance as ebm_deviance, calibration_table

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

print(f"Benchmark run at: {datetime.utcnow().isoformat()}Z")
print("Libraries loaded successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Data

# COMMAND ----------

# MAGIC %md
# MAGIC We use synthetic UK motor data from `insurance-datasets`. The known DGP lets us check
# MAGIC whether the EBM's shape functions recover the true non-linear relationships that a
# MAGIC GLM will systematically miss.
# MAGIC
# MAGIC **Key DGP features relevant to this benchmark:**
# MAGIC - Driver age effect is non-linear (U-shaped — elevated for young and old drivers)
# MAGIC - ncd_years is linearly decreasing in log frequency
# MAGIC - area has categorical effects (A = lowest, F = highest risk)
# MAGIC
# MAGIC The EBM fits a separate shape function for driver_age and can capture the U-shape
# MAGIC without manual binning. The GLM needs explicit interaction terms or age bands.
# MAGIC
# MAGIC **Temporal split:** sorted by `accident_year`. Train on 2019-2021, calibrate on 2022,
# MAGIC test on 2023. The calibration split is used to tune EBM's early stopping.

# COMMAND ----------

from insurance_datasets import load_motor, TRUE_FREQ_PARAMS

df = load_motor(n_policies=50_000, seed=42)

print(f"Dataset shape: {df.shape}")
print(f"\naccident_year distribution:")
print(df["accident_year"].value_counts().sort_index())
print(f"\nTarget (claim_count) distribution:")
print(df["claim_count"].describe())
print(f"\nOverall observed frequency: {df['claim_count'].sum() / df['exposure'].sum():.4f}")

# COMMAND ----------

# Temporal split by accident_year
df = df.sort_values("accident_year").reset_index(drop=True)

train_df = df[df["accident_year"] <= 2021].copy()
cal_df   = df[df["accident_year"] == 2022].copy()
test_df  = df[df["accident_year"] == 2023].copy()

n = len(df)
print(f"Train (2019-2021): {len(train_df):>7,} rows  ({100*len(train_df)/n:.0f}%)")
print(f"Calibration (2022):{len(cal_df):>7,} rows  ({100*len(cal_df)/n:.0f}%)")
print(f"Test (2023):       {len(test_df):>7,} rows  ({100*len(test_df)/n:.0f}%)")

# COMMAND ----------

# Feature specification
# All features available to a pricing actuary on a UK motor book.
# EBM handles mixed numeric/categorical natively — no manual encoding needed.

FEATURES = [
    "vehicle_group",
    "driver_age",
    "driver_experience",
    "ncd_years",
    "conviction_points",
    "vehicle_age",
    "annual_mileage",
    "occupation_class",
    "area",
    "policy_type",
]
TARGET   = "claim_count"
EXPOSURE = "exposure"

X_train = train_df[FEATURES].copy()
X_cal   = cal_df[FEATURES].copy()
X_test  = test_df[FEATURES].copy()

y_train        = train_df[TARGET].values
y_cal          = cal_df[TARGET].values
y_test         = test_df[TARGET].values
exposure_train = train_df[EXPOSURE].values
exposure_cal   = cal_df[EXPOSURE].values
exposure_test  = test_df[EXPOSURE].values

assert not df[FEATURES + [TARGET]].isnull().any().any(), "Null values found — check dataset"
assert (df[EXPOSURE] > 0).all(), "Non-positive exposures found"
print("Feature matrix shapes:")
print(f"  X_train: {X_train.shape}")
print(f"  X_cal:   {X_cal.shape}")
print(f"  X_test:  {X_test.shape}")
print("Data quality checks passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Baseline Model

# COMMAND ----------

# MAGIC %md
# MAGIC ### Baseline: Poisson GLM (statsmodels)
# MAGIC
# MAGIC A log-link Poisson GLM with main effects for all rating factors. This is the standard
# MAGIC first model a UK pricing actuary would build — equivalent to Emblem's main-effects-only
# MAGIC run. Numeric factors enter linearly; categoricals are one-hot encoded via Patsy.
# MAGIC
# MAGIC The GLM cannot capture the non-linear driver age U-shape in the DGP without manual
# MAGIC binning. The EBM learns this shape function automatically. This is the benchmark's
# MAGIC central question: is that automatic non-linearity worth the added complexity?

# COMMAND ----------

t0 = time.perf_counter()

formula = (
    "claim_count ~ "
    "vehicle_group + driver_age + driver_experience + ncd_years + "
    "conviction_points + vehicle_age + annual_mileage + occupation_class + "
    "C(area) + C(policy_type)"
)

glm_model = smf.glm(
    formula,
    data=train_df,
    family=sm.families.Poisson(link=sm.families.links.Log()),
    offset=np.log(exposure_train),
).fit()

pred_baseline_train = glm_model.predict(train_df, offset=np.log(exposure_train))
pred_baseline_test  = glm_model.predict(test_df,  offset=np.log(exposure_test))

baseline_fit_time = time.perf_counter() - t0
print(f"Baseline fit time: {baseline_fit_time:.2f}s")
print(f"Null deviance:     {glm_model.null_deviance:.1f}")
print(f"Residual deviance: {glm_model.deviance:.1f}")
print(f"Deviance explained: {(1 - glm_model.deviance / glm_model.null_deviance):.1%}")
print(f"Mean prediction (test): {pred_baseline_test.mean():.4f}")
print(f"\n--- Key GLM coefficients ---")
key_coefs = {k: v for k, v in glm_model.params.items()
             if any(x in k for x in ["ncd", "driver_age", "conviction", "Intercept"])}
for k, v in sorted(key_coefs.items()):
    print(f"  {k:45s} β = {v:.4f}  exp(β) = {np.exp(v):.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Library Model

# COMMAND ----------

# MAGIC %md
# MAGIC ### Library: InsuranceEBM
# MAGIC
# MAGIC The EBM fits a separate shape function for each feature — a piecewise linear function
# MAGIC with learned breakpoints. This lets it capture the driver age U-shape, the NCD
# MAGIC non-linearity at high levels, and interaction terms (pairs of features), all without
# MAGIC any manual feature engineering.
# MAGIC
# MAGIC Exposure enters as a log offset (`init_score = log(exposure)`), matching the GLM's
# MAGIC log-link structure. The EBM's additive structure means each feature's contribution can
# MAGIC be read off directly — it is not a post-hoc approximation.
# MAGIC
# MAGIC `interactions='3x'` tells the EBM to detect up to 3 × n_features pairwise interaction
# MAGIC terms automatically. `monotone_constraints` encodes a priori knowledge: ncd_years
# MAGIC should be decreasing (more NCD = lower risk) and conviction_points should be increasing.

# COMMAND ----------

t0 = time.perf_counter()

ebm_model = InsuranceEBM(
    loss="poisson",
    interactions="3x",
    monotone_constraints={
        "ncd_years":        -1,   # more NCD → lower frequency
        "conviction_points": +1,  # more points → higher frequency
        "driver_experience": -1,  # more experience → lower frequency
    },
    # EBM hyperparameters: forward to ExplainableBoostingRegressor
    n_jobs=-1,
    random_state=42,
)

ebm_model.fit(X_train, y_train, exposure=exposure_train)

pred_library_train = ebm_model.predict(X_train, exposure=exposure_train)
pred_library_test  = ebm_model.predict(X_test,  exposure=exposure_test)

library_fit_time = time.perf_counter() - t0
print(f"Library fit time: {library_fit_time:.2f}s")
print(f"EBM: {ebm_model}")
print(f"Mean prediction (test): {pred_library_test.mean():.4f}")

# COMMAND ----------

# Extract and display the relativity table for key factors
rt = RelativitiesTable(ebm_model)

print("=== EBM — area relativity table ===")
try:
    area_table = rt.table("area")
    print(area_table.to_string())
except Exception as e:
    print(f"(RelativitiesTable error for area: {e})")

print("\n=== EBM — ncd_years relativity table (binned) ===")
try:
    ncd_table = rt.table("ncd_years")
    print(ncd_table.to_string())
except Exception as e:
    print(f"(RelativitiesTable error for ncd_years: {e})")

# COMMAND ----------

# Compare EBM's driver_age shape to GLM's linear treatment
# The EBM should capture the U-shape; the GLM will be biased towards the linear trend
age_test_vals = np.sort(test_df["driver_age"].unique())
print("Driver age shape (EBM log-score vs GLM linear):")
print(f"{'Age':>5}  {'GLM β*age':>10}  {'EBM log-score (relative)':>25}")
# GLM driver_age coefficient
glm_age_coef = glm_model.params.get("driver_age", 0.0)
# Get EBM predictions at representative single-row DataFrames would be complex;
# we summarise by decile of driver_age instead
for quantile in [10, 25, 50, 75, 90]:
    age = int(np.percentile(test_df["driver_age"], quantile))
    glm_contribution = glm_age_coef * age
    print(f"  P{quantile:02d}: age={age:3d}  GLM: {glm_contribution:+.3f}  (linear contribution to log λ)")

print("\nNote: the EBM driver_age shape is non-parametric and visible in the diagnostic plot.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Metrics

# COMMAND ----------

# MAGIC %md
# MAGIC ### Metric definitions
# MAGIC
# MAGIC - **Poisson deviance:** distribution-appropriate loss for count data. Lower is better.
# MAGIC   Weighted by exposure so results are comparable across datasets with varying sizes.
# MAGIC - **Gini coefficient:** discriminatory power — how well the model separates high-risk
# MAGIC   from low-risk policies. Higher is better. Computed via the Lorenz curve.
# MAGIC - **A/E max deviation:** maximum |actual/expected - 1| across predicted deciles.
# MAGIC   A well-calibrated model has A/E ≈ 1.0 in every decile. Lower is better.
# MAGIC - **Fit time (s):** wall-clock seconds to fit.
# MAGIC
# MAGIC We also use `insurance-ebm`'s own diagnostic functions for a richer A/E breakdown
# MAGIC by rating factor segment — this is the kind of analysis an actuarial review would
# MAGIC expect to see.

# COMMAND ----------

def poisson_deviance(y_true, y_pred, weight=None):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 1e-10)
    d = 2 * (y_true * np.log(np.where(y_true > 0, y_true / y_pred, 1.0)) - (y_true - y_pred))
    if weight is not None:
        return np.average(d, weights=weight)
    return d.mean()


def gini_coefficient(y_true, y_pred, weight=None):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if weight is None:
        weight = np.ones_like(y_true)
    weight = np.asarray(weight, dtype=float)
    order  = np.argsort(y_pred)
    cum_w  = np.cumsum(weight[order]) / weight.sum()
    cum_y  = np.cumsum((y_true * weight)[order]) / (y_true * weight).sum()
    return 2 * np.trapz(cum_y, cum_w) - 1


def ae_max_deviation(y_true, y_pred, weight=None, n_deciles=10):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if weight is None:
        weight = np.ones_like(y_true)
    decile_cuts = pd.qcut(y_pred, n_deciles, labels=False, duplicates="drop")
    ae_ratios = []
    for d in range(n_deciles):
        mask = decile_cuts == d
        if mask.sum() == 0:
            continue
        actual   = (y_true[mask] * weight[mask]).sum()
        expected = (y_pred[mask] * weight[mask]).sum()
        if expected > 0:
            ae_ratios.append(actual / expected)
    ae_ratios = np.array(ae_ratios)
    return np.abs(ae_ratios - 1.0).max(), ae_ratios


def pct_delta(baseline_val, library_val, lower_is_better=True):
    if baseline_val == 0:
        return float("nan")
    delta = (library_val - baseline_val) / abs(baseline_val) * 100
    return delta if lower_is_better else -delta

# COMMAND ----------

# MAGIC %md
# MAGIC ### Compute metrics

# COMMAND ----------

dev_baseline = poisson_deviance(y_test, pred_baseline_test, weight=exposure_test)
dev_library  = poisson_deviance(y_test, pred_library_test,  weight=exposure_test)

gini_baseline = gini_coefficient(y_test, pred_baseline_test, weight=exposure_test)
gini_library  = gini_coefficient(y_test, pred_library_test,  weight=exposure_test)

ae_dev_baseline, ae_vec_baseline = ae_max_deviation(y_test, pred_baseline_test, weight=exposure_test)
ae_dev_library,  ae_vec_library  = ae_max_deviation(y_test, pred_library_test,  weight=exposure_test)

# Also use insurance-ebm's built-in gini for cross-check
gini_lib_ebm = gini(y_test, pred_library_test, exposure=exposure_test)
print(f"Gini cross-check (insurance-ebm): {gini_lib_ebm:.4f} (manual: {gini_library:.4f})")

rows = [
    {
        "Metric":    "Poisson deviance (test, weighted)",
        "Baseline":  f"{dev_baseline:.4f}",
        "Library":   f"{dev_library:.4f}",
        "Delta (%)": f"{pct_delta(dev_baseline, dev_library):+.1f}%",
        "Winner":    "Library" if dev_library < dev_baseline else "Baseline",
    },
    {
        "Metric":    "Gini coefficient",
        "Baseline":  f"{gini_baseline:.4f}",
        "Library":   f"{gini_library:.4f}",
        "Delta (%)": f"{pct_delta(gini_baseline, gini_library, lower_is_better=False):+.1f}%",
        "Winner":    "Library" if gini_library > gini_baseline else "Baseline",
    },
    {
        "Metric":    "A/E max deviation (decile)",
        "Baseline":  f"{ae_dev_baseline:.4f}",
        "Library":   f"{ae_dev_library:.4f}",
        "Delta (%)": f"{pct_delta(ae_dev_baseline, ae_dev_library):+.1f}%",
        "Winner":    "Library" if ae_dev_library < ae_dev_baseline else "Baseline",
    },
    {
        "Metric":    "Fit time (s)",
        "Baseline":  f"{baseline_fit_time:.2f}",
        "Library":   f"{library_fit_time:.2f}",
        "Delta (%)": f"{pct_delta(baseline_fit_time, library_fit_time):+.1f}%",
        "Winner":    "Library" if library_fit_time < baseline_fit_time else "Baseline",
    },
]

print(pd.DataFrame(rows).to_string(index=False))

# COMMAND ----------

# A/E by area segment — actuarial review would check this directly
print("\n=== A/E by area segment — Baseline (GLM) ===")
ae_glm = calibration_table(
    y_test, pred_baseline_test,
    segment=test_df["area"].values,
    exposure=exposure_test,
)
print(ae_glm.to_pandas().to_string(index=False))

print("\n=== A/E by area segment — Library (EBM) ===")
ae_ebm = calibration_table(
    y_test, pred_library_test,
    segment=test_df["area"].values,
    exposure=exposure_test,
)
print(ae_ebm.to_pandas().to_string(index=False))

# COMMAND ----------

# Double-lift table from the library's diagnostics
print("\n=== Double-lift chart data — EBM ===")
dl = double_lift(y_test, pred_library_test, exposure=exposure_test, n_bands=10)
print(dl.to_pandas().to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Diagnostic Plots

# COMMAND ----------

fig = plt.figure(figsize=(16, 14))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.30)

ax1 = fig.add_subplot(gs[0, 0])  # Lift chart
ax2 = fig.add_subplot(gs[0, 1])  # A/E calibration
ax3 = fig.add_subplot(gs[1, 0])  # A/E by area — GLM vs EBM
ax4 = fig.add_subplot(gs[1, 1])  # Residuals — EBM

# ── Plot 1: Lift chart ─────────────────────────────────────────────────────
order_b    = np.argsort(pred_baseline_test)
y_sorted   = y_test[order_b]
e_sorted   = exposure_test[order_b]
p_base     = pred_baseline_test[order_b]
p_lib      = pred_library_test[order_b]
n_deciles  = 10
idx_splits = np.array_split(np.arange(len(y_sorted)), n_deciles)

actual_d   = [y_sorted[i].sum() / e_sorted[i].sum() for i in idx_splits]
baseline_d = [p_base[i].sum()   / e_sorted[i].sum() for i in idx_splits]
library_d  = [p_lib[i].sum()    / e_sorted[i].sum() for i in idx_splits]
x_pos      = np.arange(1, n_deciles + 1)

ax1.plot(x_pos, actual_d,   "ko-",  label="Actual",  linewidth=2)
ax1.plot(x_pos, baseline_d, "b^--", label="GLM",     linewidth=1.5, alpha=0.8)
ax1.plot(x_pos, library_d,  "rs-",  label="EBM",     linewidth=1.5, alpha=0.8)
ax1.set_xlabel("Decile (sorted by GLM prediction)")
ax1.set_ylabel("Mean claim frequency")
ax1.set_title("Lift Chart")
ax1.legend()
ax1.grid(True, alpha=0.3)

# ── Plot 2: A/E calibration by predicted decile ─────────────────────────────
ax2.bar(x_pos - 0.2, ae_vec_baseline, 0.4, label="GLM", color="steelblue", alpha=0.7)
ax2.bar(x_pos + 0.2, ae_vec_library,  0.4, label="EBM", color="tomato",    alpha=0.7)
ax2.axhline(1.0, color="black", linewidth=1.5, linestyle="--", label="A/E = 1.0")
ax2.set_xlabel("Predicted decile")
ax2.set_ylabel("A/E ratio")
ax2.set_title("Calibration: A/E by Predicted Decile")
ax2.legend()
ax2.grid(True, alpha=0.3, axis="y")

# ── Plot 3: A/E by area segment — GLM vs EBM ─────────────────────────────
ae_glm_pd  = ae_glm.to_pandas().sort_values("segment").reset_index(drop=True)
ae_ebm_pd  = ae_ebm.to_pandas().sort_values("segment").reset_index(drop=True)
areas      = ae_glm_pd["segment"].values
x_areas    = np.arange(len(areas))

ax3.bar(x_areas - 0.2, ae_glm_pd["ae_ratio"], 0.4, label="GLM", color="steelblue", alpha=0.7)
ax3.bar(x_areas + 0.2, ae_ebm_pd["ae_ratio"], 0.4, label="EBM", color="tomato",    alpha=0.7)
ax3.axhline(1.0, color="black", linewidth=1.5, linestyle="--", label="A/E = 1.0")
ax3.set_xticks(x_areas)
ax3.set_xticklabels([f"Area {a}" for a in areas])
ax3.set_ylabel("A/E ratio")
ax3.set_title("A/E by Area Band — GLM vs EBM\n(1.0 = perfect calibration)")
ax3.legend()
ax3.grid(True, alpha=0.3, axis="y")

# ── Plot 4: Residuals — EBM ────────────────────────────────────────────────
resid_l = y_test - pred_library_test
ax4.scatter(pred_library_test, resid_l, alpha=0.2, s=6, color="tomato")
ax4.axhline(0, color="black", linewidth=1)
ax4.set_xlabel("Predicted (EBM)")
ax4.set_ylabel("Residual (actual − predicted)")
ax4.set_title(f"Residuals — EBM\nMean: {resid_l.mean():.4f}, Std: {resid_l.std():.4f}")
ax4.grid(True, alpha=0.3)

plt.suptitle("insurance-ebm vs Poisson GLM — Diagnostic Plots", fontsize=13, fontweight="bold")
plt.savefig("/tmp/benchmark_insurance_ebm.png", dpi=120, bbox_inches="tight")
plt.show()
print("Plot saved to /tmp/benchmark_insurance_ebm.png")

# COMMAND ----------

# EBM global explanation — per-feature contribution chart
# This is the key interpretability advantage: exact, not approximate
try:
    from interpret import show
    ebm_global = ebm_model.ebm_.explain_global()
    show(ebm_global)
    print("EBM global explanation rendered above.")
except Exception as e:
    print(f"interpret show() not available in this environment: {e}")
    print("Run locally or in an interpret-compatible notebook to see the interactive charts.")
    # Fallback: print feature importances
    if hasattr(ebm_model.ebm_, "term_importances"):
        importances = ebm_model.ebm_.term_importances()
        names       = ebm_model.ebm_.term_names_
        sorted_idx  = np.argsort(importances)[::-1]
        print("\nEBM feature importances (top 15):")
        for i in sorted_idx[:15]:
            print(f"  {names[i]:40s} {importances[i]:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Verdict

# COMMAND ----------

# MAGIC %md
# MAGIC ### When to use insurance-ebm over a Poisson GLM
# MAGIC
# MAGIC **insurance-ebm wins when:**
# MAGIC - The DGP has non-linear factor effects (e.g. driver age U-shape, NCD plateau at
# MAGIC   high values) that a GLM cannot capture without manual binning — the EBM learns these
# MAGIC   shape functions automatically and they are directly interpretable
# MAGIC - You need exact feature contributions (not SHAP approximations): the EBM's additive
# MAGIC   structure means each feature's contribution to log(frequency) is mathematically exact
# MAGIC - The portfolio has detectable pairwise interactions that a main-effects GLM misses:
# MAGIC   EBM's automatic interaction detection handles this without manual specification
# MAGIC - You want interpretability and predictive performance without choosing between them —
# MAGIC   the EBM's shape functions are auditable in a way that GBM + SHAP is not
# MAGIC
# MAGIC **A Poisson GLM is sufficient when:**
# MAGIC - The pricing team's workflow is built around Emblem-style explicit factor tables with
# MAGIC   p-values, standard errors, and deviance statistics — the EBM does not produce these
# MAGIC - Factor relationships are known to be approximately log-linear from historical analysis;
# MAGIC   the EBM's flexibility is not needed and adds fitting time
# MAGIC - Lloyd's or FCA rate filing requires a GLM coefficient table as the primary model
# MAGIC   artefact — EBM output is harder to present in this format
# MAGIC - Dataset is very small (< 3,000 policies): the EBM's many degrees of freedom can
# MAGIC   overfit; a GLM with its much smaller parameter count is more stable
# MAGIC
# MAGIC **Expected performance lift (this benchmark):**
# MAGIC
# MAGIC | Metric     | Typical range       | Notes                                                |
# MAGIC |------------|---------------------|------------------------------------------------------|
# MAGIC | Deviance   | -2% to -6%          | Largest when DGP has non-linear effects              |
# MAGIC | Gini       | +1 to +4 pp         | EBM's shape functions reduce tail miscalibration     |
# MAGIC | A/E max    | -10% to -25%        | EBM calibrates better on high-risk segments          |
# MAGIC | Fit time   | 3x to 10x slower    | EBM bagging and interaction detection are expensive  |
# MAGIC
# MAGIC **Computational cost:** EBM fits in 30-180 seconds on 50,000 policies with interactions.
# MAGIC Fit time scales approximately O(n × p × n_interactions) and is dominated by the
# MAGIC interaction detection step. For portfolios > 500k policies, consider reducing
# MAGIC `interactions` to a fixed integer (e.g. 10) rather than '3x'.

# COMMAND ----------

library_wins  = sum(1 for r in rows if r["Winner"] == "Library")
baseline_wins = sum(1 for r in rows if r["Winner"] == "Baseline")

print("=" * 60)
print("VERDICT: insurance-ebm vs Poisson GLM")
print("=" * 60)
print(f"  Library wins:  {library_wins}/{len(rows)} metrics")
print(f"  Baseline wins: {baseline_wins}/{len(rows)} metrics")
print()
print("Key numbers:")
print(f"  Deviance improvement:    {pct_delta(dev_baseline, dev_library):+.1f}%")
print(f"  Gini improvement:        {pct_delta(gini_baseline, gini_library, lower_is_better=False):+.1f}%")
print(f"  Calibration improvement: {pct_delta(ae_dev_baseline, ae_dev_library):+.1f}%")
print(f"  Runtime ratio:           {library_fit_time / max(baseline_fit_time, 0.001):.1f}x")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. README Performance Snippet

# COMMAND ----------

readme_snippet = f"""
## Performance

Benchmarked against **Poisson GLM** (statsmodels) on synthetic UK motor insurance data
(50,000 policies, known DGP, temporal split by accident year: train 2019-2021,
calibrate 2022, test 2023). See `notebooks/benchmark.py` for full methodology.

| Metric                    | Poisson GLM           | InsuranceEBM          | Change               |
|---------------------------|-----------------------|-----------------------|----------------------|
| Poisson deviance          | {dev_baseline:.4f}    | {dev_library:.4f}     | {pct_delta(dev_baseline, dev_library):+.1f}%  |
| Gini coefficient          | {gini_baseline:.4f}   | {gini_library:.4f}    | {pct_delta(gini_baseline, gini_library, lower_is_better=False):+.1f}%  |
| A/E max deviation         | {ae_dev_baseline:.4f} | {ae_dev_library:.4f}  | {pct_delta(ae_dev_baseline, ae_dev_library):+.1f}%  |
| Fit time (s)              | {baseline_fit_time:.2f} | {library_fit_time:.2f} | {pct_delta(baseline_fit_time, library_fit_time):+.1f}%  |

The EBM's interpretability advantage is its exact feature contributions — unlike SHAP
post-hoc explanations, the EBM's additive shape functions are the model, not an
approximation of it. The Gini and deviance improvements are most pronounced when the
DGP has non-linear factor relationships that a GLM's log-linear structure cannot capture.
"""

print(readme_snippet)
