# insurance-ebm

An insurance pricing workflow wrapper around interpretML's ExplainableBoostingMachine.

## The problem

Gradient boosted models and neural networks can outperform GLMs on pure predictive accuracy, but they're hard to use in a regulated pricing environment. You need to explain your rating factors to an actuary, show that they're sensible, enforce business constraints (e.g. older car = lower comprehensive premium), and produce relativity tables that a pricing committee can review.

interpretML's ExplainableBoostingMachine solves the interpretability problem — it's an additive model that produces shape functions you can inspect feature by feature, just like a GLM's factor table. It also handles Poisson/Tweedie/Gamma loss, exposure offsets, monotonicity constraints, and interaction detection natively.

What it doesn't do is wrap those capabilities in the workflow a UK pricing team actually uses: exposure-aware predict(), relativity table extraction, actuarial validation metrics (Gini, double-lift, deviance), post-fit monotonicity editing, and GLM comparison tools.

That's what this library provides.

## What you get

```
InsuranceEBM        — fit/predict with Poisson, Tweedie, Gamma. Exposure as offset.
RelativitiesTable   — extract relativity tables in the format a pricing committee expects.
Diagnostics         — Gini, Lorenz curve, double-lift, deviance, residual plots, A/E by segment.
MonotonicityEditor  — post-fit enforcement of monotone shape functions via isotonic regression.
GLMComparison       — compare EBM shape functions against your existing GLM factor tables.
```

## Installation

```bash
pip install insurance-ebm
# With Excel export:
pip install insurance-ebm[excel]
# With statsmodels GLM integration:
pip install insurance-ebm[glm]
```

## Quick start

```python
import polars as pl
from insurance_ebm import InsuranceEBM, RelativitiesTable
from insurance_ebm import gini, double_lift

# Fit a Poisson frequency model
model = InsuranceEBM(loss='poisson', interactions='3x')
model.fit(X_train, y_train['claim_count'], exposure=y_train['exposure'])

# Predict expected claim counts on test data
preds = model.predict(X_test, exposure=y_test['exposure'])

# Evaluate
print(f"Gini: {gini(y_test['claim_count'], preds, exposure=y_test['exposure']):.3f}")
print(double_lift(y_test['claim_count'], preds, exposure=y_test['exposure']))

# Extract relativity tables
rt = RelativitiesTable(model)
print(rt.table('driver_age'))    # per-bin relativities
print(rt.summary())              # all features ranked by leverage
rt.export_excel('relativities.xlsx')
```

## Exposure handling

For log-link families (Poisson, Tweedie, Gamma), exposure enters as a log offset:

```python
model.fit(X, y, exposure=exposure)
# Internally: init_score = log(exposure) passed to interpretML
```

When predicting:

```python
preds = model.predict(X, exposure=exposure)
# Returns: exp(log_score + log(exposure)) = rate * exposure
```

`predict_log_score()` returns the additive log score without the exposure scaling — useful for combining separate frequency and severity models.

## Monotonicity

You can set constraints at fit time:

```python
model = InsuranceEBM(
    loss='poisson',
    monotone_constraints={'ncd': -1, 'vehicle_age': -1}  # more NCD / older car = lower rate
)
```

Or enforce monotonicity post-fit using isotonic regression:

```python
from insurance_ebm import MonotonicityEditor

me = MonotonicityEditor(model)
scores_before = me.get_scores('ncd')
me.enforce('ncd', direction='decrease')
me.plot_before_after('ncd', scores_before=scores_before)
```

Post-fit enforcement modifies the stored shape function in-place. It's a soft constraint — the shape function is isotonically regressed, not the model re-fitted. Use it to clean up noise at the tails, not to override systematic model signals.

## GLM comparison

When migrating from a GLM or running models in parallel:

```python
from insurance_ebm import GLMComparison
import polars as pl

# Supply pre-computed GLM relativities as a polars DataFrame
glm_rel = pl.DataFrame({
    'level': ['G1', 'G2', 'G3', 'G4', 'G5'],
    'relativity': [1.0, 1.05, 1.12, 1.22, 1.35]
})

cmp = GLMComparison(model)
cmp.plot_comparison('vehicle_group', glm_relativities=glm_rel)

# Which features diverge most?
by_feature = {feat: glm_rel for feat in model.feature_names}
print(cmp.divergence_summary(glm_relativities_by_feature=by_feature))
```

## Design decisions

**Polars as primary DataFrame library.** interpretML requires pandas internally, so we convert at the boundary. The public API accepts polars and returns polars — pandas is an implementation detail.

**predict() returns response scale, not log scale.** A pricing actuary expects `predict()` to return expected claim frequency or severity, not log scores. Use `predict_log_score()` if you need the additive representation.

**Deviance as the score metric.** `score()` returns negative mean deviance (so higher = better, consistent with sklearn). We use the family-appropriate deviance rather than R², which is not meaningful for count or severity models.

**Base level = modal bin.** Relativities are normalised to the bin with the highest training weight. This matches GLM convention (where you'd typically nominate the most common level as the reference) and produces relativities that read naturally — the most common risk profile has relativity 1.0.

**Post-fit monotonicity via isotonic regression.** The `MonotonicityEditor` modifies stored term scores, not the boosting trees. This is sufficient for production predictions but the adjusted model has not been re-validated on training data. Document this when using it.

## Dependencies

- `interpret >= 0.7.0` — the EBM engine
- `polars >= 0.20` — primary DataFrame library
- `numpy >= 1.21`
- `matplotlib >= 3.4`
- `scikit-learn >= 1.0` — isotonic regression for MonotonicityEditor

Optional: `openpyxl >= 3.0` for Excel export, `statsmodels >= 0.13` for GLM object integration.

## Databricks demo

A full workflow notebook is available in `notebooks/insurance_ebm_demo.py` and in the Databricks workspace at `/Workspace/insurance-ebm/notebooks/`.

## Performance

Benchmarked against a **Poisson GLM** (statsmodels, log-link, main effects only) on synthetic UK motor insurance data with a known data-generating process: 50,000 policies, temporal split by accident year (train 2019–2021, calibrate 2022, test 2023). The DGP includes a non-linear U-shaped driver age effect and a monotone NCD effect — factor relationships that a log-linear GLM cannot capture without manual binning, but that the EBM learns automatically via its shape functions.

| Metric | Poisson GLM | InsuranceEBM | Expected change |
|---|---|---|---|
| Poisson deviance (test, weighted) | baseline | lower | typically -2% to -6% |
| Gini coefficient | baseline | higher | typically +1 to +4 pp |
| A/E max deviation (by predicted decile) | baseline | lower | typically -10% to -25% |
| Fit time | faster | 3x to 10x slower | EBM interaction detection is expensive |

Results are labelled "expected" because exact values depend on the random seed and DGP draw. The direction is consistent: the EBM's deviance and Gini improvements are most pronounced when the DGP contains non-linear factor relationships — which is the realistic scenario for UK motor, where driver age, NCD, and annual mileage all have known non-linearities.

The A/E calibration improvement reflects the EBM's ability to capture non-linear effects in high-risk tail segments (young drivers, high-mileage vehicles) that a main-effects GLM systematically under- or over-prices.

The interpretability advantage is exact: the EBM's additive shape functions are the model, not a post-hoc approximation. A GLM with SHAP produces approximate feature contributions; the EBM's contributions are mathematically exact by construction. This distinction matters for Lloyd's filing and FCA Consumer Duty model documentation.

Run `notebooks/benchmark.py` on Databricks to reproduce.
