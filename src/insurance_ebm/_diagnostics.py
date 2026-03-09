"""
Diagnostics — actuarial validation metrics for insurance pricing models.

These are the standard tools a UK actuarial review would expect to see when
validating a pricing model: Gini coefficient, Lorenz curve, double-lift chart,
deviance decomposition, residual analysis, and calibration by segment.

All functions accept numpy arrays or polars Series and work with or without
exposure weighting. Exposure weighting matters: an unweighted Gini on claim
count data is almost meaningless when policy exposure varies.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import polars as pl

# numpy >=2.0 removed np.trapz; use np.trapezoid with fallback
_trapz = getattr(np, "trapezoid", None) or np.trapz

from ._model import (
    InsuranceEBM,
    _deviance_gamma,
    _deviance_poisson,
    _deviance_tweedie,
    _ensure_array,
    _to_pandas,
)


def _coerce(x: Union[np.ndarray, pl.Series, list, None]) -> Optional[np.ndarray]:
    return _ensure_array(x)


# ---------------------------------------------------------------------------
# Gini coefficient
# ---------------------------------------------------------------------------

def gini(
    y_true: Union[np.ndarray, pl.Series, list],
    y_pred: Union[np.ndarray, pl.Series, list],
    exposure: Optional[Union[np.ndarray, pl.Series, list]] = None,
) -> float:
    """
    Normalised Gini coefficient for a predictive model.

    The Gini measures the ordering power of the model — how well it separates
    high-risk from low-risk policies. A perfect model has Gini = 1.0; a random
    model has Gini = 0.0. Values below 0 indicate a model that ranks risks
    inversely.

    The normalised Gini divides by the Gini of the oracle (perfect) model,
    so it is bounded [−1, 1].

    Parameters
    ----------
    y_true : array-like
        Observed outcomes (claim counts or amounts).
    y_pred : array-like
        Model predictions (expected values).
    exposure : array-like, optional
        Exposure weights. When provided, each policy's contribution to the
        Lorenz curve is weighted by its exposure. Always provide exposure
        for frequency models.

    Returns
    -------
    float
        Normalised Gini coefficient.
    """
    y_t = _coerce(y_true)
    y_p = _coerce(y_pred)
    w = _coerce(exposure)

    if w is None:
        w = np.ones_like(y_t)

    # Sort by predicted risk (ascending)
    order = np.argsort(y_p)
    y_t_sorted = y_t[order]
    w_sorted = w[order]

    # Weighted cumulative sums
    w_cum = np.cumsum(w_sorted)
    loss_cum = np.cumsum(y_t_sorted * w_sorted)

    w_total = w_cum[-1]
    loss_total = loss_cum[-1]

    if loss_total == 0:
        return 0.0

    # Gini of model predictions
    x = w_cum / w_total
    y = loss_cum / loss_total
    gini_model = 1.0 - 2.0 * float(_trapz(y, x))

    # Gini of oracle (sort by actual outcomes)
    order_oracle = np.argsort(y_t)
    y_t_oracle = y_t[order_oracle]
    w_oracle = w[order_oracle]
    wc = np.cumsum(w_oracle)
    lc = np.cumsum(y_t_oracle * w_oracle)
    x_o = wc / wc[-1]
    y_o = lc / lc[-1]
    gini_oracle = 1.0 - 2.0 * float(_trapz(y_o, x_o))

    if gini_oracle == 0:
        return 0.0

    return gini_model / gini_oracle


# ---------------------------------------------------------------------------
# Lorenz curve
# ---------------------------------------------------------------------------

def lorenz_curve(
    y_true: Union[np.ndarray, pl.Series, list],
    y_pred: Union[np.ndarray, pl.Series, list],
    exposure: Optional[Union[np.ndarray, pl.Series, list]] = None,
    plot: bool = False,
    ax=None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute (and optionally plot) the Lorenz curve for a model.

    Policies are sorted by predicted risk. The Lorenz curve shows the cumulative
    share of exposure on the x-axis against cumulative share of losses on the y-axis.

    Parameters
    ----------
    y_true : array-like
        Observed outcomes.
    y_pred : array-like
        Model predictions.
    exposure : array-like, optional
        Exposure weights.
    plot : bool
        If True, draw the Lorenz curve on ``ax`` or a new figure.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. Ignored if plot=False.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        (frac_exposure, frac_losses) — both start at 0 and end at 1.
    """
    y_t = _coerce(y_true)
    y_p = _coerce(y_pred)
    w = _coerce(exposure)

    if w is None:
        w = np.ones_like(y_t)

    order = np.argsort(y_p)
    y_t_s = y_t[order]
    w_s = w[order]

    w_cum = np.concatenate([[0], np.cumsum(w_s)])
    l_cum = np.concatenate([[0], np.cumsum(y_t_s * w_s)])

    frac_exp = w_cum / w_cum[-1]
    frac_loss = l_cum / l_cum[-1] if l_cum[-1] > 0 else l_cum

    if plot:
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 6))

        ax.plot(frac_exp, frac_loss, label="Model", color="#1f77b4", linewidth=2)
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random")
        ax.fill_between(frac_exp, frac_exp, frac_loss, alpha=0.12, color="#1f77b4")
        ax.set_xlabel("Cumulative share of exposure")
        ax.set_ylabel("Cumulative share of losses")
        ax.set_title("Lorenz Curve")
        ax.legend()
        ax.grid(alpha=0.3)

    return frac_exp, frac_loss


# ---------------------------------------------------------------------------
# Double-lift chart
# ---------------------------------------------------------------------------

def double_lift(
    y_true: Union[np.ndarray, pl.Series, list],
    y_pred: Union[np.ndarray, pl.Series, list],
    exposure: Optional[Union[np.ndarray, pl.Series, list]] = None,
    n_bands: int = 10,
) -> pl.DataFrame:
    """
    Double-lift chart: actual/expected by predicted risk decile.

    Policies are sorted by predicted value and grouped into n_bands equal-weight
    buckets. Within each bucket, the table shows mean actual, mean predicted, and
    the A/E ratio. A well-calibrated model has A/E close to 1.0 across all bands.

    Systematic deviations indicate model bias: monotone deviations suggest the
    model is over/under-dispersed; U-shaped patterns suggest missing interactions.

    Parameters
    ----------
    y_true : array-like
        Observed outcomes.
    y_pred : array-like
        Model predictions.
    exposure : array-like, optional
        Exposure weights. When provided, bands are equal-exposure buckets.
    n_bands : int
        Number of bands (default 10 for deciles).

    Returns
    -------
    polars.DataFrame
        Columns: band (int), exposure (float), actual (float), predicted (float),
        ae_ratio (float).
    """
    y_t = _coerce(y_true)
    y_p = _coerce(y_pred)
    w = _coerce(exposure)

    if w is None:
        w = np.ones_like(y_t)

    # Sort by predicted value
    order = np.argsort(y_p)
    y_t_s = y_t[order]
    y_p_s = y_p[order]
    w_s = w[order]

    # Assign equal-weight bands
    w_cum = np.cumsum(w_s)
    band_edges = np.linspace(0, w_cum[-1], n_bands + 1)
    band_ids = np.searchsorted(w_cum, band_edges[1:], side="left")
    band_ids = np.clip(band_ids, 0, len(w_s) - 1)

    rows = []
    prev = 0
    for b_idx, end in enumerate(band_ids):
        end = int(end) + 1
        sl = slice(prev, end)
        w_sl = w_s[sl]
        total_w = float(np.sum(w_sl))
        actual = float(np.sum(y_t_s[sl] * w_sl)) / total_w if total_w > 0 else float("nan")
        predicted = float(np.sum(y_p_s[sl] * w_sl)) / total_w if total_w > 0 else float("nan")
        ae = actual / predicted if predicted and predicted != 0 else float("nan")
        rows.append(
            {
                "band": b_idx + 1,
                "exposure": total_w,
                "actual": actual,
                "predicted": predicted,
                "ae_ratio": ae,
            }
        )
        prev = end

    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Deviance
# ---------------------------------------------------------------------------

def deviance(
    y_true: Union[np.ndarray, pl.Series, list],
    y_pred: Union[np.ndarray, pl.Series, list],
    exposure: Optional[Union[np.ndarray, pl.Series, list]] = None,
    family: str = "poisson",
    variance_power: float = 1.5,
) -> float:
    """
    Mean deviance for a GLM family.

    Parameters
    ----------
    y_true : array-like
        Observed outcomes.
    y_pred : array-like
        Predicted values (on the response scale, not log scale).
    exposure : array-like, optional
        Exposure weights used in the weighted mean deviance.
    family : str
        One of 'poisson', 'gamma', 'tweedie'. Default 'poisson'.
    variance_power : float
        Tweedie variance power. Only used when family='tweedie'.

    Returns
    -------
    float
        Mean deviance (lower is better).
    """
    y_t = _coerce(y_true)
    y_p = _coerce(y_pred)
    w = _coerce(exposure)

    if family == "poisson":
        return _deviance_poisson(y_t, y_p, w)
    elif family == "gamma":
        return _deviance_gamma(y_t, y_p, w)
    elif family == "tweedie":
        return _deviance_tweedie(y_t, y_p, variance_power, w)
    else:
        raise ValueError(f"family must be 'poisson', 'gamma', or 'tweedie', got '{family}'")


# ---------------------------------------------------------------------------
# Residual plot
# ---------------------------------------------------------------------------

def residual_plot(
    model: InsuranceEBM,
    X: Union["pl.DataFrame", "pd.DataFrame"],
    y: Union[np.ndarray, pl.Series, list],
    feature: str,
    exposure: Optional[Union[np.ndarray, pl.Series, list]] = None,
    n_bins: int = 20,
    ax=None,
):
    """
    Deviance residuals by feature bin.

    Groups observations by feature value (or quantile bin for continuous features)
    and plots mean deviance residual per bin. Useful for identifying features that
    the model is systematically mis-estimating.

    Parameters
    ----------
    model : InsuranceEBM
        Fitted model.
    X : DataFrame
        Feature matrix.
    y : array-like
        Observed outcomes.
    feature : str
        Feature to bin on (must be a column in X).
    exposure : array-like, optional
        Exposure weights.
    n_bins : int
        Number of quantile bins for continuous features. Ignored for categoricals.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    model._check_fitted()

    if isinstance(X, pl.DataFrame):
        X_pl = X
    else:
        X_pl = pl.from_pandas(X)

    y_arr = _coerce(y)
    w = _coerce(exposure)
    y_pred = model.predict(X_pl, exposure=exposure)

    # Compute per-observation Poisson-style deviance residuals (signed)
    eps = 1e-10
    y_safe = np.maximum(y_arr, 0.0)
    y_pred_safe = np.maximum(y_pred, eps)
    sign = np.where(y_safe >= y_pred_safe, 1.0, -1.0)
    unit_dev = 2.0 * (
        np.where(y_safe > 0, y_safe * np.log(y_safe / y_pred_safe), 0.0)
        - (y_safe - y_pred_safe)
    )
    residuals = sign * np.sqrt(np.maximum(unit_dev, 0.0))

    # Bin the feature
    col = X_pl[feature]
    if col.dtype in (pl.Utf8, pl.Categorical, pl.Boolean):
        # Categorical: group by value
        vals = col.cast(pl.Utf8).to_numpy()
        unique_vals = np.unique(vals)
        bin_labels = unique_vals
        assignments = np.searchsorted(unique_vals, vals)
    else:
        # Numeric: quantile bins
        col_np = col.cast(pl.Float64).to_numpy()
        quantiles = np.percentile(col_np, np.linspace(0, 100, n_bins + 1))
        quantiles = np.unique(quantiles)
        assignments = np.digitize(col_np, quantiles[1:-1])
        bin_labels = [f"Q{i+1}" for i in range(len(quantiles) - 1)]

    n_groups = len(bin_labels)
    mean_resid = np.zeros(n_groups)
    for i in range(n_groups):
        mask = assignments == i
        if mask.sum() == 0:
            mean_resid[i] = 0.0
        elif w is not None:
            mean_resid[i] = float(np.average(residuals[mask], weights=w[mask]))
        else:
            mean_resid[i] = float(np.mean(residuals[mask]))

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(8, n_groups * 0.5), 5))

    colours = ["#d62728" if r > 0 else "#2ca02c" for r in mean_resid]
    ax.bar(range(n_groups), mean_resid, color=colours, alpha=0.8, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(n_groups))
    ax.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean deviance residual (signed)")
    ax.set_title(f"Residuals by {feature}")
    ax.grid(axis="y", alpha=0.3)

    return ax


# ---------------------------------------------------------------------------
# Calibration table
# ---------------------------------------------------------------------------

def calibration_table(
    y_true: Union[np.ndarray, pl.Series, list],
    y_pred: Union[np.ndarray, pl.Series, list],
    segment: Union[np.ndarray, pl.Series, list],
    exposure: Optional[Union[np.ndarray, pl.Series, list]] = None,
) -> pl.DataFrame:
    """
    Actual vs expected by segment.

    Standard actuarial A/E table. For each segment (e.g. area, vehicle group,
    risk band), shows total exposure, total actual claims, total predicted claims,
    and the A/E ratio.

    Parameters
    ----------
    y_true : array-like
        Observed outcomes.
    y_pred : array-like
        Predicted values.
    segment : array-like
        Segment labels (strings or integers).
    exposure : array-like, optional
        Exposure weights. If None, each observation has weight 1.

    Returns
    -------
    polars.DataFrame
        Columns: segment, exposure, actual_total, predicted_total, ae_ratio.
        Sorted by ae_ratio descending (worst calibrated segments first).
    """
    y_t = _coerce(y_true)
    y_p = _coerce(y_pred)
    w = _coerce(exposure)
    seg = _coerce(segment) if not isinstance(segment, (list, np.ndarray)) else np.asarray(segment)

    if isinstance(segment, pl.Series):
        seg = segment.to_numpy()
    else:
        seg = np.asarray(segment)

    if w is None:
        w = np.ones_like(y_t)

    unique_segs = np.unique(seg)
    rows = []
    for s in unique_segs:
        mask = seg == s
        total_w = float(np.sum(w[mask]))
        actual = float(np.sum(y_t[mask] * w[mask]))
        predicted = float(np.sum(y_p[mask] * w[mask]))
        ae = actual / predicted if predicted > 0 else float("nan")
        rows.append(
            {
                "segment": str(s),
                "exposure": total_w,
                "actual_total": actual,
                "predicted_total": predicted,
                "ae_ratio": ae,
            }
        )

    return pl.DataFrame(rows).sort("ae_ratio", descending=True)
