"""
InsuranceEBM — interpretML EBM wrapper for insurance pricing workflows.

interpretML's ExplainableBoostingRegressor already handles Poisson/Tweedie/Gamma
loss, exposure via init_score, monotonicity constraints, interaction terms, and
bagging. This module adds the insurance-specific workflow layer: exposure-aware
fit/predict, polars DataFrame support, and deviance scoring.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
import polars as pl
from interpret.glassbox import ExplainableBoostingRegressor


_SUPPORTED_LOSSES = {"poisson", "tweedie", "gamma", "mse", "mae", "huber"}


def _to_pandas(X: Union[pl.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    """Convert polars or pandas DataFrame to pandas. Pass-through if already pandas."""
    if isinstance(X, pl.DataFrame):
        return X.to_pandas()
    if isinstance(X, pd.DataFrame):
        return X
    raise TypeError(f"X must be a polars or pandas DataFrame, got {type(X)}")


def _ensure_array(arr: Optional[Union[np.ndarray, list, pl.Series, pd.Series]]) -> Optional[np.ndarray]:
    """Coerce various array-like types to numpy 1-D array, or return None."""
    if arr is None:
        return None
    if isinstance(arr, (pl.Series,)):
        return arr.to_numpy()
    if isinstance(arr, pd.Series):
        return arr.to_numpy()
    return np.asarray(arr, dtype=float)


def _deviance_poisson(y_true: np.ndarray, y_pred: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """Unit Poisson deviance, optionally exposure-weighted."""
    # Guard against zeros
    eps = 1e-10
    y_true = np.maximum(y_true, 0.0)
    y_pred = np.maximum(y_pred, eps)
    d = 2.0 * (np.where(y_true > 0, y_true * np.log(y_true / y_pred), 0.0) - (y_true - y_pred))
    if weights is not None:
        return float(np.average(d, weights=weights))
    return float(np.mean(d))


def _deviance_gamma(y_true: np.ndarray, y_pred: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """Unit Gamma deviance, optionally exposure-weighted."""
    eps = 1e-10
    y_true = np.maximum(y_true, eps)
    y_pred = np.maximum(y_pred, eps)
    d = 2.0 * (-np.log(y_true / y_pred) + (y_true - y_pred) / y_pred)
    if weights is not None:
        return float(np.average(d, weights=weights))
    return float(np.mean(d))


def _deviance_tweedie(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    p: float,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Unit Tweedie deviance for power p, optionally exposure-weighted."""
    eps = 1e-10
    y_true = np.maximum(y_true, 0.0)
    y_pred = np.maximum(y_pred, eps)
    if p == 1:
        return _deviance_poisson(y_true, y_pred, weights)
    if p == 2:
        return _deviance_gamma(y_true, y_pred, weights)
    d = (
        2.0
        * (
            np.power(np.maximum(y_true, eps), 2 - p) / ((1 - p) * (2 - p))
            - y_true * np.power(y_pred, 1 - p) / (1 - p)
            + np.power(y_pred, 2 - p) / (2 - p)
        )
    )
    if weights is not None:
        return float(np.average(d, weights=weights))
    return float(np.mean(d))


class InsuranceEBM:
    """
    Explainable Boosting Machine configured for insurance pricing.

    Wraps interpretML's ExplainableBoostingRegressor and adds:
    - Exposure-aware fitting via log(exposure) init_score
    - Polars DataFrame input support
    - predict() returning expected values (frequency, severity, or pure premium)
      rather than raw log scores
    - Deviance scoring with the correct GLM family

    Parameters
    ----------
    loss : str
        Loss function. One of 'poisson' (default), 'tweedie', 'gamma', 'mse',
        'mae', 'huber'. Poisson is standard for claim frequency; Tweedie or
        Gamma for pure premium or severity.
    variance_power : float
        Tweedie variance power. Only used when loss='tweedie'. Set to 1.0 for
        Poisson, 2.0 for Gamma, 1.5 for compound Poisson-Gamma (pure premium).
    interactions : int or str
        Number of pairwise interaction terms, or '3x' to let the EBM pick
        approximately 3 * n_features terms via FAST interaction detection.
        Passed through to ExplainableBoostingRegressor as ``interactions``.
    monotone_constraints : dict, optional
        Mapping of {feature_name: direction} where direction is +1 (increasing)
        or -1 (decreasing). Applied at fit time via the interpretML API.
    **ebm_kwargs
        Any additional keyword arguments forwarded to ExplainableBoostingRegressor.
        Common overrides: n_estimators, learning_rate, max_bins, min_samples_leaf,
        random_state, n_jobs.

    Notes
    -----
    interpretML uses a log link for Poisson, Tweedie, and Gamma families.
    Exposure is incorporated as an offset: init_score = log(exposure). When no
    exposure is provided, all exposures are treated as 1 (no offset needed).
    """

    def __init__(
        self,
        loss: str = "poisson",
        variance_power: float = 1.5,
        interactions: Union[int, str] = "3x",
        monotone_constraints: Optional[dict] = None,
        **ebm_kwargs,
    ) -> None:
        if loss not in _SUPPORTED_LOSSES:
            raise ValueError(f"loss must be one of {_SUPPORTED_LOSSES}, got '{loss}'")
        self.loss = loss
        self.variance_power = variance_power
        self.interactions = interactions
        self.monotone_constraints = monotone_constraints or {}
        self.ebm_kwargs = ebm_kwargs
        self.ebm_: Optional[ExplainableBoostingRegressor] = None
        self._feature_names: Optional[list] = None

    def _build_ebm(self, feature_names: list) -> ExplainableBoostingRegressor:
        """Instantiate the underlying EBM with appropriate settings."""
        # Resolve interactions: '3x' means 3 * n_features
        if self.interactions == "3x":
            n_interactions = 3 * len(feature_names)
        else:
            n_interactions = int(self.interactions)

        # Map loss string to interpretML objective
        objective_map = {
            "poisson": "poisson_deviance",
            "tweedie": "tweedie_deviance",
            "gamma": "gamma_deviance",
            "mse": "rmse",
            "mae": "mae",
            "huber": "huber",
        }
        objective = objective_map[self.loss]

        # Tweedie variance power
        tweedie_power = self.variance_power if self.loss == "tweedie" else None

        # Build monotone constraints list aligned to feature order
        mc = []
        for fn in feature_names:
            mc.append(self.monotone_constraints.get(fn, 0))

        kwargs = dict(self.ebm_kwargs)
        kwargs.setdefault("random_state", 42)
        kwargs.setdefault("n_jobs", -1)

        ebm_args = {
            "interactions": n_interactions,
            "feature_names": feature_names,
            "feature_types": None,  # let interpretML infer
            "objective": objective,
            "monotone_constraints": mc if any(v != 0 for v in mc) else None,
        }
        if tweedie_power is not None:
            ebm_args["tweedie_exp_target"] = False  # y is not log-transformed

        ebm_args.update(kwargs)
        return ExplainableBoostingRegressor(**ebm_args)

    def fit(
        self,
        X: Union[pl.DataFrame, pd.DataFrame],
        y: Union[np.ndarray, list, pl.Series, pd.Series],
        exposure: Optional[Union[np.ndarray, list, pl.Series, pd.Series]] = None,
        sample_weight: Optional[Union[np.ndarray, list, pl.Series, pd.Series]] = None,
    ) -> "InsuranceEBM":
        """
        Fit the EBM to insurance training data.

        Parameters
        ----------
        X : polars.DataFrame or pandas.DataFrame
            Feature matrix. Column names are used as feature identifiers.
        y : array-like
            Target variable. For Poisson frequency models this is claim count.
            For Tweedie pure premium models this is incurred loss amount (may
            include zeros). For Gamma severity models this is average claim
            cost on non-zero rows only.
        exposure : array-like, optional
            Exposure measure (e.g. earned car years). Incorporated as a log
            offset: init_score = log(exposure). If omitted, all exposures are
            assumed to be 1.
        sample_weight : array-like, optional
            Row-level observation weights. Applied in addition to exposure.
            Useful for credibility weighting or re-balancing portfolio mix.

        Returns
        -------
        self
        """
        X_pd = _to_pandas(X)
        self._feature_names = list(X_pd.columns)

        y_arr = _ensure_array(y)
        exp_arr = _ensure_array(exposure)
        sw_arr = _ensure_array(sample_weight)

        self.ebm_ = self._build_ebm(self._feature_names)

        fit_kwargs: dict = {}

        if exp_arr is not None:
            # Exposure enters as a log offset (init_score)
            if np.any(exp_arr <= 0):
                raise ValueError("All exposure values must be strictly positive.")
            fit_kwargs["init_score"] = np.log(exp_arr)

        if sw_arr is not None:
            fit_kwargs["sample_weight"] = sw_arr

        self.ebm_.fit(X_pd, y_arr, **fit_kwargs)
        return self

    def _check_fitted(self) -> None:
        if self.ebm_ is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

    def predict(
        self,
        X: Union[pl.DataFrame, pd.DataFrame],
        exposure: Optional[Union[np.ndarray, list, pl.Series, pd.Series]] = None,
    ) -> np.ndarray:
        """
        Predict expected values on the original (not log) scale.

        For log-link families (Poisson, Tweedie, Gamma), the EBM produces additive
        scores on the log scale. This method applies exp() and multiplies by
        exposure to return predicted counts or amounts.

        Parameters
        ----------
        X : polars.DataFrame or pandas.DataFrame
            Feature matrix. Must have the same columns as the training data.
        exposure : array-like, optional
            Exposure values for prediction. If provided, predictions are scaled
            by exposure (equivalent to adding log(exposure) to the log score).
            For rate models, pass exposure=None and scale externally.

        Returns
        -------
        numpy.ndarray
            Predicted values on the response scale.
        """
        self._check_fitted()
        X_pd = _to_pandas(X)
        log_scores = self.ebm_.predict(X_pd)

        _log_link_losses = {"poisson", "tweedie", "gamma"}
        if self.loss in _log_link_losses:
            predictions = np.exp(log_scores)
            if exposure is not None:
                exp_arr = _ensure_array(exposure)
                if np.any(exp_arr <= 0):
                    raise ValueError("All exposure values must be strictly positive.")
                predictions = predictions * exp_arr
        else:
            predictions = log_scores

        return predictions

    def predict_log_score(
        self,
        X: Union[pl.DataFrame, pd.DataFrame],
    ) -> np.ndarray:
        """
        Return raw additive scores on the log scale (before exp transformation).

        Useful for combining predictions from separate frequency and severity
        models on a common additive scale.

        Parameters
        ----------
        X : polars.DataFrame or pandas.DataFrame
            Feature matrix.

        Returns
        -------
        numpy.ndarray
            Log-scale scores.
        """
        self._check_fitted()
        X_pd = _to_pandas(X)
        return self.ebm_.predict(X_pd)

    def score(
        self,
        X: Union[pl.DataFrame, pd.DataFrame],
        y: Union[np.ndarray, list, pl.Series, pd.Series],
        exposure: Optional[Union[np.ndarray, list, pl.Series, pd.Series]] = None,
    ) -> float:
        """
        Compute mean deviance on test data (lower is better).

        The deviance family matches the loss used at fit time. For Tweedie,
        the same variance_power is used.

        Parameters
        ----------
        X : polars.DataFrame or pandas.DataFrame
            Feature matrix.
        y : array-like
            Observed values.
        exposure : array-like, optional
            Exposure values used to scale predictions.

        Returns
        -------
        float
            Mean deviance (negated so higher = better, consistent with sklearn).
        """
        self._check_fitted()
        y_arr = _ensure_array(y)
        exp_arr = _ensure_array(exposure)
        y_pred = self.predict(X, exposure=exposure)
        weights = exp_arr  # use exposure as weights in deviance calculation

        if self.loss == "poisson":
            dev = _deviance_poisson(y_arr, y_pred, weights)
        elif self.loss == "gamma":
            dev = _deviance_gamma(y_arr, y_pred, weights)
        elif self.loss == "tweedie":
            dev = _deviance_tweedie(y_arr, y_pred, self.variance_power, weights)
        else:
            # MSE for non-GLM losses
            if weights is not None:
                dev = float(np.average((y_arr - y_pred) ** 2, weights=weights))
            else:
                dev = float(np.mean((y_arr - y_pred) ** 2))

        return -dev  # higher is better convention

    @property
    def feature_names(self) -> list:
        """Feature names from the training data."""
        self._check_fitted()
        return self._feature_names

    def __repr__(self) -> str:
        fitted = self.ebm_ is not None
        return (
            f"InsuranceEBM(loss='{self.loss}', "
            f"variance_power={self.variance_power}, "
            f"interactions='{self.interactions}', "
            f"fitted={fitted})"
        )
