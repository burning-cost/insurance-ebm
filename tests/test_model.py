"""
Tests for InsuranceEBM: init, fit, predict, score, validation.

interpretML is mocked in conftest.py — these tests run without it installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from insurance_ebm import InsuranceEBM


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestConstructor:
    def test_default_loss_is_poisson(self):
        m = InsuranceEBM()
        assert m.loss == "poisson"

    def test_custom_loss(self):
        m = InsuranceEBM(loss="gamma")
        assert m.loss == "gamma"

    def test_invalid_loss_raises(self):
        with pytest.raises(ValueError, match="loss must be one of"):
            InsuranceEBM(loss="negbinom")

    def test_variance_power_stored(self):
        m = InsuranceEBM(loss="tweedie", variance_power=1.8)
        assert m.variance_power == 1.8

    def test_interactions_stored_int(self):
        m = InsuranceEBM(interactions=3)
        assert m.interactions == 3

    def test_interactions_stored_3x(self):
        m = InsuranceEBM(interactions="3x")
        assert m.interactions == "3x"

    def test_monotone_constraints_default_empty(self):
        m = InsuranceEBM()
        assert m.monotone_constraints == {}

    def test_monotone_constraints_stored(self):
        m = InsuranceEBM(monotone_constraints={"ncd": -1})
        assert m.monotone_constraints == {"ncd": -1}

    def test_ebm_kwargs_forwarded(self):
        m = InsuranceEBM(random_state=99, n_jobs=2)
        assert m.ebm_kwargs["random_state"] == 99
        assert m.ebm_kwargs["n_jobs"] == 2

    def test_unfitted_ebm_is_none(self):
        m = InsuranceEBM()
        assert m.ebm_ is None

    def test_repr_unfitted(self):
        m = InsuranceEBM()
        r = repr(m)
        assert "InsuranceEBM" in r
        assert "fitted=False" in r

    def test_repr_shows_loss(self):
        m = InsuranceEBM(loss="gamma")
        assert "gamma" in repr(m)


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

class TestFit:
    def test_fit_returns_self(self, X_polars, synthetic_data):
        m = InsuranceEBM(loss="poisson", interactions=0)
        result = m.fit(X_polars, synthetic_data["claim_count"], exposure=synthetic_data["exposure"])
        assert result is m

    def test_fit_sets_ebm(self, X_polars, synthetic_data):
        m = InsuranceEBM(loss="poisson", interactions=0)
        m.fit(X_polars, synthetic_data["claim_count"])
        assert m.ebm_ is not None

    def test_fit_sets_feature_names(self, X_polars, synthetic_data):
        m = InsuranceEBM(loss="poisson", interactions=0)
        m.fit(X_polars, synthetic_data["claim_count"])
        assert m._feature_names == list(X_polars.columns)

    def test_fit_with_exposure(self, X_polars, synthetic_data):
        m = InsuranceEBM(loss="poisson", interactions=0)
        m.fit(X_polars, synthetic_data["claim_count"], exposure=synthetic_data["exposure"])
        assert m.ebm_ is not None

    def test_fit_pandas_input(self, X_pandas, synthetic_data):
        m = InsuranceEBM(loss="poisson", interactions=0)
        m.fit(X_pandas, synthetic_data["claim_count"], exposure=synthetic_data["exposure"])
        assert m.ebm_ is not None

    def test_fit_with_sample_weight(self, X_polars, synthetic_data):
        weights = np.ones(len(X_polars))
        m = InsuranceEBM(loss="poisson", interactions=0)
        m.fit(X_polars, synthetic_data["claim_count"], sample_weight=weights)
        assert m.ebm_ is not None

    def test_fit_invalid_input_type_raises(self, synthetic_data):
        m = InsuranceEBM(loss="poisson", interactions=0)
        with pytest.raises(TypeError):
            m.fit([[1, 2], [3, 4]], [0, 0])

    def test_fit_negative_exposure_raises(self, X_polars, synthetic_data):
        m = InsuranceEBM(loss="poisson", interactions=0)
        bad_exposure = synthetic_data["exposure"].copy()
        bad_exposure[0] = -1.0
        with pytest.raises(ValueError, match="strictly positive"):
            m.fit(X_polars, synthetic_data["claim_count"], exposure=bad_exposure)

    def test_fit_zero_exposure_raises(self, X_polars, synthetic_data):
        m = InsuranceEBM(loss="poisson", interactions=0)
        bad_exposure = synthetic_data["exposure"].copy()
        bad_exposure[0] = 0.0
        with pytest.raises(ValueError, match="strictly positive"):
            m.fit(X_polars, synthetic_data["claim_count"], exposure=bad_exposure)

    def test_fit_polars_series_y(self, X_polars, synthetic_data):
        m = InsuranceEBM(loss="poisson", interactions=0)
        y_series = pl.Series(synthetic_data["claim_count"])
        m.fit(X_polars, y_series)
        assert m.ebm_ is not None

    def test_fit_pandas_series_y(self, X_pandas, synthetic_data):
        m = InsuranceEBM(loss="poisson", interactions=0)
        y_series = pd.Series(synthetic_data["claim_count"])
        m.fit(X_pandas, y_series)
        assert m.ebm_ is not None

    def test_fit_repr_changes_after_fit(self, X_polars, synthetic_data):
        m = InsuranceEBM(loss="poisson", interactions=0)
        assert "fitted=False" in repr(m)
        m.fit(X_polars, synthetic_data["claim_count"])
        assert "fitted=True" in repr(m)

    def test_fit_tweedie(self, X_polars, synthetic_data):
        m = InsuranceEBM(loss="tweedie", variance_power=1.5, interactions=0)
        m.fit(X_polars, synthetic_data["claim_amount"])
        assert m.ebm_ is not None

    def test_fit_gamma(self, X_polars, synthetic_data):
        m = InsuranceEBM(loss="gamma", interactions=0)
        m.fit(X_polars, synthetic_data["claim_amount"])
        assert m.ebm_ is not None

    def test_fit_mse(self, X_polars, synthetic_data):
        m = InsuranceEBM(loss="mse", interactions=0)
        m.fit(X_polars, synthetic_data["claim_amount"])
        assert m.ebm_ is not None


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

class TestPredict:
    def test_predict_shape(self, fitted_poisson_model, X_polars):
        preds = fitted_poisson_model.predict(X_polars)
        assert preds.shape == (len(X_polars),)

    def test_predict_positive_poisson(self, fitted_poisson_model, X_polars):
        preds = fitted_poisson_model.predict(X_polars)
        assert np.all(preds > 0)

    def test_predict_with_exposure(self, fitted_poisson_model, X_polars, synthetic_data):
        preds = fitted_poisson_model.predict(X_polars, exposure=synthetic_data["exposure"])
        assert preds.shape == (len(X_polars),)
        assert np.all(preds > 0)

    def test_exposure_scales_predictions(self, fitted_poisson_model, X_polars):
        n = len(X_polars)
        preds_1 = fitted_poisson_model.predict(X_polars, exposure=np.ones(n))
        preds_2 = fitted_poisson_model.predict(X_polars, exposure=np.full(n, 2.0))
        np.testing.assert_allclose(preds_2, 2.0 * preds_1, rtol=1e-5)

    def test_predict_pandas_input(self, fitted_poisson_model, X_pandas):
        preds = fitted_poisson_model.predict(X_pandas)
        assert preds.shape == (len(X_pandas),)

    def test_predict_polars_series_exposure(self, fitted_poisson_model, X_polars, synthetic_data):
        exp_series = pl.Series(synthetic_data["exposure"])
        preds = fitted_poisson_model.predict(X_polars, exposure=exp_series)
        assert preds.shape == (len(X_polars),)

    def test_predict_invalid_exposure_raises(self, fitted_poisson_model, X_polars):
        bad_exp = np.full(len(X_polars), -1.0)
        with pytest.raises(ValueError, match="strictly positive"):
            fitted_poisson_model.predict(X_polars, exposure=bad_exp)

    def test_predict_unfitted_raises(self, X_polars):
        m = InsuranceEBM()
        with pytest.raises(RuntimeError, match="fitted"):
            m.predict(X_polars)

    def test_predict_log_score_shape(self, fitted_poisson_model, X_polars):
        scores = fitted_poisson_model.predict_log_score(X_polars)
        assert scores.shape == (len(X_polars),)

    def test_predict_log_score_exp_matches_predict(self, fitted_poisson_model, X_polars):
        scores = fitted_poisson_model.predict_log_score(X_polars)
        preds = fitted_poisson_model.predict(X_polars)
        np.testing.assert_allclose(np.exp(scores), preds, rtol=1e-5)

    def test_predict_log_score_unfitted_raises(self, X_polars):
        m = InsuranceEBM()
        with pytest.raises(RuntimeError, match="fitted"):
            m.predict_log_score(X_polars)

    def test_predict_mse_no_exp(self, X_polars, synthetic_data):
        """MSE model should return raw scores without exp()."""
        m = InsuranceEBM(loss="mse", interactions=0)
        m.fit(X_polars, synthetic_data["claim_amount"])
        preds = m.predict(X_polars)
        scores = m.predict_log_score(X_polars)
        # For non-log-link loss, predict returns raw scores (no exp applied)
        np.testing.assert_allclose(preds, scores, rtol=1e-5)


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------

class TestScore:
    def test_score_returns_float(self, fitted_poisson_model, X_polars, synthetic_data):
        s = fitted_poisson_model.score(X_polars, synthetic_data["claim_count"])
        assert isinstance(s, float)

    def test_score_is_negative_deviance(self, fitted_poisson_model, X_polars, synthetic_data):
        s = fitted_poisson_model.score(X_polars, synthetic_data["claim_count"])
        # Deviance is non-negative, so negated score should be <= 0
        assert s <= 0

    def test_score_with_exposure(self, fitted_poisson_model, X_polars, synthetic_data):
        s = fitted_poisson_model.score(
            X_polars, synthetic_data["claim_count"], exposure=synthetic_data["exposure"]
        )
        assert isinstance(s, float)

    def test_score_gamma(self, fitted_gamma_model, X_polars, synthetic_data):
        model, mask = fitted_gamma_model
        X_sev = X_polars.filter(pl.Series(mask))
        s = model.score(X_sev, synthetic_data["claim_amount"][mask])
        assert isinstance(s, float)

    def test_score_tweedie(self, fitted_tweedie_model, X_polars, synthetic_data):
        s = fitted_tweedie_model.score(X_polars, synthetic_data["claim_amount"])
        assert isinstance(s, float)

    def test_score_mse_model(self, X_polars, synthetic_data):
        m = InsuranceEBM(loss="mse", interactions=0)
        m.fit(X_polars, synthetic_data["claim_amount"])
        s = m.score(X_polars, synthetic_data["claim_amount"])
        assert isinstance(s, float)

    def test_score_unfitted_raises(self, X_polars):
        m = InsuranceEBM()
        with pytest.raises(RuntimeError, match="fitted"):
            m.score(X_polars, np.zeros(len(X_polars)))


# ---------------------------------------------------------------------------
# Feature names
# ---------------------------------------------------------------------------

class TestFeatureNames:
    def test_feature_names_match_columns(self, fitted_poisson_model, X_polars):
        assert fitted_poisson_model.feature_names == list(X_polars.columns)

    def test_feature_names_unfitted_raises(self):
        m = InsuranceEBM()
        with pytest.raises(RuntimeError, match="fitted"):
            _ = m.feature_names
