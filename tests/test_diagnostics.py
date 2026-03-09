"""
Tests for Diagnostics: Gini coefficient, Lorenz curve, double-lift,
deviance calculation, residual plot, and calibration table.
"""

import numpy as np
import polars as pl
import pytest

from insurance_ebm import gini, lorenz_curve, double_lift, deviance, calibration_table
from insurance_ebm._diagnostics import residual_plot


# ---------------------------------------------------------------------------
# Gini coefficient
# ---------------------------------------------------------------------------

class TestGini:
    def test_perfect_model_gini_one(self):
        """Perfect model (sorted by true values) has Gini = 1."""
        y_true = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y_pred = y_true.copy()  # perfect ordering
        g = gini(y_true, y_pred)
        assert np.isclose(g, 1.0, atol=0.01), f"Expected ~1.0, got {g:.4f}"

    def test_random_model_gini_near_zero(self):
        """Random model (constant prediction) has Gini near 0."""
        rng = np.random.default_rng(0)
        y_true = rng.poisson(0.1, size=1000).astype(float)
        y_pred = np.ones(1000) * 0.1  # no discrimination
        g = gini(y_true, y_pred)
        assert abs(g) < 0.15, f"Expected near 0, got {g:.4f}"

    def test_gini_with_exposure(self):
        """Gini with exposure should run without error and return float."""
        rng = np.random.default_rng(1)
        y_true = rng.poisson(0.05, size=200).astype(float)
        y_pred = rng.uniform(0.03, 0.1, size=200)
        exposure = rng.uniform(0.5, 1.0, size=200)
        g = gini(y_true, y_pred, exposure=exposure)
        assert isinstance(g, float)
        assert -1.0 <= g <= 1.1  # allow slight numerical overshoot

    def test_gini_polars_input(self):
        y_true = pl.Series([0.0, 0.0, 1.0, 2.0, 3.0])
        y_pred = pl.Series([0.05, 0.06, 0.1, 0.2, 0.3])
        g = gini(y_true, y_pred)
        assert isinstance(g, float)

    def test_gini_better_model_higher_gini(self, fitted_poisson_model, X_polars, synthetic_data):
        """A fitted model should have higher Gini than constant predictions."""
        y = synthetic_data["claim_count"]
        exp = synthetic_data["exposure"]
        preds = fitted_poisson_model.predict(X_polars, exposure=exp)
        preds_const = np.full_like(preds, np.mean(y))

        g_model = gini(y, preds, exposure=exp)
        g_const = gini(y, preds_const, exposure=exp)
        assert g_model >= g_const, (
            f"Fitted model Gini {g_model:.3f} should >= constant {g_const:.3f}"
        )

    def test_gini_all_zero_y(self):
        """All-zero actuals: Gini should return 0."""
        y_true = np.zeros(100)
        y_pred = np.ones(100) * 0.1
        g = gini(y_true, y_pred)
        assert g == 0.0


# ---------------------------------------------------------------------------
# Lorenz curve
# ---------------------------------------------------------------------------

class TestLorenzCurve:
    def test_returns_two_arrays(self):
        y_true = np.array([0.0, 1.0, 0.0, 2.0, 0.5])
        y_pred = np.array([0.05, 0.15, 0.04, 0.25, 0.08])
        frac_exp, frac_loss = lorenz_curve(y_true, y_pred)
        assert isinstance(frac_exp, np.ndarray)
        assert isinstance(frac_loss, np.ndarray)

    def test_starts_at_zero_ends_at_one(self):
        y_true = np.array([0.0, 1.0, 0.0, 2.0, 0.5])
        y_pred = np.array([0.05, 0.15, 0.04, 0.25, 0.08])
        frac_exp, frac_loss = lorenz_curve(y_true, y_pred)
        assert np.isclose(frac_exp[0], 0.0)
        assert np.isclose(frac_exp[-1], 1.0)
        assert np.isclose(frac_loss[0], 0.0)
        assert np.isclose(frac_loss[-1], 1.0)

    def test_monotone_increasing(self):
        rng = np.random.default_rng(5)
        y_true = rng.poisson(0.1, 200).astype(float)
        y_pred = rng.uniform(0.05, 0.15, 200)
        frac_exp, frac_loss = lorenz_curve(y_true, y_pred)
        assert np.all(np.diff(frac_exp) >= 0)
        # frac_loss may not be strictly monotone if many zeros, but should be non-decreasing
        assert np.all(np.diff(frac_loss) >= -1e-10)

    def test_lengths_match(self):
        y_true = np.array([0.0, 1.0, 2.0])
        y_pred = np.array([0.1, 0.2, 0.3])
        frac_exp, frac_loss = lorenz_curve(y_true, y_pred)
        assert len(frac_exp) == len(frac_loss)

    def test_plot_returns_arrays(self):
        import matplotlib
        matplotlib.use("Agg")
        y_true = np.array([0.0, 1.0, 0.0, 2.0])
        y_pred = np.array([0.05, 0.2, 0.03, 0.3])
        frac_exp, frac_loss = lorenz_curve(y_true, y_pred, plot=True)
        assert len(frac_exp) > 0

    def test_with_exposure(self):
        y_true = np.array([0.0, 1.0, 0.0, 2.0, 0.5])
        y_pred = np.array([0.05, 0.15, 0.04, 0.25, 0.08])
        exposure = np.array([1.0, 0.5, 1.0, 0.75, 0.9])
        frac_exp, frac_loss = lorenz_curve(y_true, y_pred, exposure=exposure)
        assert np.isclose(frac_exp[-1], 1.0)


# ---------------------------------------------------------------------------
# Double-lift chart
# ---------------------------------------------------------------------------

class TestDoubleLift:
    def test_returns_polars_dataframe(self, fitted_poisson_model, X_polars, synthetic_data):
        preds = fitted_poisson_model.predict(X_polars, exposure=synthetic_data["exposure"])
        df = double_lift(synthetic_data["claim_count"], preds, exposure=synthetic_data["exposure"])
        assert isinstance(df, pl.DataFrame)

    def test_n_bands_correct(self, fitted_poisson_model, X_polars, synthetic_data):
        preds = fitted_poisson_model.predict(X_polars)
        df = double_lift(synthetic_data["claim_count"], preds, n_bands=10)
        assert len(df) == 10

    def test_custom_n_bands(self, fitted_poisson_model, X_polars, synthetic_data):
        preds = fitted_poisson_model.predict(X_polars)
        for n in [5, 8, 20]:
            df = double_lift(synthetic_data["claim_count"], preds, n_bands=n)
            assert len(df) == n

    def test_columns(self, fitted_poisson_model, X_polars, synthetic_data):
        preds = fitted_poisson_model.predict(X_polars)
        df = double_lift(synthetic_data["claim_count"], preds)
        expected = {"band", "exposure", "actual", "predicted", "ae_ratio"}
        assert expected.issubset(set(df.columns))

    def test_band_numbers_sequential(self, fitted_poisson_model, X_polars, synthetic_data):
        preds = fitted_poisson_model.predict(X_polars)
        df = double_lift(synthetic_data["claim_count"], preds, n_bands=5)
        assert df["band"].to_list() == [1, 2, 3, 4, 5]

    def test_total_exposure_correct(self, synthetic_data):
        n = len(synthetic_data["exposure"])
        y = synthetic_data["claim_count"]
        y_pred = np.full(n, y.mean())
        exp = synthetic_data["exposure"]
        df = double_lift(y, y_pred, exposure=exp, n_bands=5)
        total_exp = df["exposure"].sum()
        np.testing.assert_allclose(total_exp, exp.sum(), rtol=0.01)

    def test_predicted_positive(self, fitted_poisson_model, X_polars, synthetic_data):
        preds = fitted_poisson_model.predict(X_polars)
        df = double_lift(synthetic_data["claim_count"], preds)
        assert np.all(df["predicted"].to_numpy() > 0)


# ---------------------------------------------------------------------------
# Deviance
# ---------------------------------------------------------------------------

class TestDeviance:
    def test_poisson_deviance_non_negative(self):
        y_true = np.array([0.0, 1.0, 2.0, 0.0, 3.0])
        y_pred = np.array([0.5, 0.8, 1.5, 0.2, 2.5])
        d = deviance(y_true, y_pred, family="poisson")
        assert d >= 0

    def test_poisson_deviance_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        d = deviance(y, y, family="poisson")
        assert np.isclose(d, 0.0, atol=1e-8)

    def test_gamma_deviance_non_negative(self):
        y_true = np.array([1000.0, 2000.0, 1500.0])
        y_pred = np.array([900.0, 2100.0, 1600.0])
        d = deviance(y_true, y_pred, family="gamma")
        assert d >= 0

    def test_tweedie_deviance_non_negative(self):
        y_true = np.array([0.0, 100.0, 250.0, 0.0, 500.0])
        y_pred = np.array([50.0, 120.0, 200.0, 30.0, 450.0])
        d = deviance(y_true, y_pred, family="tweedie", variance_power=1.5)
        assert d >= 0

    def test_invalid_family_raises(self):
        with pytest.raises(ValueError, match="family"):
            deviance(np.ones(5), np.ones(5), family="negbinom")

    def test_poisson_deviance_with_exposure(self):
        y_true = np.array([0.0, 1.0, 2.0])
        y_pred = np.array([0.5, 1.0, 1.8])
        exposure = np.array([1.0, 0.5, 0.8])
        d = deviance(y_true, y_pred, exposure=exposure, family="poisson")
        assert d >= 0


# ---------------------------------------------------------------------------
# Calibration table
# ---------------------------------------------------------------------------

class TestCalibrationTable:
    def test_returns_polars_dataframe(self, fitted_poisson_model, X_polars, synthetic_data):
        preds = fitted_poisson_model.predict(X_polars)
        segments = X_polars["area"].to_numpy()
        df = calibration_table(synthetic_data["claim_count"], preds, segments)
        assert isinstance(df, pl.DataFrame)

    def test_columns(self, fitted_poisson_model, X_polars, synthetic_data):
        preds = fitted_poisson_model.predict(X_polars)
        segments = X_polars["area"].to_numpy()
        df = calibration_table(synthetic_data["claim_count"], preds, segments)
        expected = {"segment", "exposure", "actual_total", "predicted_total", "ae_ratio"}
        assert expected.issubset(set(df.columns))

    def test_n_segments(self, fitted_poisson_model, X_polars, synthetic_data):
        preds = fitted_poisson_model.predict(X_polars)
        segments = X_polars["area"].to_numpy()
        unique_segs = len(np.unique(segments))
        df = calibration_table(synthetic_data["claim_count"], preds, segments)
        assert len(df) == unique_segs

    def test_ae_ratio_positive(self, fitted_poisson_model, X_polars, synthetic_data):
        preds = fitted_poisson_model.predict(X_polars)
        segments = X_polars["vehicle_group"].to_numpy()
        df = calibration_table(synthetic_data["claim_count"], preds, segments)
        # ae_ratio should be non-negative where predicted > 0
        valid = df.filter(pl.col("predicted_total") > 0)
        assert np.all(valid["ae_ratio"].to_numpy() >= 0)

    def test_sorted_by_ae_ratio(self, fitted_poisson_model, X_polars, synthetic_data):
        preds = fitted_poisson_model.predict(X_polars)
        segments = X_polars["area"].to_numpy()
        df = calibration_table(synthetic_data["claim_count"], preds, segments)
        ratios = df["ae_ratio"].to_numpy()
        # Remove NaN
        valid = ratios[~np.isnan(ratios)]
        assert np.all(np.diff(valid) <= 0), "Table should be sorted by ae_ratio descending"

    def test_with_exposure_weights(self, fitted_poisson_model, X_polars, synthetic_data):
        preds = fitted_poisson_model.predict(X_polars)
        segments = X_polars["area"].to_numpy()
        df = calibration_table(
            synthetic_data["claim_count"],
            preds,
            segments,
            exposure=synthetic_data["exposure"],
        )
        assert len(df) > 0

    def test_with_polars_series_segment(self, fitted_poisson_model, X_polars, synthetic_data):
        preds = fitted_poisson_model.predict(X_polars)
        segments = X_polars["area"]  # polars Series
        df = calibration_table(synthetic_data["claim_count"], preds, segments)
        assert len(df) > 0


# ---------------------------------------------------------------------------
# Residual plot
# ---------------------------------------------------------------------------

class TestResidualPlot:
    def test_residual_plot_numeric(self, fitted_poisson_model, X_polars, synthetic_data):
        import matplotlib
        matplotlib.use("Agg")
        ax = residual_plot(
            fitted_poisson_model,
            X_polars,
            synthetic_data["claim_count"],
            "driver_age",
            exposure=synthetic_data["exposure"],
        )
        assert ax is not None

    def test_residual_plot_categorical(self, fitted_poisson_model, X_polars, synthetic_data):
        import matplotlib
        matplotlib.use("Agg")
        ax = residual_plot(
            fitted_poisson_model,
            X_polars,
            synthetic_data["claim_count"],
            "vehicle_group",
        )
        assert ax is not None
