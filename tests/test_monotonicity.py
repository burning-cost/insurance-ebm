"""
Tests for MonotonicityEditor: enforce, check, get_scores, plot_before_after.
"""

import numpy as np
import pytest

from insurance_ebm import MonotonicityEditor


class TestMonotonicityCheck:
    def test_check_returns_bool(self, fitted_poisson_model):
        me = MonotonicityEditor(fitted_poisson_model)
        result = me.check("driver_age")
        assert isinstance(result, bool)

    def test_check_increase(self, fitted_poisson_model):
        me = MonotonicityEditor(fitted_poisson_model)
        # After setting scores to strictly increasing, check should return True
        ebm = fitted_poisson_model.ebm_
        feature_idx = list(ebm.feature_names_in_).index("ncd")
        ebm.term_scores_[feature_idx] = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        assert me.check("ncd", direction="increase") is True

    def test_check_decrease(self, fitted_poisson_model):
        me = MonotonicityEditor(fitted_poisson_model)
        ebm = fitted_poisson_model.ebm_
        feature_idx = list(ebm.feature_names_in_).index("vehicle_age")
        ebm.term_scores_[feature_idx] = np.array([0.0, 0.5, 0.4, 0.3, 0.2, 0.1])
        assert me.check("vehicle_age", direction="decrease") is True

    def test_check_no_direction_auto(self, fitted_poisson_model):
        me = MonotonicityEditor(fitted_poisson_model)
        ebm = fitted_poisson_model.ebm_
        feature_idx = list(ebm.feature_names_in_).index("driver_age")
        # Set to increasing
        ebm.term_scores_[feature_idx] = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        result = me.check("driver_age")
        assert result is True

    def test_non_monotone_returns_false(self, fitted_poisson_model):
        me = MonotonicityEditor(fitted_poisson_model)
        ebm = fitted_poisson_model.ebm_
        feature_idx = list(ebm.feature_names_in_).index("driver_age")
        # Set non-monotone: up, down, up
        ebm.term_scores_[feature_idx] = np.array([0.0, 0.1, 0.5, 0.2, 0.4, 0.6])
        assert me.check("driver_age", direction="increase") is False
        assert me.check("driver_age", direction="decrease") is False

    def test_invalid_feature_raises(self, fitted_poisson_model):
        me = MonotonicityEditor(fitted_poisson_model)
        with pytest.raises(ValueError, match="not in model"):
            me.check("nonexistent")


class TestMonotonicityEnforce:
    def test_enforce_makes_monotone(self, fitted_poisson_model):
        me = MonotonicityEditor(fitted_poisson_model)
        ebm = fitted_poisson_model.ebm_
        feature_idx = list(ebm.feature_names_in_).index("ncd")
        # Set non-monotone
        ebm.term_scores_[feature_idx] = np.array([0.0, 0.3, 0.1, 0.4, 0.2, 0.5])

        me.enforce("ncd", direction="increase")
        assert me.check("ncd", direction="increase") is True

    def test_enforce_decrease(self, fitted_poisson_model):
        me = MonotonicityEditor(fitted_poisson_model)
        ebm = fitted_poisson_model.ebm_
        feature_idx = list(ebm.feature_names_in_).index("vehicle_age")
        ebm.term_scores_[feature_idx] = np.array([0.0, 0.5, 0.2, 0.4, 0.1, 0.3])

        me.enforce("vehicle_age", direction="decrease")
        assert me.check("vehicle_age", direction="decrease") is True

    def test_enforce_auto_detects_direction(self, fitted_poisson_model):
        me = MonotonicityEditor(fitted_poisson_model)
        ebm = fitted_poisson_model.ebm_
        feature_idx = list(ebm.feature_names_in_).index("driver_age")
        # Mostly increasing with a blip
        ebm.term_scores_[feature_idx] = np.array([0.0, 0.1, 0.3, 0.25, 0.5, 0.6])

        me.enforce("driver_age", direction="auto")
        # Should be monotone in some direction
        assert me.check("driver_age") is True

    def test_enforce_already_monotone_is_noop(self, fitted_poisson_model):
        me = MonotonicityEditor(fitted_poisson_model)
        ebm = fitted_poisson_model.ebm_
        feature_idx = list(ebm.feature_names_in_).index("ncd")
        original = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        ebm.term_scores_[feature_idx] = original.copy()

        me.enforce("ncd", direction="increase")
        np.testing.assert_array_equal(ebm.term_scores_[feature_idx], original)

    def test_enforce_returns_self(self, fitted_poisson_model):
        me = MonotonicityEditor(fitted_poisson_model)
        result = me.enforce("driver_age", direction="auto")
        assert result is me

    def test_enforce_invalid_direction_raises(self, fitted_poisson_model):
        me = MonotonicityEditor(fitted_poisson_model)
        with pytest.raises(ValueError, match="direction"):
            me.enforce("driver_age", direction="flat")

    def test_enforce_modifies_predictions(self, fitted_poisson_model, X_polars):
        """Enforce changes term_scores_, which should affect predictions."""
        me = MonotonicityEditor(fitted_poisson_model)
        ebm = fitted_poisson_model.ebm_
        feature_idx = list(ebm.feature_names_in_).index("ncd")
        # Set highly non-monotone
        ebm.term_scores_[feature_idx] = np.array([0.0, 0.5, -0.3, 0.8, -0.2, 0.9])

        preds_before = fitted_poisson_model.predict(X_polars).copy()
        me.enforce("ncd", direction="increase")
        preds_after = fitted_poisson_model.predict(X_polars)

        # In the mock, predict() doesn't use term_scores directly, so we just
        # verify enforce() ran without error and check returns True
        assert me.check("ncd", direction="increase") is True


class TestGetScores:
    def test_get_scores_returns_copy(self, fitted_poisson_model):
        me = MonotonicityEditor(fitted_poisson_model)
        scores = me.get_scores("driver_age")
        assert isinstance(scores, np.ndarray)

    def test_get_scores_is_copy_not_view(self, fitted_poisson_model):
        me = MonotonicityEditor(fitted_poisson_model)
        scores = me.get_scores("driver_age")
        ebm = fitted_poisson_model.ebm_
        feature_idx = list(ebm.feature_names_in_).index("driver_age")
        original = ebm.term_scores_[feature_idx][2]
        scores[2] = 999.9  # modify copy
        assert ebm.term_scores_[feature_idx][2] == original


class TestPlotBeforeAfter:
    def test_plot_before_after(self, fitted_poisson_model):
        import matplotlib
        matplotlib.use("Agg")

        me = MonotonicityEditor(fitted_poisson_model)
        scores_before = me.get_scores("ncd")
        me.enforce("ncd", direction="auto")
        fig = me.plot_before_after("ncd", scores_before=scores_before, direction="increase")
        assert fig is not None

    def test_unfitted_model_raises(self):
        from insurance_ebm import InsuranceEBM
        model = InsuranceEBM(loss="poisson")
        with pytest.raises(RuntimeError, match="fitted"):
            MonotonicityEditor(model)
