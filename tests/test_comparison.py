"""
Tests for GLMComparison: compare EBM shapes against GLM relativities.
Uses pre-computed polars DataFrame input — no statsmodels required.
"""

import numpy as np
import polars as pl
import pytest

from insurance_ebm import GLMComparison


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_glm_rel(levels: list, relativities: list) -> pl.DataFrame:
    return pl.DataFrame({"level": levels, "relativity": relativities})


# ---------------------------------------------------------------------------
# compare_shapes
# ---------------------------------------------------------------------------

class TestCompareShapes:
    def test_requires_glm_input(self, fitted_poisson_model):
        cmp = GLMComparison(fitted_poisson_model)
        with pytest.raises(ValueError, match="Provide either"):
            cmp.compare_shapes("driver_age")

    def test_rejects_both_inputs(self, fitted_poisson_model):
        cmp = GLMComparison(fitted_poisson_model)
        rel = _make_glm_rel(["(0, 1]"], [1.0])
        with pytest.raises(ValueError, match="not both"):
            cmp.compare_shapes("driver_age", glm_model=object(), glm_relativities=rel)

    def test_invalid_schema_raises(self, fitted_poisson_model):
        cmp = GLMComparison(fitted_poisson_model)
        bad_df = pl.DataFrame({"factor": ["A"], "rel": [1.0]})
        with pytest.raises(ValueError, match="columns"):
            cmp.compare_shapes("driver_age", glm_relativities=bad_df)

    def test_no_matching_levels_raises(self, fitted_poisson_model):
        cmp = GLMComparison(fitted_poisson_model)
        # EBM bin labels are "(0, 1]" etc; GLM levels are "X", "Y" — no overlap
        rel = _make_glm_rel(["X", "Y", "Z"], [1.0, 1.2, 0.9])
        with pytest.raises(ValueError, match="No matching levels"):
            cmp.compare_shapes("driver_age", glm_relativities=rel)

    def test_returns_polars_dataframe_on_match(self, fitted_poisson_model):
        """Use labels that match the mock EBM's bin labels: '(0, 1]' through '(4, 5]'."""
        cmp = GLMComparison(fitted_poisson_model)
        # Mock EBM produces labels: "(0, 1]", "(1, 2]", "(2, 3]", "(3, 4]", "(4, 5]"
        labels = [f"({i}, {i+1}]" for i in range(5)]
        rel = _make_glm_rel(labels, [1.0, 1.1, 1.2, 1.3, 1.4])
        df = cmp.compare_shapes("driver_age", glm_relativities=rel)
        assert isinstance(df, pl.DataFrame)

    def test_output_columns(self, fitted_poisson_model):
        cmp = GLMComparison(fitted_poisson_model)
        labels = [f"({i}, {i+1}]" for i in range(5)]
        rel = _make_glm_rel(labels, [1.0, 1.1, 1.2, 1.3, 1.4])
        df = cmp.compare_shapes("driver_age", glm_relativities=rel)
        expected = {"level", "ebm_relativity", "glm_relativity", "abs_diff", "pct_diff"}
        assert expected.issubset(set(df.columns))

    def test_sorted_by_abs_diff(self, fitted_poisson_model):
        cmp = GLMComparison(fitted_poisson_model)
        labels = [f"({i}, {i+1}]" for i in range(5)]
        rel = _make_glm_rel(labels, [1.0, 0.5, 1.8, 1.1, 2.0])
        df = cmp.compare_shapes("driver_age", glm_relativities=rel)
        diffs = df["abs_diff"].to_numpy()
        assert np.all(np.diff(diffs) <= 0), "Should be sorted by abs_diff descending"

    def test_abs_diff_non_negative(self, fitted_poisson_model):
        cmp = GLMComparison(fitted_poisson_model)
        labels = [f"({i}, {i+1}]" for i in range(5)]
        rel = _make_glm_rel(labels, [1.0, 1.1, 1.2, 1.3, 1.4])
        df = cmp.compare_shapes("driver_age", glm_relativities=rel)
        assert np.all(df["abs_diff"].to_numpy() >= 0)

    def test_pct_diff_non_negative(self, fitted_poisson_model):
        cmp = GLMComparison(fitted_poisson_model)
        labels = [f"({i}, {i+1}]" for i in range(5)]
        rel = _make_glm_rel(labels, [1.0, 1.1, 1.2, 1.3, 1.4])
        df = cmp.compare_shapes("driver_age", glm_relativities=rel)
        assert np.all(df["pct_diff"].to_numpy() >= 0)

    def test_identical_relativities_zero_diff(self, fitted_poisson_model):
        """If GLM relativities match EBM exactly, abs_diff should be 0."""
        cmp = GLMComparison(fitted_poisson_model)
        # Get the actual EBM relativities first
        from insurance_ebm import RelativitiesTable
        rt = RelativitiesTable(fitted_poisson_model)
        ebm_df = rt.table("ncd")

        labels = ebm_df["bin_label"].to_list()
        rels = ebm_df["relativity"].to_list()
        glm_rel = pl.DataFrame({"level": labels, "relativity": rels})

        df = cmp.compare_shapes("ncd", glm_relativities=glm_rel)
        np.testing.assert_allclose(df["abs_diff"].to_numpy(), 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# divergence_summary
# ---------------------------------------------------------------------------

class TestDivergenceSummary:
    def test_requires_input(self, fitted_poisson_model):
        cmp = GLMComparison(fitted_poisson_model)
        with pytest.raises(ValueError, match="Provide either"):
            cmp.divergence_summary()

    def test_returns_polars_dataframe(self, fitted_poisson_model):
        cmp = GLMComparison(fitted_poisson_model)
        labels = [f"({i}, {i+1}]" for i in range(5)]
        rel_df = _make_glm_rel(labels, [1.0, 1.1, 1.2, 1.3, 1.4])
        rel_by_feature = {feat: rel_df for feat in fitted_poisson_model.feature_names}
        df = cmp.divergence_summary(glm_relativities_by_feature=rel_by_feature)
        assert isinstance(df, pl.DataFrame)

    def test_columns(self, fitted_poisson_model):
        cmp = GLMComparison(fitted_poisson_model)
        labels = [f"({i}, {i+1}]" for i in range(5)]
        rel_df = _make_glm_rel(labels, [1.0, 1.1, 1.2, 1.3, 1.4])
        rel_by_feature = {feat: rel_df for feat in fitted_poisson_model.feature_names}
        df = cmp.divergence_summary(glm_relativities_by_feature=rel_by_feature)
        expected = {"feature", "max_abs_diff", "mean_abs_diff", "n_levels_compared"}
        assert expected.issubset(set(df.columns))

    def test_sorted_by_max_abs_diff(self, fitted_poisson_model):
        cmp = GLMComparison(fitted_poisson_model)
        labels = [f"({i}, {i+1}]" for i in range(5)]
        rel_df = _make_glm_rel(labels, [1.0, 1.5, 0.8, 1.2, 1.1])
        rel_by_feature = {feat: rel_df for feat in fitted_poisson_model.feature_names}
        df = cmp.divergence_summary(glm_relativities_by_feature=rel_by_feature)
        diffs = df["max_abs_diff"].to_numpy()
        assert np.all(np.diff(diffs) <= 0)

    def test_skips_missing_features(self, fitted_poisson_model):
        """Features not in glm_relativities_by_feature should be skipped gracefully."""
        cmp = GLMComparison(fitted_poisson_model)
        labels = [f"({i}, {i+1}]" for i in range(5)]
        rel_df = _make_glm_rel(labels, [1.0, 1.1, 1.2, 1.3, 1.4])
        # Only provide relativities for first feature
        first_feature = fitted_poisson_model.feature_names[0]
        rel_by_feature = {first_feature: rel_df}
        df = cmp.divergence_summary(glm_relativities_by_feature=rel_by_feature)
        assert len(df) <= 1  # at most 1 row


# ---------------------------------------------------------------------------
# plot_comparison
# ---------------------------------------------------------------------------

class TestPlotComparison:
    def test_plot_comparison_returns_axes(self, fitted_poisson_model):
        import matplotlib
        matplotlib.use("Agg")
        cmp = GLMComparison(fitted_poisson_model)
        labels = [f"({i}, {i+1}]" for i in range(5)]
        rel = _make_glm_rel(labels, [1.0, 1.1, 1.2, 1.3, 1.4])
        ax = cmp.plot_comparison("ncd", glm_relativities=rel)
        assert ax is not None

    def test_plot_comparison_with_title(self, fitted_poisson_model):
        import matplotlib
        matplotlib.use("Agg")
        cmp = GLMComparison(fitted_poisson_model)
        labels = [f"({i}, {i+1}]" for i in range(5)]
        rel = _make_glm_rel(labels, [1.0, 1.1, 1.2, 1.3, 1.4])
        ax = cmp.plot_comparison("ncd", glm_relativities=rel, title="My test plot")
        assert ax.get_title() == "My test plot"


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestGLMComparisonConstruction:
    def test_unfitted_model_raises(self):
        from insurance_ebm import InsuranceEBM
        model = InsuranceEBM(loss="poisson")
        with pytest.raises(RuntimeError, match="fitted"):
            GLMComparison(model)
