"""
Tests for RelativitiesTable: relativity calculation, base level selection,
summary table, and Excel export.
"""

import numpy as np
import polars as pl
import pytest

from insurance_ebm import RelativitiesTable


class TestRelativitiesTable:
    def test_construction(self, fitted_poisson_model):
        rt = RelativitiesTable(fitted_poisson_model)
        assert rt.model is fitted_poisson_model

    def test_table_returns_polars_dataframe(self, fitted_poisson_model):
        rt = RelativitiesTable(fitted_poisson_model)
        df = rt.table("driver_age")
        assert isinstance(df, pl.DataFrame)

    def test_table_columns(self, fitted_poisson_model):
        rt = RelativitiesTable(fitted_poisson_model)
        df = rt.table("driver_age")
        assert set(df.columns) == {"bin_label", "raw_score", "relativity"}

    def test_base_bin_has_relativity_one(self, fitted_poisson_model):
        """The modal bin should have relativity exactly 1.0."""
        rt = RelativitiesTable(fitted_poisson_model)
        df = rt.table("driver_age")
        rels = df["relativity"].to_numpy()
        # At least one bin should have relativity = 1.0 (the base)
        assert np.any(np.isclose(rels, 1.0, atol=1e-6)), (
            f"Expected at least one bin with relativity=1.0, got min={rels.min():.4f}"
        )

    def test_relativities_positive(self, fitted_poisson_model):
        rt = RelativitiesTable(fitted_poisson_model)
        df = rt.table("driver_age")
        assert np.all(df["relativity"].to_numpy() > 0)

    def test_categorical_feature(self, fitted_poisson_model):
        rt = RelativitiesTable(fitted_poisson_model)
        df = rt.table("vehicle_group")
        assert len(df) > 0
        assert np.all(df["relativity"].to_numpy() > 0)

    def test_relativity_exp_of_score_diff(self, fitted_poisson_model):
        """Verify relativities = exp(score - base_score) manually."""
        rt = RelativitiesTable(fitted_poisson_model)
        df = rt.table("ncd")
        raw = df["raw_score"].to_numpy()
        rels = df["relativity"].to_numpy()

        # Find base bin (closest to 0 diff, i.e. where relativity = 1)
        base_idx = np.argmin(np.abs(rels - 1.0))
        base_score = raw[base_idx]

        expected_rels = np.exp(raw - base_score)
        np.testing.assert_allclose(rels, expected_rels, rtol=1e-5)

    def test_invalid_feature_raises(self, fitted_poisson_model):
        rt = RelativitiesTable(fitted_poisson_model)
        with pytest.raises(ValueError, match="not found"):
            rt.table("nonexistent_feature")

    def test_summary_returns_polars_dataframe(self, fitted_poisson_model):
        rt = RelativitiesTable(fitted_poisson_model)
        df = rt.summary()
        assert isinstance(df, pl.DataFrame)

    def test_summary_columns(self, fitted_poisson_model):
        rt = RelativitiesTable(fitted_poisson_model)
        df = rt.summary()
        expected = {"feature", "n_bins", "min_relativity", "max_relativity", "range"}
        assert expected.issubset(set(df.columns))

    def test_summary_has_all_features(self, fitted_poisson_model):
        rt = RelativitiesTable(fitted_poisson_model)
        df = rt.summary()
        # Should have at least the main features (may exclude interactions)
        assert len(df) >= 1

    def test_summary_range_geq_one(self, fitted_poisson_model):
        """range = max/min relativity >= 1 by definition."""
        rt = RelativitiesTable(fitted_poisson_model)
        df = rt.summary()
        ranges = df["range"].to_numpy()
        assert np.all(ranges >= 1.0 - 1e-9)

    def test_summary_sorted_by_range(self, fitted_poisson_model):
        """Summary should be sorted by range descending."""
        rt = RelativitiesTable(fitted_poisson_model)
        df = rt.summary()
        ranges = df["range"].to_numpy()
        assert np.all(np.diff(ranges) <= 0), "Summary should be sorted by range descending"

    def test_n_bins_positive(self, fitted_poisson_model):
        rt = RelativitiesTable(fitted_poisson_model)
        df = rt.summary()
        assert np.all(df["n_bins"].to_numpy() > 0)


class TestPlot:
    def test_plot_bar(self, fitted_poisson_model):
        import matplotlib
        matplotlib.use("Agg")
        rt = RelativitiesTable(fitted_poisson_model)
        ax = rt.plot("driver_age", kind="bar")
        assert ax is not None

    def test_plot_line(self, fitted_poisson_model):
        import matplotlib
        matplotlib.use("Agg")
        rt = RelativitiesTable(fitted_poisson_model)
        ax = rt.plot("ncd", kind="line")
        assert ax is not None

    def test_plot_invalid_kind_raises(self, fitted_poisson_model):
        rt = RelativitiesTable(fitted_poisson_model)
        with pytest.raises(ValueError, match="kind"):
            rt.plot("ncd", kind="scatter")


class TestExcelExport:
    def test_export_excel(self, fitted_poisson_model, tmp_path):
        pytest.importorskip("openpyxl")
        rt = RelativitiesTable(fitted_poisson_model)
        path = tmp_path / "relativities.xlsx"
        rt.export_excel(path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_export_excel_no_openpyxl(self, fitted_poisson_model, tmp_path, monkeypatch):
        """Should raise ImportError if openpyxl is not available."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "openpyxl":
                raise ImportError("openpyxl not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        rt = RelativitiesTable(fitted_poisson_model)
        with pytest.raises(ImportError, match="openpyxl"):
            rt.export_excel(tmp_path / "test.xlsx")
