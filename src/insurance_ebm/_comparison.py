"""
GLMComparison — compare EBM shape functions against GLM factor relativities.

When migrating from a GLM to an EBM (or running them in parallel), you need to
understand where the two models agree and disagree. This module aligns the EBM's
continuous shape function to the GLM's discrete factor levels and produces
side-by-side comparison tables and charts.

Two input modes for GLM relativities:
1. Pass a statsmodels GLM object and the feature name — the module extracts the
   factor levels and fitted coefficients automatically.
2. Pass a pre-computed polars DataFrame with columns [level, relativity] — this
   is the preferred mode because it doesn't require statsmodels in the production
   environment.

The comparison is always presented as relativities (exp(score - base_score)),
so both models are on the same scale regardless of how each was fitted.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import polars as pl

from ._model import InsuranceEBM
from ._relativities import RelativitiesTable, _get_ebm_shape, _modal_bin_idx


def _extract_glm_relativities(glm_model, feature: str) -> pl.DataFrame:
    """
    Extract factor relativities from a fitted statsmodels GLM.

    For a categorical feature, statsmodels encodes it as dummy variables named
    'feature[T.level]'. We extract the fitted coefficients for each level and
    exponentiate them.

    The reference level (coefficient = 0) gets relativity = 1.0.
    """
    try:
        import statsmodels  # noqa: F401
    except ImportError:
        raise ImportError(
            "statsmodels is required to extract relativities from a GLM object. "
            "Install with: pip install insurance-ebm[glm] "
            "Or pass glm_relativities as a pre-computed DataFrame instead."
        )

    params = glm_model.params
    # Filter to params that correspond to this feature
    prefix = f"{feature}["
    feature_params = {k: v for k, v in params.items() if k.startswith(prefix)}

    rows = []
    for param_name, coef in feature_params.items():
        # Extract level from 'feature[T.level]' or 'feature[level]'
        level = param_name.replace(f"{feature}[T.", "").replace(f"{feature}[", "").rstrip("]")
        rows.append({"level": level, "relativity": float(np.exp(coef))})

    if not rows:
        raise ValueError(
            f"No parameters found for feature '{feature}' in GLM. "
            f"Available params: {list(params.index)}"
        )

    # Add reference level with relativity = 1.0
    rows.append({"level": "(reference)", "relativity": 1.0})

    return pl.DataFrame(rows)


def _align_ebm_to_glm(
    ebm_df: pl.DataFrame, glm_df: pl.DataFrame
) -> pl.DataFrame:
    """
    Align EBM bins to GLM factor levels by string matching on bin labels.

    For categorical features, EBM bin labels and GLM level names should match
    directly. For continuous features, this is a best-effort join — EBM bins
    that don't match any GLM level are excluded.
    """
    # Try inner join on bin_label == level
    ebm_renamed = ebm_df.rename({"bin_label": "level", "relativity": "ebm_relativity"}).drop("raw_score")
    glm_renamed = glm_df.rename({"relativity": "glm_relativity"})

    joined = ebm_renamed.join(glm_renamed, on="level", how="inner")
    return joined


class GLMComparison:
    """
    Compare EBM shape functions against GLM factor relativities.

    Use this when you have an existing GLM and want to see where the EBM agrees
    or disagrees. The key output is a ranked list of features by divergence —
    features near the top deserve the most scrutiny.

    Parameters
    ----------
    model : InsuranceEBM
        A fitted InsuranceEBM instance.

    Examples
    --------
    Comparing against a pre-computed GLM relativity table:

    >>> glm_rel = pl.DataFrame({"level": ["A", "B", "C"], "relativity": [1.0, 1.25, 0.85]})
    >>> cmp = GLMComparison(model)
    >>> cmp.compare_shapes("vehicle_group", glm_relativities=glm_rel)
    """

    def __init__(self, model: InsuranceEBM) -> None:
        model._check_fitted()
        self.model = model
        self._ebm = model.ebm_
        self._rt = RelativitiesTable(model)

    def compare_shapes(
        self,
        feature: str,
        glm_model=None,
        glm_relativities: Optional[pl.DataFrame] = None,
    ) -> pl.DataFrame:
        """
        Align EBM bins with GLM factor levels for a single feature.

        Either glm_model or glm_relativities must be provided.

        Parameters
        ----------
        feature : str
            Feature name.
        glm_model : statsmodels GLM result, optional
            A fitted statsmodels GLM result object. The method extracts
            factor relativities from params automatically.
        glm_relativities : polars.DataFrame, optional
            Pre-computed GLM relativities with columns [level, relativity].
            'level' values should match the EBM bin labels (for categorical
            features) or be provided as string bin labels.

        Returns
        -------
        polars.DataFrame
            Columns: level (str), ebm_relativity (float), glm_relativity (float),
            abs_diff (float), pct_diff (float).
            Sorted by abs_diff descending.
        """
        if glm_model is None and glm_relativities is None:
            raise ValueError("Provide either glm_model or glm_relativities.")
        if glm_model is not None and glm_relativities is not None:
            raise ValueError("Provide either glm_model or glm_relativities, not both.")

        if glm_model is not None:
            glm_df = _extract_glm_relativities(glm_model, feature)
        else:
            # Validate schema
            required = {"level", "relativity"}
            if not required.issubset(set(glm_relativities.columns)):
                raise ValueError(
                    f"glm_relativities must have columns {required}, "
                    f"got {glm_relativities.columns}"
                )
            glm_df = glm_relativities

        ebm_df = self._rt.table(feature)
        aligned = _align_ebm_to_glm(ebm_df, glm_df)

        if aligned.is_empty():
            raise ValueError(
                f"No matching levels between EBM bins and GLM for feature '{feature}'. "
                "Check that bin labels match GLM factor level names."
            )

        aligned = aligned.with_columns(
            abs_diff=(pl.col("ebm_relativity") - pl.col("glm_relativity")).abs(),
            pct_diff=((pl.col("ebm_relativity") - pl.col("glm_relativity")).abs()
                      / pl.col("glm_relativity") * 100),
        ).sort("abs_diff", descending=True)

        return aligned

    def plot_comparison(
        self,
        feature: str,
        glm_model=None,
        glm_relativities: Optional[pl.DataFrame] = None,
        ax=None,
        title: Optional[str] = None,
    ):
        """
        Overlaid chart of EBM shape function and GLM relativities.

        Parameters
        ----------
        feature : str
            Feature name.
        glm_model : statsmodels GLM result, optional
            Fitted statsmodels GLM.
        glm_relativities : polars.DataFrame, optional
            Pre-computed GLM relativities [level, relativity].
        ax : matplotlib.axes.Axes, optional
            Axes to draw on.
        title : str, optional
            Plot title. Defaults to feature name.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        df = self.compare_shapes(feature, glm_model=glm_model, glm_relativities=glm_relativities)
        levels = df["level"].to_list()
        ebm_rels = df["ebm_relativity"].to_list()
        glm_rels = df["glm_relativity"].to_list()

        if ax is None:
            fig, ax = plt.subplots(figsize=(max(8, len(levels) * 0.7), 5))

        x = np.arange(len(levels))
        ax.bar(x - 0.2, ebm_rels, width=0.4, label="EBM", color="#1f77b4", alpha=0.8)
        ax.bar(x + 0.2, glm_rels, width=0.4, label="GLM", color="#ff7f0e", alpha=0.8)
        ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(levels, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Relativity")
        ax.set_title(title or f"{feature}: EBM vs GLM")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        return ax

    def divergence_summary(
        self,
        glm_model=None,
        glm_relativities_by_feature: Optional[dict] = None,
    ) -> pl.DataFrame:
        """
        Features ranked by maximum absolute relativity difference vs GLM.

        Useful for triaging: features near the top of the table are where the
        EBM and GLM disagree most strongly and deserve the most investigation.

        Parameters
        ----------
        glm_model : statsmodels GLM result, optional
            Fitted GLM. If provided, relativities are extracted for all features.
        glm_relativities_by_feature : dict, optional
            Mapping of {feature_name: polars.DataFrame} where each DataFrame
            has columns [level, relativity]. Alternative to glm_model when you
            have pre-computed tables for each feature.

        Returns
        -------
        polars.DataFrame
            Columns: feature (str), max_abs_diff (float), mean_abs_diff (float),
            n_levels_compared (int).
            Sorted by max_abs_diff descending.
        """
        if glm_model is None and glm_relativities_by_feature is None:
            raise ValueError(
                "Provide either glm_model or glm_relativities_by_feature."
            )

        rows = []
        for feature in self.model.feature_names:
            try:
                if glm_relativities_by_feature is not None:
                    rel_df = glm_relativities_by_feature.get(feature)
                    if rel_df is None:
                        continue
                    df = self.compare_shapes(feature, glm_relativities=rel_df)
                else:
                    df = self.compare_shapes(feature, glm_model=glm_model)

                diffs = df["abs_diff"].to_numpy()
                rows.append(
                    {
                        "feature": feature,
                        "max_abs_diff": float(np.max(diffs)),
                        "mean_abs_diff": float(np.mean(diffs)),
                        "n_levels_compared": len(diffs),
                    }
                )
            except (ValueError, KeyError):
                continue

        return pl.DataFrame(rows).sort("max_abs_diff", descending=True)
