"""
RelativitiesTable — extract insurance-standard relativity tables from a fitted EBM.

A relativity table shows each bin of a rating factor as a multiplier relative to a
base level. The convention used here: base level is the bin with the largest total
weight (i.e. the most common value in the training data). All other bins are expressed
as exp(score_bin - score_base).

This matches the output format that UK pricing teams expect from a GLM factor table.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Union

import numpy as np
import polars as pl

from ._model import InsuranceEBM


def _get_ebm_shape(ebm, feature: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract bin edges and shape function scores for a given feature.

    Returns (bin_labels, scores) as numpy arrays. bin_labels are string
    representations of the bin intervals or category values.
    """
    try:
        feature_idx = ebm.feature_names_in_.tolist().index(feature)
    except ValueError:
        raise ValueError(
            f"Feature '{feature}' not found in model. "
            f"Available features: {list(ebm.feature_names_in_)}"
        )

    # interpretML stores term scores differently depending on version
    # Access via the explain_global() interface for compatibility
    explanation = ebm.explain_global(name="EBM")
    data = explanation.data(feature_idx)

    names = data.get("names", [])
    scores = data.get("scores", [])

    # For interaction terms, data has a 2-D scores array — skip those
    if np.ndim(scores) > 1:
        raise ValueError(
            f"Feature '{feature}' appears to be part of an interaction term. "
            "Use table() only for main effects."
        )

    names_arr = np.array([str(n) for n in names])
    scores_arr = np.array(scores, dtype=float)

    # interpretML returns n+1 names for n bins (bin edges); take midpoint labels
    # In older versions names may already be bin labels (equal length to scores)
    if len(names_arr) == len(scores_arr) + 1:
        # Numeric feature — names are bin edges, create interval labels
        labels = []
        for i in range(len(scores_arr)):
            lo = names_arr[i]
            hi = names_arr[i + 1]
            labels.append(f"({lo}, {hi}]")
        names_arr = np.array(labels)
    elif len(names_arr) != len(scores_arr):
        # Mismatch — truncate to min length
        n = min(len(names_arr), len(scores_arr))
        names_arr = names_arr[:n]
        scores_arr = scores_arr[:n]

    return names_arr, scores_arr


def _modal_bin_idx(ebm, feature: str) -> int:
    """
    Find the index of the largest-weight bin (modal category / most common interval).

    interpretML stores bin weights in ebm.term_bin_weights_. We use these to
    identify which bin the largest share of training observations fell into.
    Falls back to zero-score bin (the intercept bin) if weights are unavailable.
    """
    try:
        feature_idx = ebm.feature_names_in_.tolist().index(feature)
        weights = ebm.term_bin_weights_[feature_idx]
        # weights includes the missing-value bin as the last element; exclude it
        main_weights = weights[:-1] if len(weights) > 1 else weights
        return int(np.argmax(main_weights))
    except (AttributeError, IndexError):
        return 0


class RelativitiesTable:
    """
    Extract and present insurance-standard relativity tables from a fitted InsuranceEBM.

    A relativity is defined as exp(score_bin - score_base) where the base bin is the
    modal bin (the one with the highest training weight). A relativity of 1.0 means
    the bin contributes the same as the base level; >1 means higher risk, <1 lower.

    Parameters
    ----------
    model : InsuranceEBM
        A fitted InsuranceEBM instance.

    Examples
    --------
    >>> rt = RelativitiesTable(model)
    >>> rt.table("driver_age")
    shape: (10, 3)
    ┌────────────────┬───────────┬────────────┐
    │ bin_label      ┆ raw_score ┆ relativity │
    ...
    """

    def __init__(self, model: InsuranceEBM) -> None:
        model._check_fitted()
        self.model = model
        self._ebm = model.ebm_

    def table(self, feature: str) -> pl.DataFrame:
        """
        Return relativity table for a single feature.

        Parameters
        ----------
        feature : str
            Name of the feature (must be a main effect, not an interaction).

        Returns
        -------
        polars.DataFrame
            Columns: bin_label (str), raw_score (float), relativity (float).
            The base bin has relativity = 1.0.
        """
        names, scores = _get_ebm_shape(self._ebm, feature)
        base_idx = _modal_bin_idx(self._ebm, feature)

        # Clip base_idx to valid range (can happen if weights vector is shorter)
        base_idx = min(base_idx, len(scores) - 1)
        base_score = scores[base_idx]

        relativities = np.exp(scores - base_score)

        return pl.DataFrame(
            {
                "bin_label": names.tolist(),
                "raw_score": scores.tolist(),
                "relativity": relativities.tolist(),
            }
        )

    def plot(
        self,
        feature: str,
        kind: str = "bar",
        ax=None,
        title: Optional[str] = None,
    ):
        """
        Bar or line chart of relativities for a feature.

        Parameters
        ----------
        feature : str
            Feature name.
        kind : str
            'bar' (default) or 'line'.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. Creates a new figure if None.
        title : str, optional
            Plot title. Defaults to the feature name.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        df = self.table(feature)
        labels = df["bin_label"].to_list()
        rels = df["relativity"].to_list()

        if ax is None:
            fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.6), 5))

        if kind == "bar":
            colours = ["#d62728" if r > 1.0 else "#2ca02c" for r in rels]
            ax.bar(range(len(labels)), rels, color=colours, alpha=0.8, edgecolor="white")
        elif kind == "line":
            ax.plot(range(len(labels)), rels, marker="o", color="#1f77b4", linewidth=2)
            ax.fill_between(range(len(labels)), 1.0, rels, alpha=0.15, color="#1f77b4")
        else:
            raise ValueError(f"kind must be 'bar' or 'line', got '{kind}'")

        ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Relativity")
        ax.set_title(title or feature)
        ax.grid(axis="y", alpha=0.3)

        return ax

    def export_excel(self, path: Union[str, Path]) -> None:
        """
        Export relativity tables for all main-effect features to an Excel workbook.

        One sheet per feature. Interaction terms are skipped (they have 2-D score
        arrays that don't translate cleanly to a flat relativity table).

        Parameters
        ----------
        path : str or Path
            Output path for the .xlsx file.

        Notes
        -----
        Requires openpyxl. Install with: pip install insurance-ebm[excel]
        """
        try:
            import openpyxl  # noqa: F401
        except ImportError:
            raise ImportError(
                "openpyxl is required for Excel export. "
                "Install with: pip install insurance-ebm[excel]"
            )

        path = Path(path)
        feature_names = self.model.feature_names
        frames: dict[str, pl.DataFrame] = {}

        for feature in feature_names:
            try:
                df = self.table(feature)
                frames[feature] = df
            except ValueError:
                # Skip interaction terms or features that can't be extracted
                continue

        if not frames:
            raise RuntimeError("No main-effect features found to export.")

        # Write via pandas ExcelWriter — polars converts to pandas per sheet
        import pandas as pd
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for sheet_name, df in frames.items():
                # Excel sheet names are limited to 31 chars
                safe_name = sheet_name[:31]
                df.to_pandas().to_excel(writer, sheet_name=safe_name, index=False)

    def summary(self) -> pl.DataFrame:
        """
        Summary table of all main-effect features.

        Returns
        -------
        polars.DataFrame
            Columns: feature (str), n_bins (int), min_relativity (float),
            max_relativity (float), range (float).

            range = max_relativity / min_relativity, a measure of how much
            leverage the factor has over premium.
        """
        rows = []
        for feature in self.model.feature_names:
            try:
                df = self.table(feature)
                rels = df["relativity"].to_numpy()
                rows.append(
                    {
                        "feature": feature,
                        "n_bins": len(rels),
                        "min_relativity": float(np.min(rels)),
                        "max_relativity": float(np.max(rels)),
                        "range": float(np.max(rels) / np.maximum(np.min(rels), 1e-10)),
                    }
                )
            except ValueError:
                continue

        return pl.DataFrame(rows).sort("range", descending=True)
