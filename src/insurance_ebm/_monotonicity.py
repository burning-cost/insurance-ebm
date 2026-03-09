"""
MonotonicityEditor — post-fit monotonicity enforcement for EBM shape functions.

interpretML supports monotonicity constraints at fit time via monotone_constraints.
This module adds post-fit enforcement: you can fit the model without constraints,
inspect the shape functions, and then apply isotonic regression to any shape that
should be monotone.

Important: this is a soft constraint applied to the shape function after fitting.
It's implemented via isotonic regression on the bin scores, not a re-fit of the
model. The underlying boosting trees are not changed — only the stored term scores
are adjusted. This is sufficient for prediction purposes but means the model has
not been re-validated on training data after the adjustment.

When to use constraints at fit time vs post-fit:
- At fit time: when you have strong prior knowledge (e.g. older vehicle = higher risk)
  and want to prevent overfitting against the constraint.
- Post-fit: when the shape function is mostly monotone but has a small non-monotone
  tail due to limited data, and you want to smooth it without a full re-fit.
"""

from __future__ import annotations

from typing import Literal, Optional, Union

import numpy as np

from ._model import InsuranceEBM


def _get_term_scores(ebm, feature: str) -> tuple[int, np.ndarray]:
    """
    Return (term_index, scores_array) for a named feature.

    The scores array is a view into ebm.term_scores_ for main effects.
    For interaction terms this will raise ValueError.
    """
    try:
        feature_idx = list(ebm.feature_names_in_).index(feature)
    except ValueError:
        raise ValueError(
            f"Feature '{feature}' not in model. "
            f"Available: {list(ebm.feature_names_in_)}"
        )

    # Find which term index corresponds to this feature (main effect only)
    for term_idx, term_features in enumerate(ebm.term_features_):
        if list(term_features) == [feature_idx]:
            scores = ebm.term_scores_[term_idx]
            if np.ndim(scores) > 1:
                raise ValueError(
                    f"'{feature}' appears in an interaction term. "
                    "MonotonicityEditor only operates on main effects."
                )
            return term_idx, scores

    raise ValueError(f"No main-effect term found for feature '{feature}'.")


def _isotonic_regression(scores: np.ndarray, direction: str) -> np.ndarray:
    """Apply isotonic regression to enforce monotone shape function."""
    from sklearn.isotonic import IsotonicRegression

    ir = IsotonicRegression(
        increasing=(direction == "increase"),
        out_of_bounds="clip",
    )
    x = np.arange(len(scores), dtype=float)
    return ir.fit_transform(x, scores)


def _detect_direction(scores: np.ndarray) -> str:
    """
    Detect the dominant direction of a shape function.

    Compares sum of positive increments vs sum of negative increments.
    Returns 'increase' or 'decrease'.
    """
    diffs = np.diff(scores)
    positive_mass = float(np.sum(diffs[diffs > 0]))
    negative_mass = float(np.abs(np.sum(diffs[diffs < 0])))
    return "increase" if positive_mass >= negative_mass else "decrease"


def _is_monotone(scores: np.ndarray, direction: str) -> bool:
    """Check whether scores are monotone in the given direction."""
    diffs = np.diff(scores)
    if direction == "increase":
        return bool(np.all(diffs >= -1e-10))
    elif direction == "decrease":
        return bool(np.all(diffs <= 1e-10))
    return False


class MonotonicityEditor:
    """
    Post-fit monotonicity enforcement for EBM shape functions.

    Applies isotonic regression to bin scores for a named feature, adjusting
    the stored term_scores_ in-place. The base model object (ebm_) is modified
    directly, so predictions from InsuranceEBM.predict() will reflect the change.

    Parameters
    ----------
    model : InsuranceEBM
        A fitted InsuranceEBM instance. The underlying ebm_ will be modified in-place.

    Notes
    -----
    The adjustment is not reversible once applied (the original scores are
    overwritten). Keep a copy of the model if you need to compare before/after
    predictions on a holdout set.
    """

    def __init__(self, model: InsuranceEBM) -> None:
        model._check_fitted()
        self.model = model
        self._ebm = model.ebm_

    def enforce(
        self,
        feature: str,
        direction: Literal["increase", "decrease", "auto"] = "auto",
    ) -> "MonotonicityEditor":
        """
        Enforce monotonicity on a feature's shape function via isotonic regression.

        The bin scores (term_scores_) are updated in-place. After calling this,
        model.predict() will reflect the monotone shape.

        Parameters
        ----------
        feature : str
            Feature name.
        direction : str
            'increase', 'decrease', or 'auto'. When 'auto', the dominant
            direction is detected from the existing shape function.

        Returns
        -------
        self
            Returns self for method chaining.
        """
        term_idx, scores = _get_term_scores(self._ebm, feature)

        if direction == "auto":
            direction = _detect_direction(scores)

        if direction not in ("increase", "decrease"):
            raise ValueError(f"direction must be 'increase', 'decrease', or 'auto', got '{direction}'")

        if _is_monotone(scores, direction):
            # Already monotone — nothing to do
            return self

        # interpretML stores a special first bin for missing values (index 0)
        # We preserve the missing-value score and only enforce on the main bins
        if len(scores) > 1:
            main_scores = scores[1:]
            main_scores_mono = _isotonic_regression(main_scores, direction)
            new_scores = np.concatenate([[scores[0]], main_scores_mono])
        else:
            new_scores = _isotonic_regression(scores, direction)

        self._ebm.term_scores_[term_idx] = new_scores
        return self

    def check(self, feature: str, direction: Optional[str] = None) -> bool:
        """
        Check whether a feature's shape function is monotone.

        Parameters
        ----------
        feature : str
            Feature name.
        direction : str, optional
            'increase' or 'decrease'. If None, checks both directions (returns
            True if monotone in either direction).

        Returns
        -------
        bool
            True if the shape function is monotone in the specified direction.
        """
        _, scores = _get_term_scores(self._ebm, feature)
        main_scores = scores[1:] if len(scores) > 1 else scores

        if direction is None:
            return _is_monotone(main_scores, "increase") or _is_monotone(main_scores, "decrease")
        return _is_monotone(main_scores, direction)

    def plot_before_after(
        self,
        feature: str,
        scores_before: np.ndarray,
        direction: Optional[str] = None,
        ax=None,
    ):
        """
        Side-by-side comparison of shape function before and after monotonicity enforcement.

        Because enforce() modifies scores in-place, you must capture the scores
        before calling enforce(). Use MonotonicityEditor.get_scores() to obtain
        a copy before enforcement.

        Parameters
        ----------
        feature : str
            Feature name.
        scores_before : numpy.ndarray
            Scores before enforcement (use get_scores() to capture these).
        direction : str, optional
            Direction label for the title.
        ax : matplotlib.axes.Axes, optional
            Two-element axes array. Creates a new figure if None.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        _, scores_after = _get_term_scores(self._ebm, feature)
        main_before = scores_before[1:] if len(scores_before) > 1 else scores_before
        main_after = scores_after[1:] if len(scores_after) > 1 else scores_after

        n = max(len(main_before), len(main_after))
        x = np.arange(n)

        if ax is None:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        else:
            axes = ax
            fig = axes[0].get_figure()

        for a, scores, title in zip(
            axes,
            [main_before, main_after],
            ["Before enforcement", f"After enforcement ({direction or 'monotone'})"],
        ):
            a.plot(x[:len(scores)], scores, marker="o", color="#1f77b4", linewidth=2)
            a.set_title(f"{feature} — {title}")
            a.set_xlabel("Bin index")
            a.set_ylabel("Score (log scale)")
            a.grid(alpha=0.3)

        fig.tight_layout()
        return fig

    def get_scores(self, feature: str) -> np.ndarray:
        """
        Return a copy of the current scores for a feature.

        Use this to capture scores before calling enforce() if you want to
        plot a before/after comparison.

        Parameters
        ----------
        feature : str
            Feature name.

        Returns
        -------
        numpy.ndarray
            Copy of term_scores_ for this feature.
        """
        _, scores = _get_term_scores(self._ebm, feature)
        return scores.copy()
