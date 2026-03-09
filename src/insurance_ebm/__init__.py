"""
insurance-ebm — insurance pricing workflow for interpretML's ExplainableBoostingMachine.

interpretML handles: Poisson/Tweedie/Gamma loss, exposure via init_score,
monotonicity constraints, interaction detection, bagging.

This library adds: exposure-aware fit/predict, relativity table extraction,
actuarial diagnostics (Gini, Lorenz, double-lift, deviance), post-fit
monotonicity enforcement, and GLM vs EBM comparison tools.

Quick start::

    from insurance_ebm import InsuranceEBM, RelativitiesTable
    from insurance_ebm.diagnostics import gini, double_lift

    model = InsuranceEBM(loss='poisson', interactions='3x')
    model.fit(X_train, y_train, exposure=exposure_train)

    preds = model.predict(X_test, exposure=exposure_test)
    print(f"Gini: {gini(y_test, preds, exposure=exposure_test):.3f}")

    rt = RelativitiesTable(model)
    print(rt.table('driver_age'))
    print(rt.summary())
"""

from ._model import InsuranceEBM
from ._relativities import RelativitiesTable
from ._monotonicity import MonotonicityEditor
from ._comparison import GLMComparison
from . import _diagnostics as diagnostics

# Expose the most commonly used diagnostic functions at top level
from ._diagnostics import (
    gini,
    lorenz_curve,
    double_lift,
    deviance,
    residual_plot,
    calibration_table,
)

__version__ = "0.1.0"
__all__ = [
    "InsuranceEBM",
    "RelativitiesTable",
    "MonotonicityEditor",
    "GLMComparison",
    "diagnostics",
    # Diagnostic functions
    "gini",
    "lorenz_curve",
    "double_lift",
    "deviance",
    "residual_plot",
    "calibration_table",
]
