"""
Shared fixtures for insurance-ebm tests.

Strategy: mock interpret entirely so tests run without interpretML installed.
The mock EBM exposes the same interface insurance-ebm uses:
  - feature_names_in_ (numpy array)
  - term_features_ (list of tuples)
  - term_scores_ (list of numpy arrays, mutable)
  - term_bin_weights_ (list of numpy arrays)
  - fit(X, y, **kwargs)
  - predict(X) -> log scores
  - explain_global(name) -> mock explanation object

Synthetic data mimics a UK motor insurance book:
- driver_age: continuous, 17-80
- vehicle_group: categorical, G1-G5
- area: categorical, A-E (urban/rural mix)
- ncd: no claims discount years 0-5
- vehicle_age: continuous, 0-20
- exposure: earned car years, 0.1-1.0
- claim_count: Poisson frequency * exposure
- claim_amount: severity (when claim_count > 0)
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import polars as pl
import pytest


# ---------------------------------------------------------------------------
# Build a fake interpret module so imports don't fail
# ---------------------------------------------------------------------------

def _make_mock_interpret():
    """Build a minimal fake interpret package in sys.modules."""
    interpret_mod = types.ModuleType("interpret")
    glassbox_mod = types.ModuleType("interpret.glassbox")

    class FakeEBR:
        """
        Minimal stand-in for ExplainableBoostingRegressor.

        Stores enough state that InsuranceEBM._build_ebm() can instantiate it
        and fit/predict/explain_global work correctly.
        """

        def __init__(self, **kwargs):
            self._kwargs = kwargs
            self.feature_names_in_ = None
            self.term_features_ = None
            self.term_scores_ = None
            self.term_bin_weights_ = None

        def fit(self, X, y, init_score=None, sample_weight=None):
            cols = list(X.columns)
            n_features = len(cols)
            self.feature_names_in_ = np.array(cols)

            # Main effects only (no interactions for simplicity)
            self.term_features_ = [(i,) for i in range(n_features)]

            # Each feature gets 5 bins + 1 missing-value bin (index 0)
            rng = np.random.default_rng(42)
            self.term_scores_ = [
                rng.standard_normal(6) * 0.3  # 6 values: [missing, b1..b5]
                for _ in range(n_features)
            ]

            # Bin weights: uniform-ish, missing bin last
            self.term_bin_weights_ = [
                np.array([10.0, 20.0, 25.0, 30.0, 20.0, 5.0])  # [b1..b5, missing]
                for _ in range(n_features)
            ]
            return self

        def predict(self, X):
            # Return a constant log score vector so predictions are exp(score)
            n = len(X)
            scores = np.zeros(n)
            # Add small per-row variation so predictions are not all identical
            rng = np.random.default_rng(0)
            scores += rng.standard_normal(n) * 0.05
            return scores

        def explain_global(self, name="EBM"):
            """Return a mock GlobalExplanation-like object."""
            explanation = MagicMock()

            def _data(feature_idx):
                n_bins = 5
                # names are bin edges (n+1 for numeric)
                names = [str(i) for i in range(n_bins + 1)]
                scores = self.term_scores_[feature_idx][1:]  # skip missing bin
                return {"names": names, "scores": scores.tolist()}

            explanation.data.side_effect = _data
            return explanation

    glassbox_mod.ExplainableBoostingRegressor = FakeEBR
    interpret_mod.glassbox = glassbox_mod

    return interpret_mod, glassbox_mod


# Inject mock before any insurance_ebm imports happen
_interpret_mod, _glassbox_mod = _make_mock_interpret()
sys.modules.setdefault("interpret", _interpret_mod)
sys.modules.setdefault("interpret.glassbox", _glassbox_mod)


# ---------------------------------------------------------------------------
# Synthetic motor insurance data
# ---------------------------------------------------------------------------

def _make_synthetic_motor(n: int = 300, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)

    driver_age = rng.integers(17, 80, size=n).astype(float)
    vehicle_group = rng.choice(["G1", "G2", "G3", "G4", "G5"], size=n)
    area = rng.choice(["A", "B", "C", "D", "E"], size=n)
    ncd = rng.integers(0, 6, size=n).astype(float)
    vehicle_age = rng.integers(0, 21, size=n).astype(float)
    exposure = rng.uniform(0.1, 1.0, size=n)

    log_rate = (
        -3.0
        + 0.01 * np.maximum(0, 30 - driver_age)
        + 0.3 * (vehicle_group == "G5").astype(float)
        + 0.2 * (vehicle_group == "G4").astype(float)
        - 0.2 * (area == "E").astype(float)
        - 0.1 * ncd
    )
    freq = np.exp(log_rate)
    claim_count = rng.poisson(freq * exposure)

    base_severity = 2000.0
    severity = rng.gamma(shape=2.0, scale=base_severity / 2.0, size=n)
    claim_amount = np.where(claim_count > 0, severity, 0.0)

    return {
        "driver_age": driver_age,
        "vehicle_group": vehicle_group,
        "area": area,
        "ncd": ncd,
        "vehicle_age": vehicle_age,
        "exposure": exposure,
        "claim_count": claim_count.astype(float),
        "claim_amount": claim_amount,
    }


@pytest.fixture(scope="session")
def synthetic_data():
    return _make_synthetic_motor(n=300, seed=42)


@pytest.fixture(scope="session")
def X_polars(synthetic_data):
    return pl.DataFrame({
        "driver_age": synthetic_data["driver_age"],
        "vehicle_group": synthetic_data["vehicle_group"],
        "area": synthetic_data["area"],
        "ncd": synthetic_data["ncd"],
        "vehicle_age": synthetic_data["vehicle_age"],
    })


@pytest.fixture(scope="session")
def X_pandas(synthetic_data):
    return pd.DataFrame({
        "driver_age": synthetic_data["driver_age"],
        "vehicle_group": synthetic_data["vehicle_group"],
        "area": synthetic_data["area"],
        "ncd": synthetic_data["ncd"],
        "vehicle_age": synthetic_data["vehicle_age"],
    })


@pytest.fixture(scope="session")
def fitted_poisson_model(X_polars, synthetic_data):
    from insurance_ebm import InsuranceEBM
    model = InsuranceEBM(loss="poisson", interactions=0, random_state=42, n_jobs=1)
    model.fit(X_polars, synthetic_data["claim_count"], exposure=synthetic_data["exposure"])
    return model


@pytest.fixture(scope="session")
def fitted_gamma_model(X_polars, synthetic_data):
    from insurance_ebm import InsuranceEBM
    mask = synthetic_data["claim_count"] > 0
    if mask.sum() < 10:
        pytest.skip("Not enough non-zero claims for gamma model test")
    X_sev = X_polars.filter(pl.Series(mask))
    model = InsuranceEBM(loss="gamma", interactions=0, random_state=42, n_jobs=1)
    model.fit(X_sev, synthetic_data["claim_amount"][mask], exposure=synthetic_data["exposure"][mask])
    return model, mask


@pytest.fixture(scope="session")
def fitted_tweedie_model(X_polars, synthetic_data):
    from insurance_ebm import InsuranceEBM
    model = InsuranceEBM(loss="tweedie", variance_power=1.5, interactions=0, random_state=42, n_jobs=1)
    model.fit(X_polars, synthetic_data["claim_amount"], exposure=synthetic_data["exposure"])
    return model
