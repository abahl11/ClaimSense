"""End-to-end scoring of a raw claim with the saved models (skipped until they are trained)."""

import numpy as np
import pytest

from src.config import load_config
from src.models.predict import (
    clean_new_claims,
    explain_claim,
    load_models,
    make_explainer,
    score_claims,
)

CFG = load_config()
pytestmark = pytest.mark.skipif(
    not (CFG.path("regression_model").is_file() and CFG.path("classifier_model").is_file()),
    reason="models not trained yet (run the pipeline)",
)


@pytest.fixture(scope="module")
def models():
    return load_models(CFG)


def test_scores_a_raw_claim(make_claims, models):
    raw = make_claims({}, {"description": "CRUSHED LEFT FOOT FRACTURE", "initial_estimate": 50000})
    raw = raw.drop(columns=CFG.col("target"))  # new claims have no final cost yet
    scored = score_claims(clean_new_claims(raw, CFG), CFG, models)
    assert len(scored) == 2
    assert (scored["expected_cost"] > 0).all()
    assert (scored["interval_low"] <= scored["interval_high"]).all()
    assert scored["high_cost_probability"].between(0, 1).all()
    assert scored["flag_for_review"].dtype == bool


def test_expected_cost_is_at_least_typical_cost(make_claims, models):
    scored = score_claims(clean_new_claims(make_claims(), CFG), CFG, models)
    assert (scored["expected_cost"] >= scored["typical_cost"] - 1e-6).all()


def test_explain_claim_returns_top_drivers(make_claims, models):
    cleaned = clean_new_claims(make_claims(), CFG)
    drivers = explain_claim(cleaned, CFG, models, make_explainer(models), top_n=5)
    assert len(drivers) == 5
    assert np.allclose(drivers["factor"], np.exp(drivers["shap"]))
    assert drivers["shap"].abs().is_monotonic_decreasing
