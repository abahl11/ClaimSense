"""Tests for the tabular feature helpers in src/features/tabular.py."""

import pandas as pd

from src.data.clean import clean_claims
from src.features.tabular import (
    age_band,
    band_labels,
    delay_days,
    dependents_group,
    make_bands,
    total_dependents,
)


def test_band_labels_integer_and_open_last_band():
    assert band_labels([0, 20, 25], integer=True) == ["<20", "20-24", "25+"]


def test_band_labels_continuous():
    assert band_labels([0, 200, 300], integer=False) == ["<200", "200-300", "300+"]


def test_bands_include_lower_edge():
    bands = make_bands(pd.Series([19, 20, 24, 25, 99]), [0, 20, 25])
    assert bands.astype(str).tolist() == ["<20", "20-24", "20-24", "25+", "25+"]


def test_missing_values_stay_missing_in_bands():
    assert make_bands(pd.Series([None, 5.0]), [0, 10]).isna().tolist() == [True, False]


def test_age_band_uses_config_edges(cfg, make_claims):
    claims = clean_claims(make_claims({"age": 35}), cfg).data
    assert age_band(claims, cfg).astype(str).iloc[0] == "35-39"


def test_delay_days(cfg, make_claims):
    claims = clean_claims(make_claims(), cfg).data
    assert delay_days(claims, cfg).iloc[0] == 15


def test_dependents_total_and_cap(cfg, make_claims):
    cap = cfg["features"]["dependents_cap"]
    claims = clean_claims(
        make_claims(
            {"dependent_children": 1, "dependents_other": 1},
            {"dependent_children": cap + 2, "dependents_other": 0},
        ),
        cfg,
    ).data
    assert total_dependents(claims, cfg).tolist() == [2, cap + 2]
    assert dependents_group(claims, cfg).astype(str).tolist() == ["2", f"{cap}+"]
