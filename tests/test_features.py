"""Tests for engineered features (tabular.py), text embeddings and the pipeline (build.py)."""

import numpy as np
import pandas as pd
import pytest

from src.data.clean import clean_claims
from src.features.build import build_preprocessor, feature_names, model_input_columns
from src.features.report import FEATURE_DOCS
from src.features.tabular import (
    GroupMedianImputer,
    age_band,
    band_labels,
    delay_days,
    dependents_group,
    engineer_features,
    make_bands,
    total_dependents,
)

NO_EMBEDDINGS = {"embedding_clusters": False}


def engineered(cfg, make_claims, *overrides) -> pd.DataFrame:
    return engineer_features(clean_claims(make_claims(*overrides), cfg).data, cfg)


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


# ---------------------------------------------------------------------------
# Row-wise engineered features
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("time", "night", "unknown"),
    [("22:00", 1, 0), ("03:00", 1, 0), ("12:00", 0, 0), ("00:00", 0, 1)],
)
def test_night_shift_excludes_unknown_midnight(cfg, make_claims, time, night, unknown):
    features = engineered(cfg, make_claims, {"accident_date": f"2001-03-05T{time}:00Z"})
    assert features["is_night_shift"].iloc[0] == night
    assert features["accident_time_unknown"].iloc[0] == unknown


def test_weekend_flag(cfg, make_claims):
    # 2001-03-10 was a Saturday, 2001-03-05 a Monday.
    features = engineered(
        cfg,
        make_claims,
        {"accident_date": "2001-03-10T10:00:00Z"},
        {"accident_date": "2001-03-05T10:00:00Z"},
    )
    assert features["is_weekend"].tolist() == [1, 0]


def test_hourly_wage_clips_small_hours(cfg, make_claims):
    minimum = cfg["features"]["min_hours_for_rate"]
    features = engineered(cfg, make_claims, {"weekly_wages": 50.0, "hours_per_week": minimum / 2})
    assert features["hourly_wage"].iloc[0] == pytest.approx(50.0 / minimum)


def test_ratios_and_logs(cfg, make_claims):
    features = engineered(cfg, make_claims, {"weekly_wages": 400.0, "initial_estimate": 2000})
    row = features.iloc[0]
    assert row["initial_to_wage_ratio"] == pytest.approx(5.0)
    assert row["log_initial_estimate"] == pytest.approx(np.log1p(2000))
    assert row["hours_per_day"] == pytest.approx(38.0 / 5)


def test_default_flags_use_recorded_values(cfg, make_claims):
    wage_default = cfg["cleaning"]["flags"]["heaped_wage_values"][0]
    initial_default = cfg["features"]["initial_default_values"][0]
    features = engineered(
        cfg,
        make_claims,
        {"weekly_wages": float(wage_default), "initial_estimate": initial_default},
        {"weekly_wages": 1.0},  # invalid: cleaned to missing, so it is not a default
    )
    assert features["wage_is_default"].tolist() == [1, 0]
    assert features["initial_is_default"].tolist() == [1, 0]


def test_missing_wage_gives_missing_derived_features(cfg, make_claims):
    features = engineered(cfg, make_claims, {"weekly_wages": 1.0})
    assert (
        features[["log_weekly_wages", "hourly_wage", "initial_to_wage_ratio"]].isna().all(axis=None)
    )


def test_engineering_never_uses_the_target(cfg, make_claims):
    claims = clean_claims(make_claims(), cfg).data
    with_target = engineer_features(claims, cfg)
    without = engineer_features(claims.drop(columns=cfg.col("target")), cfg)
    shared = without.columns
    pd.testing.assert_frame_equal(with_target[shared], without)


# ---------------------------------------------------------------------------
# Group-median imputation
# ---------------------------------------------------------------------------


def imputer_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"group": ["P", "P", "F", "F", None], "hours": [10.0, 20.0, 38.0, np.nan, np.nan]}
    )


def test_group_median_imputer_uses_group_then_overall_median():
    imputer = GroupMedianImputer("group", ["hours"], ["hours"], ["hours_was_missing"])
    out = imputer.fit_transform(imputer_frame())
    assert out["hours"].tolist() == [10.0, 20.0, 38.0, 38.0, 20.0]
    assert out["hours_was_missing"].tolist() == [0, 0, 0, 1, 1]


def test_group_median_imputer_learns_only_from_fit_rows():
    train = imputer_frame().iloc[:3]
    unseen = pd.DataFrame({"group": ["F"], "hours": [np.nan]})
    imputer = GroupMedianImputer("group", ["hours"], [], []).fit(train)
    extra = pd.concat([train, pd.DataFrame({"group": ["F"], "hours": [1000.0]})])
    assert imputer.transform(unseen)["hours"].iloc[0] == 38.0
    assert (
        GroupMedianImputer("group", ["hours"], [], []).fit(extra).transform(unseen)["hours"].iloc[0]
        != 38.0
    )


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------


def test_pipeline_output_is_complete_and_excludes_target(cfg, small_claims):
    pipe = build_preprocessor(cfg, NO_EMBEDDINGS).fit(small_claims)
    matrix = pipe.transform(small_claims)
    names = feature_names(pipe)
    assert matrix.shape == (len(small_claims), len(names))
    assert not np.isnan(matrix).any()
    assert cfg.col("target") not in model_input_columns(cfg)
    assert not any(cfg.col("target") in name for name in names)


@pytest.mark.parametrize("layer", ["rules", "tfidf_svd"])
def test_layer_switch_removes_its_columns(cfg, small_claims, layer):
    on = feature_names(build_preprocessor(cfg, NO_EMBEDDINGS).fit(small_claims))
    off = feature_names(build_preprocessor(cfg, {**NO_EMBEDDINGS, layer: False}).fit(small_claims))
    assert any(name.startswith(layer) for name in on)
    assert not any(name.startswith(layer) for name in off)


def test_embedding_layer_with_fake_embeddings(cfg, small_claims, fake_embeddings):
    layers = {"rules": False, "tfidf_svd": False, "embedding_clusters": True}
    pipe = build_preprocessor(cfg, layers).fit(small_claims)
    cluster_columns = [n for n in feature_names(pipe) if n.startswith("embedding_clusters")]
    assert 1 <= len(cluster_columns) <= cfg["embeddings"]["n_clusters"]


def test_unknown_layer_name_is_rejected(cfg):
    with pytest.raises(ValueError, match="Unknown text layers"):
        build_preprocessor(cfg, {"not_a_layer": True})


def test_every_model_input_is_documented(cfg):
    f = cfg["features"]
    names = f["numeric"] + f["log_numeric"] + f["binary"] + f["categorical"]
    names += f["text_rule_categorical"] + f["text_rule_numeric"]
    assert set(names) <= set(FEATURE_DOCS)
