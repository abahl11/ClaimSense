"""Leakage checks: the target is never an input, and learned steps see training rows only."""

import numpy as np
import pandas as pd
import pytest

from src.features.build import LAYERS, build_preprocessor, feature_names, model_input_columns
from src.features.tabular import engineer_features

ALL_LAYERS_OFF = {layer: False for layer in LAYERS}


def split_small(claims):
    """First 45 rows as 'training', last 15 as 'test' (the fixture is in date order)."""
    return claims.iloc[:45].reset_index(drop=True), claims.iloc[45:].reset_index(drop=True)


@pytest.mark.parametrize("layers", [ALL_LAYERS_OFF, {}, {"embedding_clusters": False}])
def test_target_is_never_a_model_input(cfg, layers):
    inputs = model_input_columns(cfg, layers)
    assert cfg.col("target") not in inputs


def test_target_values_do_not_change_the_features(cfg, small_claims):
    train, test = split_small(small_claims)
    pipe = build_preprocessor(cfg, {"embedding_clusters": False}).fit(train)
    shuffled = test.copy()
    shuffled[cfg.col("target")] = shuffled[cfg.col("target")].sample(frac=1, random_state=0).values
    np.testing.assert_array_equal(pipe.transform(test), pipe.transform(shuffled))
    assert not any(cfg.col("target") in name for name in feature_names(pipe))


def test_group_medians_are_learned_from_training_rows_only(cfg, small_claims):
    train, test = split_small(small_claims)
    test.loc[:, cfg.col("weekly_wages")] = 1e6  # extreme test values must not move the medians
    pipe = build_preprocessor(cfg, {"embedding_clusters": False}).fit(train)
    group = cfg.col(cfg["features"]["impute_group"])
    expected = engineer_features(train, cfg).groupby(group)["log_weekly_wages"].median()
    learned = pipe.named_steps["impute"].group_medians_["log_weekly_wages"]
    assert learned.to_dict() == pytest.approx(expected.to_dict())
    refit = build_preprocessor(cfg, {"embedding_clusters": False}).fit(pipe_input(train, test))
    assert refit.named_steps["impute"].group_medians_["log_weekly_wages"].max() > learned.max()


def pipe_input(train, test):
    """Training and test rows together (only used to show that medians would move)."""
    return pd.concat([train, test], ignore_index=True)


def test_scaler_statistics_come_from_training_rows(cfg, small_claims):
    train, _ = split_small(small_claims)
    pipe = build_preprocessor(cfg, {"embedding_clusters": False}).fit(train)
    scaler = pipe.named_steps["columns"].named_transformers_["numeric"].named_steps["scale"]
    age_position = cfg.resolve(cfg["features"]["numeric"]).index(cfg.col("age"))
    assert scaler.mean_[age_position] == pytest.approx(train[cfg.col("age")].mean())


def test_tfidf_vocabulary_excludes_words_seen_only_in_test(cfg, small_claims):
    train, test = split_small(small_claims)
    text = cfg.col("description")
    test.loc[:, text] = "ZEBRAWORD " * 10
    pipe = build_preprocessor(cfg, {"embedding_clusters": False}).fit(train)
    tfidf = pipe.named_steps["columns"].named_transformers_["tfidf_svd"].named_steps["tfidf"]
    assert "zebraword" not in tfidf.vocabulary_


def test_kmeans_is_fit_on_training_descriptions_only(cfg, small_claims, fake_embeddings):
    train, test = split_small(small_claims)
    layers = {"rules": False, "tfidf_svd": False, "embedding_clusters": True}
    build_preprocessor(cfg, layers).fit(train)
    fitted_texts = set(fake_embeddings[0])
    assert fitted_texts == set(train[cfg.col("description")])
    assert fitted_texts.isdisjoint(test[cfg.col("description")])
