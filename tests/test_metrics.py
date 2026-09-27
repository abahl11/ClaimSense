"""Tests for src/evaluation/metrics.py, checked against hand-computed values."""

import numpy as np
import pytest

from src.evaluation.metrics import (
    capacity_table,
    classification_metrics,
    decile_calibration,
    log_metrics,
    regression_metrics,
    reliability_table,
    top_positions,
)


def test_perfect_predictions():
    y = np.array([100.0, 1000.0, 10000.0])
    metrics = regression_metrics(y, y, y)
    assert metrics["rmsle"] == pytest.approx(0.0)
    assert metrics["r2_log"] == pytest.approx(1.0)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["bias_ratio"] == pytest.approx(1.0)


def test_log_metrics_by_hand():
    y = np.expm1(np.array([1.0, 2.0, 3.0]))
    pred = np.expm1(np.array([1.5, 2.0, 2.5]))
    metrics = log_metrics(y, pred)
    assert metrics["rmsle"] == pytest.approx(np.sqrt((0.25 + 0 + 0.25) / 3))
    assert metrics["mean_log_error"] == pytest.approx(0.0)
    assert metrics["r2_log"] == pytest.approx(1 - 0.5 / 2.0)


def test_currency_metrics_use_expected_predictions():
    y = np.array([100.0, 300.0])
    typical = np.array([100.0, 300.0])
    expected = np.array([150.0, 350.0])
    metrics = regression_metrics(y, typical, expected)
    assert metrics["rmsle"] == pytest.approx(0.0)
    assert metrics["mae"] == pytest.approx(50.0)
    assert metrics["bias_ratio"] == pytest.approx(500 / 400)


def test_under_prediction_has_negative_log_bias():
    y = np.array([1000.0, 2000.0])
    assert log_metrics(y, y / 2)["mean_log_error"] < 0


def test_decile_calibration_groups_by_prediction():
    y = np.arange(1, 101, dtype=float)
    table = decile_calibration(y, y, 10)
    assert table["claims"].tolist() == [10] * 10
    assert table["ratio"].tolist() == pytest.approx([1.0] * 10)
    assert table["mean_actual"].is_monotonic_increasing


def test_top_positions_rounds_up_and_breaks_ties_by_order():
    scores = np.array([0.1, 0.9, 0.9, 0.5])
    assert top_positions(scores, 0.5).tolist() == [1, 2]
    assert len(top_positions(scores, 0.3)) == 2


def test_capacity_table_recall_and_precision():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    scores = np.array([9, 8, 7, 1, 1, 1, 1, 1, 1, 0], dtype=float)
    table = capacity_table(y, scores, [0.2, 0.3]).set_index("capacity")
    assert table.loc[0.2, "caught"] == 1
    assert table.loc[0.3, "recall"] == pytest.approx(2 / 3)
    assert table.loc[0.3, "precision"] == pytest.approx(2 / 3)
    assert table.loc[0.3, "lift"] == pytest.approx((2 / 3) / 0.3)


def test_classification_metrics_with_threshold():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.6, 0.4, 0.9])
    metrics = classification_metrics(y, scores, threshold=0.5)
    assert metrics["roc_auc"] == pytest.approx(0.75)
    assert metrics["flag_rate"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)


def test_reliability_table_bins():
    probabilities = np.linspace(0, 1, 20)
    y = (probabilities > 0.5).astype(int)
    table = reliability_table(y, probabilities, 4)
    assert table["claims"].tolist() == [5] * 4
    assert table["observed_rate"].iloc[0] == 0 and table["observed_rate"].iloc[-1] == 1
