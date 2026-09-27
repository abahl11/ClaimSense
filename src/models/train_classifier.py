"""Phase 6: flag likely high-cost claims at reporting time for early escalation.

* **Label.** 1 if the ultimate cost is at or above the 90th percentile of the TRAINING target
  (the same cutoff is applied to the test set). Inside cross-validation each fold uses the
  percentile of its own training rows, so the label never borrows from validation data.
* **Candidates.** Rank by the adjuster's initial estimate (baseline); logistic regression;
  LightGBM with class weighting for the 1-in-10 imbalance; and ranking by the regression
  model's predicted cost. The flag uses whichever of the last three has the best
  cross-validated PR-AUC. If a dedicated classifier cannot beat the cost model's own ranking,
  one model serves both purposes.
* **Calibration.** Raw scores (a class-weighted probability, or a predicted cost) are turned
  into probabilities by a logistic calibration fit on out-of-time scores: the scorer is fit on
  the earlier training claims and calibrated on the latest ones.
* **Threshold.** Set from review capacity: the probability at which the configured share of
  the latest training claims would be flagged.
* **Relative label.** Because costs rise over time, far more test claims pass the fixed
  training threshold, and almost any ranking finds them. A second label, "top 10% within its
  own accident year", removes that base-rate shift and is the sharper test of triage skill;
  every candidate is compared on it too.
"""

from __future__ import annotations

import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve

from src.config import Config
from src.evaluation.metrics import (
    brier_score,
    capacity_table,
    classification_metrics,
    reliability_table,
)
from src.features.build import select_columns
from src.features.tabular import accident_year
from src.models.artifacts import HighCostClassifier, Scorer, calibrated, calibration_features
from src.models.datasets import Datasets, Split, get_datasets, inner_split
from src.models.estimators import make_classifier
from src.models.train_regression import Candidate, costs, dropped_prefixes, fit_candidate
from src.reporting.build_results import build_results, load_metrics
from src.reporting.model_figures import plot_pr_curves, plot_reliability
from src.reporting.outputs import write_json

logger = logging.getLogger(__name__)

INITIAL = "Baseline: rank by initial estimate"
REGRESSION = "Rank by regression model's predicted cost"
LOGISTIC = "Logistic regression"
LIGHTGBM = "LightGBM (class-weighted)"
CANDIDATES = (INITIAL, REGRESSION, LOGISTIC, LIGHTGBM)
SELECTABLE = (REGRESSION, LOGISTIC, LIGHTGBM)
CLASSIFIER_FAMILIES = {LOGISTIC: "logistic", LIGHTGBM: "lightgbm"}


def cost_threshold(frame: pd.DataFrame, cfg: Config) -> float:
    """High-cost cutoff: the configured quantile of the given (training) claims."""
    return float(np.quantile(costs(frame, cfg), cfg["labels"]["high_cost_quantile"]))


def labels(frame: pd.DataFrame, threshold: float, cfg: Config) -> np.ndarray:
    """1 for claims at or above the cost threshold."""
    return (costs(frame, cfg) >= threshold).astype(int)


def relative_labels(frame: pd.DataFrame, cfg: Config) -> np.ndarray:
    """1 for claims in the top share of cost *within their own accident year*."""
    years = accident_year(frame, cfg).to_numpy()
    y = costs(frame, cfg)
    cutoffs = pd.Series(y).groupby(years).quantile(cfg["labels"]["high_cost_quantile"])
    return (y >= cutoffs.reindex(years).to_numpy()).astype(int)


def final_candidate(cfg: Config) -> Candidate:
    """The chosen regression configuration, rebuilt from the regression metrics file."""
    record = load_metrics(cfg, "regression_metrics")
    if record is None:
        raise FileNotFoundError("Regression metrics not found. Run --stage train_reg first.")
    d = record["final"]["description"]
    return Candidate(
        name=d["name"],
        group=d["group"],
        family=d["family"],
        target=d["target"],
        params=tuple(sorted(d["params"].items())),
        layers=tuple(sorted(d["layers"].items())),
    )


def fit_scorer(
    name: str, split: Split, y: np.ndarray, regression: Candidate, cfg: Config
) -> Scorer:
    """Fit one learned candidate on a split's training rows.

    Args:
        name: ``REGRESSION``, ``LOGISTIC`` or ``LIGHTGBM``.
        split: Matrices and rows to fit on.
        y: High-cost labels of the split's training rows (unused by the regression ranker).
        regression: The chosen regression configuration (also defines the input columns).
        cfg: Loaded project configuration.
    """
    if name == REGRESSION:
        fitted = fit_candidate(regression, split, cfg)
        return Scorer("cost_rank", fitted.model, fitted.columns, fitted.target)
    columns = select_columns(split.feature_names, dict(regression.layers), dropped_prefixes(cfg))
    weight = float((y == 0).sum() / max((y == 1).sum(), 1))
    model = make_classifier(CLASSIFIER_FAMILIES[name], cfg, weight)
    return Scorer("classifier", model.fit(split.X_fit[:, columns], y), columns)


def scores(
    name: str, split: Split, y_fit: np.ndarray, regression: Candidate, cfg: Config
) -> np.ndarray:
    """Ranking scores for a split's evaluation rows."""
    if name == INITIAL:
        return split.eval_frame[cfg.col("initial_estimate")].to_numpy(dtype=float)
    return fit_scorer(name, split, y_fit, regression, cfg).raw(split.X_eval, split.eval_frame)


def cross_validate(name: str, data: Datasets, regression: Candidate, cfg: Config) -> dict[str, Any]:
    """PR-AUC and ROC-AUC over the time-ordered folds (fixed-threshold label)."""
    folds = []
    for split in data.folds:
        threshold = cost_threshold(split.fit_frame, cfg)
        y_fit, y_eval = labels(split.fit_frame, threshold, cfg), labels(
            split.eval_frame, threshold, cfg
        )
        folds.append(classification_metrics(y_eval, scores(name, split, y_fit, regression, cfg)))
    table = pd.DataFrame(folds)
    result = {
        "pr_auc": float(table["pr_auc"].mean()),
        "pr_auc_std": float(table["pr_auc"].std(ddof=0)),
        "roc_auc": float(table["roc_auc"].mean()),
        "base_rate": float(table["base_rate"].mean()),
        "folds": folds,
    }
    logger.info("CV  %-45s PR-AUC %.4f", name, result["pr_auc"])
    return result


def calibration_stage(
    name: str, data: Datasets, regression: Candidate, cfg: Config
) -> dict[str, Any]:
    """Fit on earlier training claims, calibrate and set the threshold on the latest ones."""
    inner = inner_split(data.final, cfg)
    threshold = cost_threshold(data.final.fit_frame, cfg)
    y_early, y_late = labels(inner.fit_frame, threshold, cfg), labels(
        inner.eval_frame, threshold, cfg
    )
    scorer = fit_scorer(name, inner, y_early, regression, cfg)
    raw = scorer.raw(inner.X_eval, inner.eval_frame)
    calibrator = LogisticRegression().fit(calibration_features(raw, scorer.kind), y_late)
    probabilities = calibrated(calibrator, raw, scorer.kind)
    capacity = cfg["evaluation"]["review_capacity"]
    bins = cfg["evaluation"]["reliability_bins"]
    return {
        "calibrator": calibrator,
        "probability_threshold": float(np.quantile(probabilities, 1 - capacity)),
        "inner": {
            "claims": len(y_late),
            "base_rate": float(y_late.mean()),
            "brier_calibrated": brier_score(y_late, probabilities),
            "reliability_calibrated": reliability_table(y_late, probabilities, bins),
            **(
                {
                    "brier_raw": brier_score(y_late, raw),
                    "reliability_raw": reliability_table(y_late, raw, bins),
                }
                if scorer.kind == "classifier"
                else {}
            ),
        },
    }


def ranking_results(
    split: Split, y_fit: np.ndarray, y_eval: np.ndarray, regression: Candidate, cfg: Config
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Ranking metrics and capacity tables on the evaluation rows for every candidate."""
    capacities = cfg["evaluation"]["review_capacities"]
    results, all_scores = {}, {}
    for name in CANDIDATES:
        score = scores(name, split, y_fit, regression, cfg)
        all_scores[name] = score
        results[name] = {
            **classification_metrics(y_eval, score),
            "capacity": capacity_table(y_eval, score, capacities).to_dict(orient="records"),
        }
    return results, all_scores


def write_figures(
    y_test: np.ndarray,
    test_scores: dict[str, np.ndarray],
    reliability: dict[str, pd.DataFrame],
    cfg: Config,
) -> None:
    """Precision-recall curves and reliability curves."""
    curves = {}
    for name, score in test_scores.items():
        precision, recall, _ = precision_recall_curve(y_test, score)
        curves[name] = (recall, precision)
    plot_pr_curves(curves, float(y_test.mean()), cfg)
    plot_reliability(reliability, cfg)


def run_train_classifier(cfg: Config) -> None:
    """Stage ``train_clf``: choose, calibrate, test and save the high-cost flag."""
    data = get_datasets(cfg)
    regression = final_candidate(cfg)
    cv = {name: cross_validate(name, data, regression, cfg) for name in CANDIDATES}
    chosen = max(SELECTABLE, key=lambda name: cv[name]["pr_auc"])
    calibration = calibration_stage(chosen, data, regression, cfg)
    split = data.final
    threshold = cost_threshold(split.fit_frame, cfg)
    y_train, y_test = labels(split.fit_frame, threshold, cfg), labels(
        split.eval_frame, threshold, cfg
    )
    test, test_scores = ranking_results(split, y_train, y_test, regression, cfg)
    relative, _ = ranking_results(
        split,
        relative_labels(split.fit_frame, cfg),
        relative_labels(split.eval_frame, cfg),
        regression,
        cfg,
    )
    scorer = fit_scorer(chosen, split, y_train, regression, cfg)
    raw_test = scorer.raw(split.X_eval, split.eval_frame)
    probabilities = calibrated(calibration["calibrator"], raw_test, scorer.kind)
    bins = cfg["evaluation"]["reliability_bins"]
    test_reliability = reliability_table(y_test, probabilities, bins)
    inner = calibration["inner"]
    metrics = {
        "cost_threshold": threshold,
        "label_quantile": cfg["labels"]["high_cost_quantile"],
        "review_capacity": cfg["evaluation"]["review_capacity"],
        "base_rate": {"train": float(y_train.mean()), "test": float(y_test.mean())},
        "cv": cv,
        "chosen": chosen,
        "chosen_kind": scorer.kind,
        "test": test,
        "relative_label": relative,
        "threshold": {
            "probability": calibration["probability_threshold"],
            "test": classification_metrics(
                y_test, probabilities, calibration["probability_threshold"]
            ),
        },
        "calibration": {
            "inner_claims": inner["claims"],
            "inner_base_rate": inner["base_rate"],
            "inner_brier_raw": inner.get("brier_raw"),
            "inner_brier_calibrated": inner["brier_calibrated"],
            "test_brier_raw": (
                brier_score(y_test, raw_test) if scorer.kind == "classifier" else None
            ),
            "test_brier_calibrated": brier_score(y_test, probabilities),
            "test_mean_probability": float(probabilities.mean()),
            "reliability_test": test_reliability.to_dict(orient="records"),
        },
    }
    write_json(metrics, cfg.path("classifier_metrics"))
    reliability = {
        "Latest training claims, calibrated": inner["reliability_calibrated"],
        "Test set, calibrated": test_reliability,
    }
    if "reliability_raw" in inner:
        reliability = {"Latest training claims, raw": inner["reliability_raw"], **reliability}
    write_figures(y_test, test_scores, reliability, cfg)
    artifact = HighCostClassifier(
        preprocessor=split.preprocessor,
        scorer=scorer,
        calibrator=calibration["calibrator"],
        threshold=calibration["probability_threshold"],
        cost_threshold=threshold,
        summary={"model": chosen, "review_capacity": cfg["evaluation"]["review_capacity"]},
    )
    path = cfg.path("classifier_model")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path, compress=3)
    logger.info("Saved the high-cost flag (%s) to %s", chosen, path)
    build_results(cfg)
