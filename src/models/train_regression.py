"""Phase 5: regression models for the ultimate claim cost.

Every choice is made with time-ordered cross-validation inside the training period. The test
set is scored once, at the end, for every candidate, so the comparison is transparent but
cannot influence any decision. Order of work (as in the project brief):

1. Baselines: A = training median, B = the adjuster's initial estimate.
2. Ridge, random forest and LightGBM on the log target.
3. LightGBM with Tweedie (several variance powers) and Gamma losses on the raw cost.
4. Price-drift approaches A-D (``src.models.targets``) for the best log-target model.
5. Text ablation: structured fields alone vs. adding the keyword-rule features.
6. Optuna tuning of the chosen configuration.
7. Final fit on the whole training period, 80% prediction intervals from quantile models,
   test-set error analysis, and the saved model artifact.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.config import Config
from src.evaluation.error_analysis import analysis_frame, bias_by_year, error_tables
from src.evaluation.metrics import decile_calibration, log_metrics, regression_metrics
from src.features.build import LAYERS, select_columns
from src.features.tabular import accident_year, fractional_years
from src.models.artifacts import CostModel
from src.models.datasets import Datasets, Split, get_datasets
from src.models.estimators import make_regressor
from src.models.split import period_summary
from src.models.targets import APPROACHES, LOG_TARGETS, LogCost, make_target, smearing_factor
from src.models.tune import TuningResult, tune
from src.reporting.build_results import build_results
from src.reporting.model_figures import (
    plot_bias_by_year,
    plot_calibration_deciles,
    plot_ranked_scores,
    plot_timeline,
)
from src.reporting.outputs import write_json

logger = logging.getLogger(__name__)

BASELINE_FAMILIES = ("median", "initial")
FAMILY_LABELS = {"ridge": "Ridge", "random_forest": "Random forest", "lightgbm": "LightGBM"}
# Ablation feature sets, from simplest to richest.
LAYER_SETS = {
    "No text features": {"rules": False},
    "Keyword rules": {"rules": True},
}


@dataclass(frozen=True)
class Candidate:
    """One model configuration: estimator family, target transform, settings, text features."""

    name: str
    group: str
    family: str
    target: str = "log"
    params: tuple[tuple[str, Any], ...] = ()
    layers: tuple[tuple[str, bool], ...] = ()
    include_gender: bool | None = None  # None follows fairness.exclude_gender in the config

    def key(self) -> tuple[Any, ...]:
        """Identity of the model (switched-on text features are the default, so ignored)."""
        dropped = tuple(sorted(name for name, keep in self.layers if not keep))
        return self.family, self.target, tuple(sorted(self.params)), dropped, self.include_gender

    def describe(self) -> dict[str, Any]:
        """Plain description for the metrics file."""
        return {
            "name": self.name,
            "group": self.group,
            "family": self.family,
            "target": self.target,
            "approach": APPROACHES.get(self.target, "raw cost"),
            "params": dict(self.params),
            "layers": {layer: dict(self.layers).get(layer, True) for layer in LAYERS},
        }


@dataclass
class Fitted:
    """A fitted regressor with its target transform and the columns it uses."""

    model: Any
    target: Any
    columns: np.ndarray


@dataclass
class Prediction:
    """Typical (median-type) and expected (mean) predictions for a split's eval rows."""

    typical: np.ndarray
    expected: np.ndarray
    smearing: float = 1.0
    fitted: Fitted | None = None


# ---------------------------------------------------------------------------
# Fitting and predicting one candidate on one split
# ---------------------------------------------------------------------------


def costs(frame: pd.DataFrame, cfg: Config) -> np.ndarray:
    """Actual ultimate costs of a set of claims."""
    return frame[cfg.col("target")].to_numpy(dtype=float)


def dropped_prefixes(cfg: Config, include_gender: bool | None = None) -> tuple[str, ...]:
    """Column prefixes removed from a model: gender's one-hot columns unless it is included.

    Args:
        cfg: Loaded project configuration.
        include_gender: Explicit choice; ``None`` follows ``fairness.exclude_gender``.
    """
    include = not cfg["fairness"]["exclude_gender"] if include_gender is None else include_gender
    return () if include else (f"categorical__{cfg.col('gender')}_",)


def fit_candidate(candidate: Candidate, split: Split, cfg: Config) -> Fitted:
    """Fit the target transform and the regressor on a split's training rows."""
    y = costs(split.fit_frame, cfg)
    target = make_target(candidate.target, cfg).fit(split.fit_frame, y)
    drop = dropped_prefixes(cfg, candidate.include_gender)
    columns = select_columns(split.feature_names, dict(candidate.layers), drop)
    model = make_regressor(candidate.family, cfg, dict(candidate.params))
    model.fit(split.X_fit[:, columns], target.forward(split.fit_frame, y))
    return Fitted(model, target, columns)


def baseline_values(candidate: Candidate, split: Split, cfg: Config) -> np.ndarray:
    """Baseline A (training median) or B (the adjuster's initial estimate)."""
    if candidate.family == "median":
        return np.full(len(split.eval_frame), float(np.median(costs(split.fit_frame, cfg))))
    return split.eval_frame[cfg.col("initial_estimate")].to_numpy(dtype=float)


def out_of_fold_smearing(candidate: Candidate, split: Split, cfg: Config) -> float:
    """Smearing factor from out-of-fold residuals within a split's training rows.

    In-sample residuals of a boosted model are too small, so each residual comes from a model
    that did not see that claim (K-fold within the training period). The folds are random
    rather than time-ordered on purpose: the factor should correct the log retransformation
    only, not absorb part of the price drift, which is handled and tested separately.
    """
    folds = KFold(cfg["regression"]["smearing_folds"], shuffle=True, random_state=cfg.random_seed)
    residuals = []
    for fit_rows, eval_rows in folds.split(split.X_fit):
        part = Split(
            name=f"{split.name} (smearing fold)",
            X_fit=split.X_fit[fit_rows],
            X_eval=split.X_fit[eval_rows],
            fit_frame=split.fit_frame.iloc[fit_rows].reset_index(drop=True),
            eval_frame=split.fit_frame.iloc[eval_rows].reset_index(drop=True),
            feature_names=split.feature_names,
        )
        fitted = fit_candidate(candidate, part, cfg)
        predicted = fitted.model.predict(part.X_eval[:, fitted.columns])
        actual = fitted.target.forward(part.eval_frame, costs(part.eval_frame, cfg))
        residuals.append(actual - predicted)
    return smearing_factor(np.concatenate(residuals))


def predict(
    candidate: Candidate, split: Split, cfg: Config, expected: bool = False, keep: bool = False
) -> Prediction:
    """Fit on a split's training rows and predict its evaluation rows.

    Args:
        candidate: Model configuration.
        split: Matrices and rows to use.
        cfg: Loaded project configuration.
        expected: Also estimate the smearing factor (needed for currency metrics).
        keep: Keep the fitted model in the result.
    """
    if candidate.family in BASELINE_FAMILIES:
        values = baseline_values(candidate, split, cfg)
        return Prediction(values, values)
    fitted = fit_candidate(candidate, split, cfg)
    z = fitted.model.predict(split.X_eval[:, fitted.columns])
    typical = fitted.target.inverse(split.eval_frame, z)
    smearing = 1.0
    if expected and candidate.target in LOG_TARGETS:
        smearing = out_of_fold_smearing(candidate, split, cfg)
    mean = fitted.target.inverse(split.eval_frame, z + np.log(smearing))
    return Prediction(typical, mean, smearing, fitted if keep else None)


class Evaluator:
    """Cross-validation and test scores for candidates, each computed once and cached."""

    def __init__(self, datasets: Datasets, cfg: Config) -> None:
        self.data = datasets
        self.cfg = cfg
        self._cv: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._test: dict[tuple[Any, ...], tuple[dict[str, float], Prediction]] = {}

    def cv(self, candidate: Candidate) -> dict[str, Any]:
        """Mean (and per-fold) log-scale metrics over the time-ordered folds."""
        key = candidate.key()
        if key not in self._cv:
            start = time.perf_counter()
            folds = [
                log_metrics(
                    costs(split.eval_frame, self.cfg), predict(candidate, split, self.cfg).typical
                )
                for split in self.data.folds
            ]
            table = pd.DataFrame(folds)
            self._cv[key] = {
                "rmsle": float(table["rmsle"].mean()),
                "rmsle_std": float(table["rmsle"].std(ddof=0)),
                "r2_log": float(table["r2_log"].mean()),
                "mean_log_error": float(table["mean_log_error"].mean()),
                "folds": folds,
            }
            logger.info(
                "CV  %-52s RMSLE %.4f (%.0fs)",
                candidate.name,
                self._cv[key]["rmsle"],
                time.perf_counter() - start,
            )
        return self._cv[key]

    def rmsle(self, candidate: Candidate) -> float:
        """Mean cross-validated RMSLE (the selection metric)."""
        return self.cv(candidate)["rmsle"]

    def test(self, candidate: Candidate, keep: bool = False) -> tuple[dict[str, float], Prediction]:
        """Fit on the whole training period and score the test set (once per candidate)."""
        key = candidate.key()
        cached = self._test.get(key)
        if cached is None or (keep and cached[1].fitted is None):
            split = self.data.final
            prediction = predict(candidate, split, self.cfg, expected=True, keep=keep)
            metrics = regression_metrics(
                costs(split.eval_frame, self.cfg), prediction.typical, prediction.expected
            )
            metrics["smearing"] = prediction.smearing
            self._test[key] = (metrics, prediction)
            logger.info("TEST %-51s RMSLE %.4f", candidate.name, metrics["rmsle"])
        return self._test[key]


# ---------------------------------------------------------------------------
# Candidate lists and selection rules
# ---------------------------------------------------------------------------


def stage_one(cfg: Config) -> list[Candidate]:
    """Baselines, the three log-target model families and the raw-cost losses."""
    powers = cfg["regression"]["tweedie_variance_powers"]
    return [
        Candidate("Baseline A: training median", "baseline", "median", "raw"),
        Candidate("Baseline B: adjuster's initial estimate", "baseline", "initial", "raw"),
        Candidate("Ridge (log target)", "family", "ridge"),
        Candidate("Random forest (log target)", "family", "random_forest"),
        Candidate("LightGBM (log target)", "family", "lightgbm"),
        *[
            Candidate(
                f"LightGBM Tweedie, power {p:g} (raw cost)",
                "loss",
                "tweedie",
                "raw",
                (("tweedie_variance_power", p),),
            )
            for p in powers
        ],
        Candidate("LightGBM Gamma (raw cost)", "loss", "gamma", "raw"),
    ]


def best_by_cv(candidates: list[Candidate], evaluator: Evaluator) -> Candidate:
    """Candidate with the lowest cross-validated RMSLE."""
    return min(candidates, key=evaluator.rmsle)


def drift_candidates(base: Candidate) -> list[Candidate]:
    """The base model under each price-drift approach A-D."""
    label = FAMILY_LABELS[base.family]
    return [
        replace(base, name=f"{label}, {APPROACHES[t]}", group="drift", target=t)
        for t in LOG_TARGETS
    ]


def ablation_candidates(base: Candidate) -> list[Candidate]:
    """The base model with and without the keyword-rule features."""
    return [
        replace(base, name=label, group="ablation", layers=tuple(sorted(layers.items())))
        for label, layers in LAYER_SETS.items()
    ]


def choose_layers(candidates: list[Candidate], evaluator: Evaluator, tolerance: float) -> Candidate:
    """Simplest feature set, unless a richer one beats it by more than ``tolerance``."""
    chosen = candidates[0]
    for candidate in candidates[1:]:
        if evaluator.rmsle(candidate) < evaluator.rmsle(chosen) - tolerance:
            chosen = candidate
    return chosen


def tuned(base: Candidate, evaluator: Evaluator, cfg: Config) -> tuple[Candidate, TuningResult]:
    """Search the settings of ``base``'s model family by cross-validated RMSLE."""

    def score(params: dict[str, Any]) -> float:
        return evaluator.rmsle(replace(base, params=tuple(sorted(params.items()))))

    result = tune(score, base.family, cfg)
    best = replace(
        base,
        name=f"{FAMILY_LABELS[base.family]}, tuned ({APPROACHES[base.target][0]}, {base.name})",
        group="tuned",
        params=tuple(sorted(result.best_params.items())),
    )
    return best, result


# ---------------------------------------------------------------------------
# Final model: intervals, error analysis, artifact
# ---------------------------------------------------------------------------


def quantile_models(
    final: Candidate, fitted: Fitted, split: Split, cfg: Config
) -> tuple[dict[float, Any], Any]:
    """Quantile regressors on the final model's (log-type) target, fit on training rows."""
    y = costs(split.fit_frame, cfg)
    target = fitted.target if final.target in LOG_TARGETS else LogCost().fit(split.fit_frame, y)
    params = dict(final.params) if final.family == "lightgbm" else {}
    z = target.forward(split.fit_frame, y)
    X = split.X_fit[:, fitted.columns]
    models = {
        q: make_regressor("quantile", cfg, {**params, "alpha": q}).fit(X, z)
        for q in cfg["regression"]["interval_quantiles"]
    }
    return models, target


def interval_summary(
    models: dict[float, Any], target: Any, columns: np.ndarray, split: Split, cfg: Config
) -> dict[str, Any]:
    """Coverage and width of the prediction interval on the evaluation rows."""
    frame, X = split.eval_frame, split.X_eval[:, columns]
    y = costs(frame, cfg)
    predicted = {q: target.inverse(frame, model.predict(X)) for q, model in models.items()}
    low_q, high_q = min(models), max(models)
    bounds = np.sort(np.column_stack([predicted[low_q], predicted[high_q]]), axis=1)
    inside = (y >= bounds[:, 0]) & (y <= bounds[:, 1])
    years = accident_year(frame, cfg).to_numpy()
    by_year = pd.Series(inside).groupby(years).mean()
    median_q = sorted(models)[len(models) // 2]
    return {
        "nominal_coverage": high_q - low_q,
        "coverage": float(inside.mean()),
        "below_interval": float((y < bounds[:, 0]).mean()),
        "above_interval": float((y > bounds[:, 1]).mean()),
        "coverage_by_year": {int(year): float(share) for year, share in by_year.items()},
        "median_width_ratio": float(np.median(bounds[:, 1] / bounds[:, 0])),
        "median_model_rmsle": log_metrics(y, predicted[median_q])["rmsle"],
    }


def build_artifact(
    final: Candidate,
    prediction: Prediction,
    quantiles: tuple[dict[float, Any], Any],
    split: Split,
    summary: dict[str, Any],
) -> CostModel:
    """Bundle the fitted pieces of the final model for the app and batch scoring."""
    fitted = prediction.fitted
    return CostModel(
        preprocessor=split.preprocessor,
        columns=fitted.columns,
        regressor=fitted.model,
        target=fitted.target,
        smearing=prediction.smearing,
        quantile_models=quantiles[0],
        quantile_target=quantiles[1],
        feature_names=[split.feature_names[i] for i in fitted.columns],
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Validation-design summary
# ---------------------------------------------------------------------------


def split_periods(split: Split, cfg: Config) -> dict[str, Any]:
    """Dates, sizes and median cost of a split's training and evaluation rows."""
    date = cfg.col(cfg["split"]["date_column"])
    return {
        "fit": {**period_summary(split.fit_frame[date]), "claims": len(split.fit_frame)},
        "eval": {
            **period_summary(split.eval_frame[date]),
            "claims": len(split.eval_frame),
            "median_cost": float(np.median(costs(split.eval_frame, cfg))),
        },
    }


def validation_summary(data: Datasets, cfg: Config) -> dict[str, Any]:
    """Train/test split, folds and how different the test period is."""
    train, test = data.final.fit_frame, data.final.eval_frame
    y_train, y_test = costs(train, cfg), costs(test, cfg)
    threshold = float(np.quantile(y_train, cfg["labels"]["high_cost_quantile"]))
    return {
        "cutoff": data.cutoff,
        "cv_folds": len(data.folds),
        "inner_holdout_fraction": cfg["split"]["inner_holdout_fraction"],
        "train": {
            "claims": len(train),
            "median_cost": float(np.median(y_train)),
            "mean_cost": float(y_train.mean()),
        },
        "test": {
            "claims": len(test),
            "median_cost": float(np.median(y_test)),
            "mean_cost": float(y_test.mean()),
            "share_above_train_high_cost_threshold": float((y_test >= threshold).mean()),
        },
        "high_cost_threshold": threshold,
        "high_cost_quantile": cfg["labels"]["high_cost_quantile"],
        "folds": [split_periods(split, cfg) for split in data.folds],
        "final": split_periods(data.final, cfg),
    }


def timeline_frame(data: Datasets, cfg: Config) -> pd.DataFrame:
    """Decimal-year spans of every fold and of the final split, for the timeline figure."""
    date = cfg.col(cfg["split"]["date_column"])
    rows = []
    for label, split in [
        *[(f"CV fold {i}", s) for i, s in enumerate(data.folds, 1)],
        ("Final: train / test", data.final),
    ]:
        fit_years, eval_years = fractional_years(split.fit_frame[date]), fractional_years(
            split.eval_frame[date]
        )
        rows.append(
            {
                "label": label,
                "fit_start": fit_years.min(),
                "fit_end": fit_years.max(),
                "eval_start": eval_years.min(),
                "eval_end": eval_years.max(),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------


def candidate_record(candidate: Candidate, evaluator: Evaluator) -> dict[str, Any]:
    """Description, CV scores and test scores of one candidate."""
    cv = evaluator.cv(candidate)
    return {
        **candidate.describe(),
        "model_key": repr(candidate.key()),
        "cv": {k: v for k, v in cv.items() if k != "folds"},
        "cv_folds": cv["folds"],
        "test": evaluator.test(candidate)[0],
    }


def select_final(evaluator: Evaluator, cfg: Config) -> dict[str, Any]:
    """Run stages 1-6 and return every candidate list plus the chosen configuration."""
    first = stage_one(cfg)
    log_models = [c for c in first if c.group == "family"]
    best_family = best_by_cv(log_models, evaluator)
    drift = drift_candidates(best_family)
    best_drift = best_by_cv(drift, evaluator)
    ablation = ablation_candidates(best_drift)
    best_layers = choose_layers(ablation, evaluator, cfg["regression"]["ablation_tolerance"])
    tuned_candidate, tuning = tuned(best_layers, evaluator, cfg)
    final = (
        tuned_candidate
        if evaluator.rmsle(tuned_candidate) < evaluator.rmsle(best_layers)
        else best_layers
    )
    for candidate in first:
        evaluator.cv(candidate)
    return {
        "stage_one": first,
        "drift": drift,
        "ablation": ablation,
        "tuned": [tuned_candidate],
        "tuning": tuning,
        "choices": {
            "best_family": best_family.name,
            "best_drift": best_drift.name,
            "best_layers": best_layers.name,
            "final": final.name,
        },
        "final": final,
    }


def unique_candidates(groups: list[list[Candidate]]) -> list[Candidate]:
    """All candidates, first occurrence of each distinct model only."""
    seen, ordered = set(), []
    for candidate in (c for group in groups for c in group):
        if candidate.key() not in seen:
            seen.add(candidate.key())
            ordered.append(candidate)
    return ordered


def final_report(
    final: Candidate, evaluator: Evaluator, data: Datasets, cfg: Config
) -> tuple[dict[str, Any], CostModel]:
    """Test metrics, calibration, intervals and error analysis for the final model."""
    split = data.final
    metrics, prediction = evaluator.test(final, keep=True)
    y = costs(split.eval_frame, cfg)
    quantiles = quantile_models(final, prediction.fitted, split, cfg)
    interval = interval_summary(*quantiles, prediction.fitted.columns, split, cfg)
    errors = error_tables(
        analysis_frame(split.eval_frame, y, prediction.typical, prediction.expected, cfg), cfg
    )
    deciles = decile_calibration(y, prediction.expected, cfg["evaluation"]["n_deciles"])
    trend = getattr(prediction.fitted.target, "slope_", None)
    report = {
        "name": final.name,
        "description": final.describe(),
        "test": metrics,
        "smearing": prediction.smearing,
        "trend_annual_growth": float(np.expm1(trend)) if trend is not None else None,
        "calibration_deciles": deciles.to_dict(orient="records"),
        "interval": interval,
        "error_analysis": {name: table.to_dict(orient="records") for name, table in errors.items()},
        "n_features": len(prediction.fitted.columns),
    }
    summary = {
        "name": final.name,
        **final.describe(),
        "test": metrics,
        "interval_coverage": interval["coverage"],
    }
    return report, build_artifact(final, prediction, quantiles, split, summary)


def drift_by_year(
    drift: list[Candidate], evaluator: Evaluator, data: Datasets, cfg: Config
) -> dict[str, dict[int, float]]:
    """Test-set bias by accident year for each drift approach."""
    split = data.final
    y = costs(split.eval_frame, cfg)
    return {
        c.name: bias_by_year(split.eval_frame, cfg, y, evaluator.test(c)[1].expected) for c in drift
    }


def write_figures(
    selection: dict[str, Any],
    evaluator: Evaluator,
    data: Datasets,
    report: dict[str, Any],
    by_year: dict[str, dict[int, float]],
    cfg: Config,
) -> None:
    """All Phase 5 figures."""
    plot_timeline(timeline_frame(data, cfg), cfg)
    comparison = pd.DataFrame(
        [
            {"name": c.name, "rmsle": evaluator.rmsle(c)}
            for c in unique_candidates([selection["stage_one"], selection["tuned"]])
        ]
    )
    plot_ranked_scores(
        comparison,
        "rmsle",
        "Cross-validated RMSE on the log scale (lower is better)",
        "regression_model_comparison",
        cfg,
    )
    ablation = pd.DataFrame(
        [{"name": c.name, "rmsle": evaluator.rmsle(c)} for c in selection["ablation"]]
    )
    plot_ranked_scores(
        ablation,
        "rmsle",
        "Text ablation: cross-validated RMSE on the log scale",
        "regression_ablation",
        cfg,
        "{:.4f}",
    )
    plot_bias_by_year(by_year, cfg)
    plot_calibration_deciles(pd.DataFrame(report["calibration_deciles"]), cfg)


def run_train_regression(cfg: Config) -> None:
    """Stage ``train_reg``: select, tune, test and save the cost model."""
    data = get_datasets(cfg)
    evaluator = Evaluator(data, cfg)
    selection = select_final(evaluator, cfg)
    final = selection["final"]
    everything = [
        c for name in ("stage_one", "drift", "ablation", "tuned") for c in selection[name]
    ]
    report, artifact = final_report(final, evaluator, data, cfg)
    by_year = drift_by_year(selection["drift"], evaluator, data, cfg)
    tuning = selection["tuning"]
    metrics = {
        "selection_metric": cfg["evaluation"]["selection_metric"],
        "candidates": [candidate_record(c, evaluator) for c in everything],
        "groups": {
            name: [c.name for c in selection[name]]
            for name in ("stage_one", "drift", "ablation", "tuned")
        },
        "choices": selection["choices"],
        "drift_bias_by_year": by_year,
        "final": report,
    }
    write_json(validation_summary(data, cfg), cfg.path("validation_metrics"))
    write_json(metrics, cfg.path("regression_metrics"))
    if tuning is not None:
        default = selection["tuned"][0]
        write_json(
            {
                "best_params": tuning.best_params,
                "best_cv_rmsle": tuning.best_score,
                "untuned_cv_rmsle": evaluator.rmsle(replace(default, params=())),
                "n_trials": len(tuning.trials),
                "trials": tuning.trials.to_dict(orient="records"),
            },
            cfg.path("best_params"),
        )
    write_figures(selection, evaluator, data, report, by_year, cfg)
    path = cfg.path("regression_model")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path, compress=3)
    logger.info("Saved the cost model to %s", path)
    build_results(cfg)
