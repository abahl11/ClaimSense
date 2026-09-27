"""Inference for new claims: clean, predict cost and interval, flag high-cost, explain.

Used by the Streamlit app, and for batch scoring from the command line::

    python -m src.models.predict --input new_claims.csv --output scored_claims.csv

The input has the same columns as the Kaggle ``train.csv`` (the ultimate cost is optional).
New claims go through exactly the same cleaning rules and fitted preprocessing as training.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import Config, load_config
from src.data.clean import clean_claims
from src.explain.naming import describe_model_columns, group_contributions
from src.features.tabular import engineer_features
from src.features.text_rules import TAXONOMY_FIELDS
from src.logging_setup import setup_logging
from src.models.artifacts import CostModel, HighCostClassifier

logger = logging.getLogger(__name__)


@dataclass
class Models:
    """The saved cost model and high-cost classifier."""

    cost: CostModel
    classifier: HighCostClassifier


def load_models(cfg: Config) -> Models:
    """Load both saved models.

    Raises:
        FileNotFoundError: If the training stages have not been run.
    """
    for key in ("regression_model", "classifier_model"):
        if not cfg.path(key).is_file():
            raise FileNotFoundError(
                f"{cfg.path(key)} not found. Run --stage train_reg train_clf first."
            )
    return Models(
        joblib.load(cfg.path("regression_model")), joblib.load(cfg.path("classifier_model"))
    )


def clean_new_claims(raw: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Apply the training-time cleaning rules to new claims (in the raw file format)."""
    return clean_claims(raw, cfg).data


def score_claims(cleaned: pd.DataFrame, cfg: Config, models: Models) -> pd.DataFrame:
    """Predicted cost, interval, high-cost probability, flag and text labels per claim.

    Args:
        cleaned: Claims after :func:`clean_new_claims`.
        cfg: Loaded project configuration.
        models: Loaded models.

    Returns:
        One row per claim.
    """
    cost = models.cost.predict_frame(cleaned)
    probability = models.classifier.probabilities(cleaned)
    labels = engineer_features(cleaned, cfg)[list(TAXONOMY_FIELDS)]
    scored = pd.concat([cleaned[[cfg.col("claim_id")]], cost, labels], axis=1)
    scored["high_cost_probability"] = probability
    scored["flag_for_review"] = probability >= models.classifier.threshold
    return scored


def make_explainer(models: Models) -> Any:
    """SHAP TreeExplainer for the cost model (create once and reuse)."""
    import shap

    return shap.TreeExplainer(models.cost.regressor)


def explain_claim(
    cleaned: pd.DataFrame, cfg: Config, models: Models, explainer: Any, top_n: int
) -> pd.DataFrame:
    """The strongest drivers of one claim's prediction, summed per business concept.

    Args:
        cleaned: One cleaned claim (a single-row frame).
        cfg: Loaded project configuration.
        models: Loaded models.
        explainer: Output of :func:`make_explainer`.
        top_n: Number of drivers to return.

    Returns:
        Columns ``driver``, ``shap`` (log-scale contribution) and ``factor`` (multiplicative
        effect on the predicted cost), strongest first.
    """
    values = np.asarray(explainer.shap_values(models.cost.matrix(cleaned)))
    groups = [group for _, group, _ in describe_model_columns(models.cost, cfg)]
    contributions = group_contributions(values, groups).iloc[0]
    top = contributions.reindex(contributions.abs().sort_values(ascending=False).index).head(top_n)
    return pd.DataFrame(
        {"driver": top.index, "shap": top.to_numpy(), "factor": np.exp(top.to_numpy())}
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command-line arguments for batch scoring."""
    parser = argparse.ArgumentParser(description="Score new workers' comp claims.")
    parser.add_argument("--input", type=Path, required=True, help="CSV in the train.csv format.")
    parser.add_argument("--output", type=Path, required=True, help="Where to write the scores.")
    parser.add_argument("--config", type=Path, default=None, help="Path to a config YAML file.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Score a CSV of claims and write the results."""
    args = parse_args(argv)
    cfg = load_config(args.config)
    setup_logging(cfg)
    try:
        models = load_models(cfg)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    scored = score_claims(clean_new_claims(pd.read_csv(args.input), cfg), cfg, models)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output, index=False)
    logger.info("Scored %d claims -> %s", len(scored), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
