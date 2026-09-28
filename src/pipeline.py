"""Command-line entry point for every pipeline stage.

Usage:
    python -m src.pipeline --stage validate
    python -m src.pipeline --stage all
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from src.analysis.eda import run_eda
from src.config import Config, load_config
from src.data.clean import clean_claims, save_clean
from src.data.load import load_raw
from src.data.validate import run_validation
from src.explain.shap_report import run_explain
from src.features.report import run_features
from src.logging_setup import setup_logging
from src.models.train_classifier import run_train_classifier
from src.models.train_regression import run_train_regression
from src.reporting.documents import build_documents

logger = logging.getLogger(__name__)

StageFn = Callable[[Config], None]


def run_clean(cfg: Config) -> None:
    """Apply the deterministic cleaning rules and save the cleaned table."""
    result = clean_claims(load_raw(cfg), cfg)
    save_clean(result.data, cfg)


# Execution order for --stage all.
STAGES: dict[str, StageFn] = {
    "validate": run_validation,
    "clean": run_clean,
    "eda": run_eda,
    "features": run_features,
    "train_reg": run_train_regression,
    "train_clf": run_train_classifier,
    "explain": run_explain,
    "report": build_documents,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Workers' comp claim cost pipeline.")
    parser.add_argument(
        "--stage",
        required=True,
        nargs="+",
        choices=[*STAGES, "all"],
        help="Stage(s) to run, in the order given; 'all' runs every stage in order.",
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to a config YAML file.")
    parser.add_argument("--log-level", default=None, help="Override the configured log level.")
    return parser.parse_args(argv)


def run_stages(cfg: Config, stages: Sequence[str]) -> None:
    """Run the given stages in order; ``all`` expands to every stage."""
    names = [name for stage in stages for name in (STAGES if stage == "all" else [stage])]
    for name in names:
        logger.info("=== Stage: %s ===", name)
        STAGES[name](cfg)
        logger.info("=== Stage %s finished ===", name)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested stage(s) and return a process exit code."""
    args = parse_args(argv)
    cfg = load_config(args.config)
    setup_logging(cfg, level=args.log_level)
    logger.info("Random seed: %d", cfg.random_seed)
    try:
        run_stages(cfg, args.stage)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
