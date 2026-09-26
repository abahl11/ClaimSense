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

from src.config import Config, load_config
from src.data.load import compare_columns, load_raw, log_overview
from src.logging_setup import setup_logging

logger = logging.getLogger(__name__)

StageFn = Callable[[Config], None]


def run_validate(cfg: Config) -> None:
    """Load the raw data, log an overview, and check its columns against the config.

    Raises:
        ValueError: If any documented column is missing from the file.
    """
    df = load_raw(cfg)
    log_overview(df, head_rows=cfg["data"]["inspect_head_rows"])
    diff = compare_columns(df.columns, cfg["data"]["expected_columns"])
    if diff["unexpected"]:
        logger.warning("Columns in the file but not in the config: %s", diff["unexpected"])
    if diff["missing"]:
        raise ValueError(f"Columns in the config but missing from the file: {diff['missing']}")
    logger.info("All %d expected columns are present.", len(cfg["data"]["expected_columns"]))


def _planned(stage: str, phase: int) -> StageFn:
    """Return a placeholder for a stage that a later phase will implement."""

    def placeholder(cfg: Config) -> None:
        raise NotImplementedError(f"Stage '{stage}' is not built yet (planned for Phase {phase}).")

    return placeholder


# Execution order for --stage all.
STAGES: dict[str, StageFn] = {
    "validate": run_validate,
    "clean": _planned("clean", 1),
    "eda": _planned("eda", 2),
    "features": _planned("features", 3),
    "train_reg": _planned("train_reg", 5),
    "train_clf": _planned("train_clf", 6),
    "explain": _planned("explain", 7),
    "report": _planned("report", 8),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Workers' comp claim cost pipeline.")
    parser.add_argument(
        "--stage",
        required=True,
        choices=[*STAGES, "all"],
        help="Stage to run; 'all' runs every stage in order.",
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to a config YAML file.")
    parser.add_argument("--log-level", default=None, help="Override the configured log level.")
    return parser.parse_args(argv)


def run_stages(cfg: Config, stage: str) -> None:
    """Run one stage, or every stage in order when ``stage == "all"``."""
    names = list(STAGES) if stage == "all" else [stage]
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
    except (FileNotFoundError, NotImplementedError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
