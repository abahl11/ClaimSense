"""Read the raw Kaggle file and confirm its schema against the config."""

from __future__ import annotations

import logging
from collections.abc import Iterable

import pandas as pd

from src.config import Config

logger = logging.getLogger(__name__)


def load_raw(cfg: Config) -> pd.DataFrame:
    """Read the raw CSV exactly as stored, without parsing or type coercion.

    Dtype problems are left in place on purpose so the validate stage can report them.

    Args:
        cfg: Loaded project configuration.

    Returns:
        The raw claims table.

    Raises:
        FileNotFoundError: If the raw file is not at ``paths.raw_data``.
    """
    path = cfg.path("raw_data")
    if not path.is_file():
        raise FileNotFoundError(
            f"Raw data not found at {path}. Download train.csv from "
            f"{cfg['data']['source_url']} and place it at that path."
        )
    df = pd.read_csv(path)
    logger.info("Loaded %d rows x %d columns from %s", df.shape[0], df.shape[1], path)
    return df


def compare_columns(actual: Iterable[str], expected: Iterable[str]) -> dict[str, list[str]]:
    """List the columns missing from, and unexpected in, the data.

    Args:
        actual: Column names found in the file.
        expected: Column names documented in the config.

    Returns:
        A dict with sorted ``"missing"`` and ``"unexpected"`` column lists.
    """
    actual_set, expected_set = set(actual), set(expected)
    return {
        "missing": sorted(expected_set - actual_set),
        "unexpected": sorted(actual_set - expected_set),
    }


def load_clean(cfg: Config) -> pd.DataFrame:
    """Read the cleaned Parquet table written by the clean stage.

    Raises:
        FileNotFoundError: If the clean stage has not been run yet.
    """
    path = cfg.path("clean_data")
    if not path.is_file():
        raise FileNotFoundError(f"Cleaned data not found at {path}. Run --stage clean first.")
    df = pd.read_parquet(path)
    logger.info("Loaded %d cleaned rows from %s", len(df), path)
    return df


def log_overview(df: pd.DataFrame, head_rows: int) -> None:
    """Log the shape, column dtypes and first rows of a DataFrame.

    Args:
        df: Table to describe.
        head_rows: Number of leading rows to log.
    """
    logger.info("Shape: %d rows x %d columns", df.shape[0], df.shape[1])
    logger.info("Dtypes:\n%s", df.dtypes.to_string())
    head = df.head(head_rows)
    with pd.option_context("display.width", 250, "display.max_columns", None):
        logger.info("First %d rows:\n%s", len(head), head.to_string())
