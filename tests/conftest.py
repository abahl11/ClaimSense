"""Shared test fixtures."""

from collections.abc import Callable

import pandas as pd
import pytest

from src.config import Config, load_config

# One plausible claim, keyed by logical column name. Tests override single fields.
VALID_CLAIM = {
    "accident_date": "2001-03-05T10:00:00Z",
    "report_date": "2001-03-20T00:00:00Z",
    "age": 35,
    "gender": "M",
    "marital_status": "M",
    "dependent_children": 0,
    "dependents_other": 0,
    "weekly_wages": 450.0,
    "part_time": "F",
    "hours_per_week": 38.0,
    "days_per_week": 5,
    "description": "LIFTING BOX LOWER BACK STRAIN",
    "initial_estimate": 1500,
    "target": 2100.5,
}


@pytest.fixture(scope="session")
def cfg() -> Config:
    return load_config()


@pytest.fixture
def make_claims(cfg: Config) -> Callable[..., pd.DataFrame]:
    """Build a raw-format claims table: one row per dict of logical-name overrides."""

    def _make(*overrides: dict) -> pd.DataFrame:
        rows = []
        for i, override in enumerate(overrides or ({},)):
            claim = {"claim_id": f"WC{i:07d}", **VALID_CLAIM, **override}
            rows.append({cfg.col(name): value for name, value in claim.items()})
        return pd.DataFrame(rows)

    return _make
