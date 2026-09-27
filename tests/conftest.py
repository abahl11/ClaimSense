"""Shared test fixtures."""

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from src.config import Config, load_config
from src.data.clean import clean_claims
from src.features import text_embed

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


@pytest.fixture
def small_claims(cfg: Config, make_claims: Callable[..., pd.DataFrame]) -> pd.DataFrame:
    """Sixty cleaned claims with varied values, for pipeline-level tests."""
    rng = np.random.default_rng(0)
    texts = [
        "LIFTING BOX STRAINED LOWER BACK",
        "CUT LEFT INDEX FINGER KNIFE",
        "SLIPPED FELL BRUISED KNEE",
    ]
    overrides = [
        {
            "description": f"{texts[i % 3]} CASE {i}",
            "age": int(rng.integers(18, 65)),
            "weekly_wages": float(rng.uniform(200, 900)),
            "initial_estimate": int(rng.integers(100, 20000)),
            "accident_date": f"2001-{1 + i % 12:02d}-05T{i % 24:02d}:00:00Z",
            "report_date": f"2001-{1 + i % 12:02d}-20T00:00:00Z",
            "part_time": "P" if i % 4 == 0 else "F",
        }
        for i in range(60)
    ]
    overrides[0]["hours_per_week"] = 0.0  # becomes missing, then imputed
    return clean_claims(make_claims(*overrides), cfg).data


@pytest.fixture
def fake_embeddings(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Replace the sentence-transformer with deterministic pseudo-embeddings.

    Returns the list of text batches that were embedded, so tests can check which rows a
    fitted step saw.
    """
    calls: list[list[str]] = []

    def fake(texts: list[str], *args: object, **kwargs: object) -> np.ndarray:
        calls.append(list(texts))
        seeds = [sum(map(ord, text)) for text in texts]
        return np.array([np.random.default_rng(seed).normal(size=8) for seed in seeds])

    monkeypatch.setattr(text_embed, "embed_texts", fake)
    return calls
