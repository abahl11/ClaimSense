"""Deployable model objects saved with joblib and used by the app and batch scoring.

Each object bundles everything needed to go from a *cleaned* claim to a prediction: the
fitted preprocessor, the chosen output columns, the model and (for cost) the target
transform, smearing factor and quantile models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


@dataclass
class CostModel:
    """Ultimate-cost model.

    Attributes:
        preprocessor: Fitted preprocessing pipeline (cleaned claims -> full matrix).
        columns: Positions of the matrix columns the regressor uses.
        regressor: Fitted regressor predicting the transformed target.
        target: Fitted target transform (see ``src.models.targets``).
        smearing: Retransformation factor turning typical predictions into expected ones.
        quantile_models: Fitted quantile regressors keyed by quantile (0.1, 0.5, 0.9).
        quantile_target: Target transform the quantile models were trained on.
        feature_names: Names of the used columns, in order.
        summary: Description of the choices behind the model (for reports and the app).
    """

    preprocessor: Pipeline
    columns: np.ndarray
    regressor: Any
    target: Any
    smearing: float
    quantile_models: dict[float, Any]
    quantile_target: Any
    feature_names: list[str]
    summary: dict[str, Any] = field(default_factory=dict)

    def matrix(self, claims: pd.DataFrame) -> np.ndarray:
        """Model input matrix for cleaned claims."""
        return self.preprocessor.transform(claims)[:, self.columns].astype(np.float32)

    def predict_frame(self, claims: pd.DataFrame, X: np.ndarray | None = None) -> pd.DataFrame:
        """Expected and typical cost plus the 80% interval for each claim.

        Args:
            claims: Cleaned claims (index is kept).
            X: Precomputed ``matrix(claims)``, to avoid transforming twice.

        Returns:
            Columns ``expected_cost`` (mean, for reserving), ``typical_cost`` (median-type),
            ``interval_low`` and ``interval_high``.
        """
        X = self.matrix(claims) if X is None else X
        z = self.regressor.predict(X)
        low, high = min(self.quantile_models), max(self.quantile_models)
        bounds = np.sort(
            np.column_stack(
                [
                    self.quantile_target.inverse(claims, self.quantile_models[q].predict(X))
                    for q in (low, high)
                ]
            ),
            axis=1,
        )
        return pd.DataFrame(
            {
                "expected_cost": self.target.inverse(claims, z + np.log(self.smearing)),
                "typical_cost": self.target.inverse(claims, z),
                "interval_low": bounds[:, 0],
                "interval_high": bounds[:, 1],
            },
            index=claims.index,
        )


@dataclass
class Scorer:
    """Something that ranks claims by how likely they are to become high-cost.

    Attributes:
        kind: ``classifier`` (a model with ``predict_proba``) or ``cost_rank`` (a regressor
            whose predicted cost is the score).
        model: The fitted model.
        columns: Positions of the matrix columns the model uses.
        target: For ``cost_rank``, the target transform turning model output into a cost.
    """

    kind: str
    model: Any
    columns: np.ndarray
    target: Any = None

    def raw(self, X: np.ndarray, claims: pd.DataFrame) -> np.ndarray:
        """Uncalibrated score: a probability, or the typical predicted cost."""
        if self.kind == "cost_rank":
            return self.target.inverse(claims, self.model.predict(X[:, self.columns]))
        return self.model.predict_proba(X[:, self.columns])[:, 1]


@dataclass
class HighCostClassifier:
    """High-cost claim flag.

    Attributes:
        preprocessor: Fitted preprocessing pipeline (cleaned claims -> full matrix).
        scorer: The fitted ranking model (see :class:`Scorer`).
        calibrator: Fitted logistic calibrator mapping raw scores to probabilities.
        threshold: Probability at or above which a claim is flagged.
        cost_threshold: Ultimate cost that defines "high cost" (training 90th percentile).
        summary: Description of the choices behind the model.
    """

    preprocessor: Pipeline
    scorer: Scorer
    calibrator: Any
    threshold: float
    cost_threshold: float
    summary: dict[str, Any] = field(default_factory=dict)

    def probabilities(self, claims: pd.DataFrame) -> np.ndarray:
        """Calibrated probability that each claim ends up high-cost."""
        X = self.preprocessor.transform(claims).astype(np.float32)
        return calibrated(self.calibrator, self.scorer.raw(X, claims), self.scorer.kind)


def calibration_features(raw: np.ndarray, kind: str) -> np.ndarray:
    """Input of the logistic calibrator: log-odds of a probability, or log of a cost."""
    raw = np.asarray(raw, dtype=float)
    if kind == "cost_rank":
        return np.log1p(np.clip(raw, 0, None)).reshape(-1, 1)
    p = np.clip(raw, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def calibrated(calibrator: Any, raw: np.ndarray, kind: str) -> np.ndarray:
    """Apply a Platt-style calibrator (logistic regression on the transformed raw score)."""
    return calibrator.predict_proba(calibration_features(raw, kind))[:, 1]
