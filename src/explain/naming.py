"""Readable names for preprocessor output columns, shared by the SHAP report and the app.

Each matrix column gets a label ("Injury nature = fracture"), a group (its business concept,
"Injury nature") and the engineered source column it came from, so SHAP values can be shown
per column or summed per concept.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Config
from src.features.text_embed import name_clusters
from src.models.artifacts import CostModel

TFIDF_GROUP = "Description wording (TF-IDF/SVD)"
CLUSTER_GROUP = "Description meaning (embedding cluster)"
NUMERIC_BRANCHES = ("numeric", "log", "rules_numeric")
ONE_HOT_BRANCHES = ("categorical", "rules_categorical")
TEXT_SOURCES = (
    "body_part",
    "injury_nature",
    "injury_cause",
    "num_body_parts_mentioned",
    "description_length",
)
DISPLAY = {
    "age": "Age",
    "dependent_children": "Dependent children",
    "dependents_other": "Other dependants",
    "total_dependents": "Total dependants",
    "hours_per_week": "Hours per week",
    "days_per_week": "Days per week",
    "hours_per_day": "Hours per day",
    "accident_year": "Accident year",
    "accident_hour": "Accident hour",
    "log_weekly_wages": "Weekly wage",
    "log_initial_estimate": "Initial estimate",
    "report_delay_days": "Reporting delay",
    "hourly_wage": "Hourly wage",
    "initial_to_wage_ratio": "Initial estimate / weekly wage",
    "is_weekend": "Weekend accident",
    "is_night_shift": "Night-shift accident",
    "accident_time_unknown": "Accident time unknown (00:00)",
    "is_part_time": "Part-time worker",
    "wage_is_default": "Wage is a system default",
    "initial_is_default": "Initial estimate is a placeholder",
    "initial_is_nominal": "Nominal initial estimate",
    "weekly_wages_was_missing": "Wage was missing",
    "hours_per_week_was_missing": "Hours were missing",
    "days_per_week_was_missing": "Days were missing",
    "gender": "Gender",
    "marital_status": "Marital status",
    "age_band": "Age band",
    "accident_month": "Accident month",
    "accident_weekday": "Accident weekday",
    "body_part": "Body part",
    "injury_nature": "Injury nature",
    "injury_cause": "Injury cause",
    "num_body_parts_mentioned": "Body parts mentioned",
    "description_length": "Description length",
}
TEXT_GROUPS = {TFIDF_GROUP, CLUSTER_GROUP} | {DISPLAY[s] for s in TEXT_SOURCES}


def logical_names(cfg: Config) -> dict[str, str]:
    """Real column name -> logical name for every configured model input."""
    f = cfg["features"]
    names = f["numeric"] + f["log_numeric"] + f["binary"] + f["categorical"]
    names += f["text_rule_categorical"] + f["text_rule_numeric"]
    return dict(zip(cfg.resolve(names), names, strict=True))


def describe_column(column: str, cfg: Config, clusters: dict[int, str]) -> tuple[str, str, str]:
    """(label, group, engineered source column) for one preprocessor output column."""
    branch, feature = column.split("__", 1)
    logical = logical_names(cfg)
    if branch == "tfidf_svd":
        return f"TF-IDF component {feature.removeprefix('truncatedsvd')}", TFIDF_GROUP, ""
    if branch == "embedding_clusters":
        cluster = int(feature.rsplit("_", 1)[1])
        return f"Cluster: {clusters.get(cluster, cluster)}", CLUSTER_GROUP, ""
    if branch in ONE_HOT_BRANCHES:
        source = max((s for s in logical if feature.startswith(f"{s}_")), key=len)
        category = feature[len(source) + 1 :].replace("infrequent_sklearn", "rare categories")
        name = DISPLAY[logical[source]]
        return f"{name} = {category.replace('_', ' ')}", name, source
    name = DISPLAY[logical[feature]]
    return name, name, feature


def cluster_names_of(model: CostModel, cfg: Config) -> dict[int, str]:
    """Names of the embedding clusters in a fitted preprocessor (empty if not used)."""
    columns = model.preprocessor.named_steps["columns"]
    if "embedding_clusters" not in columns.named_transformers_:
        return {}
    kmeans = columns.named_transformers_["embedding_clusters"].named_steps["clusters"].kmeans_
    return name_clusters(kmeans, cfg)


def describe_model_columns(model: CostModel, cfg: Config) -> list[tuple[str, str, str]]:
    """Label, group and source column for every column the cost model uses."""
    clusters = cluster_names_of(model, cfg)
    return [describe_column(column, cfg, clusters) for column in model.feature_names]


def group_contributions(values: np.ndarray, groups: list[str]) -> pd.DataFrame:
    """SHAP values summed within each group: one column per business concept."""
    frame = pd.DataFrame(np.atleast_2d(values))
    return frame.T.groupby(np.array(groups)).sum().T
