"""Phase 7: SHAP explanations of the final cost model plus the fairness checks.

SHAP (TreeExplainer) splits each prediction into additive contributions, one per input
column, on the model's output scale. The model predicts a log-type target, so a
contribution of +0.69 means "this input roughly doubles the predicted cost" (exp(0.69) = 2),
holding the other inputs fixed. One-hot columns are summed back into their source feature
(e.g. every ``injury_nature`` column) so importance is reported per business concept.
SHAP describes what the model relies on; it is not a causal statement.

Writes ``reports/explainability.md``, its figures, ``reports/metrics/explainability.json``
and ``reports/metrics/fairness.json``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor

from src.config import Config
from src.data.load import load_clean
from src.evaluation.error_analysis import band_sort_key
from src.evaluation.fairness import flag_frame, gender_comparison, group_fairness
from src.explain.naming import (
    DISPLAY,
    NUMERIC_BRANCHES,
    TEXT_GROUPS,
    describe_model_columns,
    group_contributions,
)
from src.models.artifacts import CostModel, HighCostClassifier
from src.models.datasets import get_datasets
from src.models.predict import load_models
from src.models.split import train_test_split_by_time
from src.models.targets import APPROACHES
from src.models.train_classifier import final_candidate
from src.models.train_regression import Evaluator
from src.reporting.build_results import money, pct
from src.reporting.outputs import (
    GENERATED_NOTE,
    figure_markdown,
    markdown_table,
    write_json,
    write_text,
)
from src.reporting.plots import apply_style, hbar, new_figure, save_figure, series_color

logger = logging.getLogger(__name__)


@dataclass
class Explained:
    """SHAP values for a sample of test claims, with everything needed to describe them."""

    values: np.ndarray
    base_value: float
    labels: list[str]
    groups: list[str]
    branches: list[str]
    columns: list[str]
    frame: pd.DataFrame
    engineered: pd.DataFrame
    predictions: pd.DataFrame


# ---------------------------------------------------------------------------
# SHAP values
# ---------------------------------------------------------------------------


def engineered_inputs(model: CostModel, frame: pd.DataFrame) -> pd.DataFrame:
    """Engineered and imputed (but unscaled) inputs, for readable plots and tables."""
    steps = model.preprocessor.named_steps
    return steps["impute"].transform(steps["engineer"].transform(frame)).reset_index(drop=True)


def explain(model: CostModel, test: pd.DataFrame, cfg: Config) -> Explained:
    """SHAP values for a seeded random sample of test claims."""
    e = cfg["explain"]
    rng = np.random.default_rng(cfg.random_seed)
    forest = isinstance(model.regressor, RandomForestRegressor)
    size = min(e["shap_sample_forest"] if forest else e["shap_sample"], len(test))
    sample = test.iloc[np.sort(rng.choice(len(test), size=size, replace=False))].reset_index(
        drop=True
    )
    X = model.matrix(sample)
    explainer = shap.TreeExplainer(model.regressor)
    values = np.asarray(explainer.shap_values(X))
    described = describe_model_columns(model, cfg)
    return Explained(
        values=values,
        base_value=float(np.ravel(explainer.expected_value)[0]),
        labels=[d[0] for d in described],
        groups=[d[1] for d in described],
        branches=[c.split("__", 1)[0] for c in model.feature_names],
        columns=[d[2] for d in described],
        frame=sample,
        engineered=engineered_inputs(model, sample),
        predictions=model.predict_frame(sample, X).reset_index(drop=True),
    )


def grouped_values(ex: Explained) -> pd.DataFrame:
    """SHAP values summed within each source feature (one column per group)."""
    return group_contributions(ex.values, ex.groups)


def grouped_importance(grouped: pd.DataFrame) -> pd.Series:
    """Mean absolute grouped SHAP value per source feature, largest first."""
    return grouped.abs().mean().sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_importance(importance: pd.Series, cfg: Config) -> Path:
    """Horizontal bars of grouped importance."""
    top = importance.head(cfg["explain"]["top_features"])
    fig, ax = new_figure(cfg)
    hbar(ax, top, pd.Series(top.index), "{:.3f}", cfg)
    ax.invert_yaxis()
    ax.set_title("Mean |SHAP value| per feature (one-hot columns summed), test sample")
    return save_figure(fig, cfg, "shap_importance")


def plot_beeswarm(ex: Explained, cfg: Config) -> Path:
    """SHAP's beeswarm summary of the individual columns."""
    apply_style(cfg)
    explanation = shap.Explanation(
        values=ex.values, base_values=ex.base_value, feature_names=ex.labels
    )
    shap.plots.beeswarm(explanation, max_display=cfg["explain"]["top_features"], show=False)
    fig = plt.gcf()
    fig.set_size_inches(cfg["plots"]["width"], cfg["plots"]["height"] * 1.4)
    fig.suptitle(
        "SHAP summary: each dot is one test claim (colour = input value)", x=0.02, ha="left"
    )
    return save_figure(fig, cfg, "shap_beeswarm")


def dependence_features(ex: Explained, importance: pd.Series, cfg: Config) -> list[int]:
    """Column positions of the most important single numeric features."""
    positions = []
    for group in importance.index:
        matches = [
            i
            for i, (g, b) in enumerate(zip(ex.groups, ex.branches, strict=True))
            if g == group and b in NUMERIC_BRANCHES
        ]
        if len(matches) == 1:
            positions.append(matches[0])
        if len(positions) == cfg["explain"]["dependence_features"]:
            break
    return positions


def plot_dependence(ex: Explained, positions: list[int], cfg: Config) -> Path:
    """Small multiples: SHAP value against the (unscaled) input, with a binned mean line."""
    p = cfg["plots"]
    fig, axes = new_figure(cfg, ncols=2, nrows=2)
    for ax, position in zip(axes.flat, positions, strict=False):
        x = natural_values(ex, position)
        y = ex.values[:, position]
        ax.scatter(x, y, s=6, color=p["de_emphasis"], alpha=0.5, linewidths=0)
        bins = pd.qcut(pd.Series(x).rank(method="first"), 20, labels=False)
        means = pd.DataFrame({"x": x, "y": y, "bin": bins}).groupby("bin").mean()
        ax.plot(means["x"], means["y"], color=series_color(cfg, 0))
        ax.axhline(0, color=p["baseline"], linewidth=1)
        if is_logged(ex, position):
            ax.set_xscale("symlog", linthresh=1)
        ax.set_title(ex.labels[position])
        ax.set_ylabel("SHAP value")
    for ax in list(axes.flat)[len(positions) :]:
        ax.set_visible(False)
    fig.suptitle(
        "How the top numeric inputs move predictions (line = average; +0.69 = x2)",
        x=0.02,
        ha="left",
    )
    return save_figure(fig, cfg, "shap_dependence")


def example_rows(ex: Explained, cfg: Config) -> dict[str, int]:
    """Sample rows whose expected cost sits at the configured low / medium / high quantiles."""
    expected = ex.predictions["expected_cost"]
    labels = ("low", "medium", "high")
    rows = {}
    for label, q in zip(labels, cfg["explain"]["example_quantiles"], strict=True):
        rows[label] = int((expected - expected.quantile(q)).abs().idxmin())
    return rows


def display_value(ex: Explained, row: int, position: int) -> float:
    """Input value shown next to a feature in a waterfall (unscaled where possible)."""
    if ex.branches[position] in NUMERIC_BRANCHES or ex.branches[position] == "binary":
        return float(natural_values(ex, position)[row])
    return float(ex.values[row, position] != 0)


def plot_waterfall(ex: Explained, row: int, label: str, cfg: Config) -> Path:
    """SHAP waterfall for one claim."""
    apply_style(cfg)
    data = np.array([display_value(ex, row, i) for i in range(ex.values.shape[1])])
    explanation = shap.Explanation(
        values=ex.values[row], base_values=ex.base_value, data=data, feature_names=ex.labels
    )
    shap.plots.waterfall(explanation, max_display=cfg["explain"]["waterfall_features"], show=False)
    fig = plt.gcf()
    fig.set_size_inches(cfg["plots"]["width"], cfg["plots"]["height"] * 1.2)
    cost = ex.predictions.loc[row, "expected_cost"]
    fig.suptitle(f"Example {label}-cost claim: expected cost {money(cost)}", x=0.02, ha="left")
    return save_figure(fig, cfg, f"shap_waterfall_{label}")


# ---------------------------------------------------------------------------
# Insights (templated from SHAP statistics)
# ---------------------------------------------------------------------------


def natural_values(ex: Explained, position: int) -> np.ndarray:
    """An input's values in natural units (log-transformed features are converted back)."""
    column = ex.columns[position]
    values = ex.engineered[column].to_numpy(dtype=float)
    return np.expm1(values) if column.startswith("log_") else values


def is_logged(ex: Explained, position: int) -> bool:
    """Whether an input is skewed enough to plot on a log axis."""
    return ex.branches[position] == "log" or ex.columns[position].startswith("log_")


def factor(log_change: float) -> str:
    """A log-scale change as a multiplicative factor, e.g. 0.69 -> 'x2.00'."""
    return f"x{np.exp(log_change):.2f}"


def numeric_effect(ex: Explained, position: int) -> dict[str, float]:
    """Mean SHAP value in the bottom and top tenth of an input, and their difference."""
    x = natural_values(ex, position)
    y = ex.values[:, position]
    low, high = np.quantile(x, [0.1, 0.9])
    bottom, top = y[x <= low].mean(), y[x >= high].mean()
    return {
        "low_value": float(low),
        "high_value": float(high),
        "bottom": float(bottom),
        "top": float(top),
        "difference": float(top - bottom),
    }


def category_effects(ex: Explained, grouped: pd.DataFrame, group: str, min_count: int) -> pd.Series:
    """Mean grouped SHAP value per category of a one-hot source feature."""
    source = next(c for g, c in zip(ex.groups, ex.columns, strict=True) if g == group)
    categories = ex.engineered[source].astype(str)
    effects = grouped[group].groupby(categories.to_numpy()).agg(["mean", "size"])
    return effects.loc[effects["size"] >= min_count, "mean"].sort_values()


def insights(
    ex: Explained, grouped: pd.DataFrame, importance: pd.Series, target: str, cfg: Config
) -> list[str]:
    """Four to six plain-English findings, each computed from the SHAP values."""
    total = importance.sum()
    top = importance.head(3)
    lines = [
        "**What drives predictions most.** "
        + ", ".join(
            f"{name} ({value / total:.0%} of total attribution)" for name, value in top.items()
        )
        + " are the three strongest drivers. Together the top five account for "
        f"{importance.head(5).sum() / total:.0%}."
    ]
    text_groups = [g for g in importance.index if g in TEXT_GROUPS]
    text_share = importance[text_groups].sum() / total
    weight = "a large part" if text_share >= 0.25 else "a smaller but real part"
    lines.append(
        "**The claim description matters.** Features derived from the free text ("
        + ", ".join(text_groups[:4])
        + f") carry {text_share:.0%} of all attribution: {weight} of what the model knows "
        "comes from the description alone."
    )
    positions = {
        ex.labels[i]: i for i in range(len(ex.labels)) if ex.branches[i] in NUMERIC_BRANCHES
    }
    if "Initial estimate" in positions:
        effect = numeric_effect(ex, positions["Initial estimate"])
        ratio_note = (
            " Because the model predicts the ratio of ultimate cost to the estimate, this is on "
            "top of the estimate itself: a negative effect means large estimates tend to be "
            "overshoots, and small ones tend to be exceeded."
            if "ratio" in target
            else ""
        )
        lines.append(
            f"**The adjuster's estimate is informative but not taken at face value.** Moving from "
            f"the lowest tenth of initial estimates (<= {money(effect['low_value'])}) to the "
            "highest "
            f"(>= {money(effect['high_value'])}) changes the model output by "
            f"{factor(effect['difference'])} on average." + ratio_note
        )
    if "Reporting delay" in positions:
        effect = numeric_effect(ex, positions["Reporting delay"])
        lines.append(
            f"**Late reporting raises predicted cost.** Claims in the slowest-reported tenth "
            f"(>= {effect['high_value']:.0f} days) get {factor(effect['difference'])} the "
            f"attribution of the fastest (<= {effect['low_value']:.0f} days), matching the EDA."
            if effect["difference"] > 0
            else f"**Reporting delay has little systematic effect** in the model "
            f"({factor(effect['difference'])} from fastest to slowest tenth)."
        )
    natures = category_effects(
        ex, grouped, DISPLAY["injury_nature"], cfg["fairness"]["min_group_size"]
    )
    if len(natures) >= 2:
        high, low = natures.index[-1], natures.index[0]
        lines.append(
            f"**Injury type shifts costs strongly.** Relative to the average claim, the injury "
            f"nature alone moves predictions by {factor(natures.iloc[-1])} for "
            f"{high.replace('_', ' ')} claims and {factor(natures.iloc[0])} for "
            f"{low.replace('_', ' ')} claims."
        )
    if "Age" in positions:
        effect = numeric_effect(ex, positions["Age"])
        direction = "older" if effect["difference"] > 0 else "younger"
        lines.append(
            f"**Age matters, but less than the injury itself.** The oldest tenth of claimants "
            f"(>= {effect['high_value']:.0f}) gets {factor(effect['difference'])} the "
            "attribution of "
            f"the youngest (<= {effect['low_value']:.0f}): {direction} workers' claims cost more, "
            "consistent with longer recovery."
            if effect["difference"] > 0
            else "**Age has a small, negative effect** "
            f"({factor(effect['difference'])} from youngest "
            f"to oldest tenth)."
        )
    return lines[:6]


def example_table(ex: Explained, grouped: pd.DataFrame, rows: dict[str, int], cfg: Config) -> str:
    """Markdown table describing the three waterfall example claims."""
    records = []
    for label, row in rows.items():
        drivers = (
            grouped.loc[row]
            .reindex(grouped.loc[row].abs().sort_values(ascending=False).index)
            .head(3)
        )
        prediction = ex.predictions.loc[row]
        records.append(
            {
                "Example": label,
                "Description": ex.frame.loc[row, cfg.col("description")],
                "Actual cost": money(ex.frame.loc[row, cfg.col("target")]),
                "Expected cost": money(prediction["expected_cost"]),
                "80% interval": (
                    f"{money(prediction['interval_low'])} - {money(prediction['interval_high'])}"
                ),
                "Top drivers": "; ".join(
                    f"{name} {factor(value)}" for name, value in drivers.items()
                ),
            }
        )
    return markdown_table(pd.DataFrame(records))


# ---------------------------------------------------------------------------
# Fairness section
# ---------------------------------------------------------------------------


def fairness_tables(
    cfg: Config, test: pd.DataFrame, model: CostModel, classifier: HighCostClassifier
) -> dict[str, pd.DataFrame]:
    """Per-gender and per-age-band error and flag metrics on the whole test set."""
    X = model.matrix(test)
    predictions = model.predict_frame(test, X)
    probabilities = classifier.probabilities(test)
    frame = flag_frame(
        test,
        predictions["typical_cost"].to_numpy(),
        predictions["expected_cost"].to_numpy(),
        probabilities,
        classifier.threshold,
        classifier.cost_threshold,
        cfg,
    )
    size = cfg["fairness"]["min_group_size"]
    ages = group_fairness(frame, "age_band", size)
    order = sorted(range(len(ages)), key=lambda i: band_sort_key(ages["age_band"].iloc[i]))
    return {"gender": group_fairness(frame, "gender", size), "age_band": ages.iloc[order]}


def fairness_markdown(table: pd.DataFrame, by: str, label: str) -> str:
    """Readable per-group fairness table."""
    return markdown_table(
        pd.DataFrame(
            {
                label: table[by],
                "Claims": table["claims"].map(lambda n: f"{n:,}"),
                "Mean actual": table["mean_actual"].map(money),
                "Predicted / actual": table["bias_ratio"].map(lambda x: f"{x:.2f}"),
                "RMSLE": table["rmsle"].map(lambda x: f"{x:.3f}"),
                "Log bias": table["mean_log_error"].map(lambda x: f"{x:+.3f}"),
                "True high-cost rate": table["high_cost_rate"].map(pct),
                "Flag rate (threshold)": table["flag_rate"].map(pct),
                "Flagged at 10% capacity": table["capacity_flag_rate"].map(pct),
                "Recall at capacity": table["recall"].map(pct),
            }
        )
    )


def worst_gender_bias(variant: dict[str, Any]) -> float:
    """Largest distance from 1.0 of predicted / actual cost across genders."""
    return max(abs(values["bias_ratio"] - 1) for values in variant["by_gender"].values())


def gender_bias_text(comparison: dict[str, Any]) -> str:
    """Whether removing gender makes the per-gender bias worse."""
    with_gender = worst_gender_bias(comparison["with_gender"])
    without = worst_gender_bias(comparison["without_gender"])
    change = "does not worsen" if without <= with_gender + 0.01 else "worsens"
    return (
        f"Removing it {change} the per-gender bias (largest gap between predicted and actual "
        f"cost for either gender: {with_gender:.1%} with it, {without:.1%} without)."
    )


def recommendation(comparison: dict[str, Any], cfg: Config) -> str:
    """Neutral, templated recommendation on keeping gender as a feature."""
    change = comparison["cv_rmsle_change_without_gender"]
    shipped = "excludes" if cfg["fairness"]["exclude_gender"] else "includes"
    if comparison["negligible"]:
        advice = (
            f"**Recommendation: exclude Gender.** Removing it changes cross-validated RMSLE by "
            f"{change:+.4f}, below the {cfg['fairness']['negligible_rmsle_change']} threshold set "
            "in advance for a meaningful difference. A protected characteristic that adds no "
            "measurable accuracy only adds legal and reputational risk. "
            + gender_bias_text(comparison)
        )
    elif change > 0:
        advice = (
            f"**Gender carries some signal**: removing it worsens cross-validated RMSLE by "
            f"{change:+.4f}. Whether that justifies using a protected characteristic is a policy "
            "and legal decision; the recommendation is to exclude it from any decision about an "
            "individual claim and to monitor group-level errors instead."
        )
    else:
        advice = (
            f"**Recommendation: exclude Gender.** The model is slightly *more* accurate without "
            f"it (cross-validated RMSLE {change:+.4f})."
        )
    return (
        advice + f" The shipped model {shipped} Gender (`fairness.exclude_gender` in the config)."
    )


def fairness_blocks(
    tables: dict[str, pd.DataFrame], comparison: dict[str, Any], cfg: Config
) -> list[str]:
    """Markdown for the fairness section."""
    rows = []
    for label in ("with_gender", "without_gender"):
        variant = comparison[label]
        row = {
            "Model": label.replace("_", " "),
            "CV RMSLE": f"{variant['cv']['rmsle']:.4f}",
            "Test RMSLE": f"{variant['test']['rmsle']:.4f}",
            "Test predicted / actual": f"{variant['test']['bias_ratio']:.3f}",
        }
        for gender, values in variant["by_gender"].items():
            row[f"Predicted / actual, {gender}"] = f"{values['bias_ratio']:.3f}"
        rows.append(row)
    gender = tables["gender"]
    spread = gender["rmsle"].max() - gender["rmsle"].min() if len(gender) > 1 else 0.0
    return [
        "## Fairness",
        "",
        "Test-set accuracy and flag rates by group. *Flag rate (threshold)* uses the fixed "
        "probability threshold; *Flagged at 10% capacity* is the share of the group among the "
        "10% of claims ranked highest. Differences in flag rates are expected where true "
        "high-cost rates differ; what matters is whether errors are systematically worse for "
        f"a group. Groups with fewer than {cfg['fairness']['min_group_size']} test claims are "
        "not shown (for gender, the handful recorded as unknown).",
        "",
        "### By gender",
        "",
        fairness_markdown(gender, "gender", "Gender"),
        "",
        f"The log-scale error differs by {spread:.3f} RMSLE between genders.",
        "",
        "### By age band",
        "",
        fairness_markdown(tables["age_band"], "age_band", "Age band"),
        "",
        "Age is a legitimate predictor of recovery time, but it is also a protected "
        "characteristic in employment law, so these age-band results should be monitored in "
        "production rather than read as a verdict.",
        "",
        "### The model with and without gender",
        "",
        markdown_table(pd.DataFrame(rows)),
        "",
        recommendation(comparison, cfg),
        "",
    ]


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------


def run_explain(cfg: Config) -> None:
    """Stage ``explain``: SHAP report, business insights and fairness checks."""
    models = load_models(cfg)
    model, classifier = models.cost, models.classifier
    _, test, _ = train_test_split_by_time(load_clean(cfg), cfg)
    test = test.reset_index(drop=True)
    ex = explain(model, test, cfg)
    grouped = grouped_values(ex)
    importance = grouped_importance(grouped)
    report = cfg.path("explainability_report")
    rows = example_rows(ex, cfg)
    positions = dependence_features(ex, importance, cfg)
    target = model.summary.get("target", "")
    figures = {
        "importance": plot_importance(importance, cfg),
        "beeswarm": plot_beeswarm(ex, cfg),
        "dependence": plot_dependence(ex, positions, cfg),
        **{
            f"waterfall_{label}": plot_waterfall(ex, row, label, cfg) for label, row in rows.items()
        },
    }
    findings = insights(ex, grouped, importance, target, cfg)
    tables = fairness_tables(cfg, test, model, classifier)
    comparison = gender_comparison(final_candidate(cfg), Evaluator(get_datasets(cfg), cfg), cfg)
    blocks = [
        "# Explainability and fairness",
        "",
        GENERATED_NOTE.format(stage="explain"),
        "",
        f"The final cost model ({model.summary.get('name', '')}) predicts "
        f"**{APPROACHES.get(target, target)}**. SHAP values are on that log scale: +0.69 means "
        "an input roughly doubles the predicted cost (x2), -0.69 halves it, holding everything "
        f"else fixed. Explained: a seeded random sample of {len(ex.frame):,} test claims. SHAP "
        "describes what the model relies on, not what causes costs.",
        "",
        "## Business insights",
        "",
        *[f"{i}. {line}" for i, line in enumerate(findings, 1)],
        "",
        "## Global importance",
        "",
        figure_markdown(
            figures["importance"],
            report,
            "Grouped SHAP importance",
            importance_takeaway(importance),
        ),
        figure_markdown(
            figures["beeswarm"],
            report,
            "SHAP beeswarm",
            "Each dot is one claim; position shows how much that input moved its prediction "
            "and colour shows the input's value (red high, blue low), so the direction of each "
            "effect can be read off.",
        ),
        "## How the top numeric inputs act",
        "",
        figure_markdown(
            figures["dependence"],
            report,
            "SHAP dependence",
            "Grey dots are claims, the blue line is the average SHAP value along the input. "
            "Where the line is above zero the input pushes predictions up.",
        ),
        "## Three example claims",
        "",
        example_table(ex, grouped, rows, cfg),
        "",
        *[
            figure_markdown(
                figures[f"waterfall_{label}"],
                report,
                f"Waterfall: {label}",
                waterfall_takeaway(ex, grouped, row, label),
            )
            for label, row in rows.items()
        ],
        *fairness_blocks(tables, comparison, cfg),
    ]
    write_text("\n".join(blocks), report)
    write_json(
        {
            "importance": importance.to_dict(),
            "insights": findings,
            "examples": rows,
            "base_value": ex.base_value,
        },
        cfg.path("explainability_metrics"),
    )
    write_json(
        {name: table.to_dict(orient="records") for name, table in tables.items()}
        | {"gender_comparison": comparison},
        cfg.path("fairness_metrics"),
    )


def importance_takeaway(importance: pd.Series) -> str:
    """Two sentences on the grouped importance figure."""
    total = importance.sum()
    first, second = importance.index[:2]
    return (
        f"{first} is the single most important input ({importance.iloc[0] / total:.0%} of total "
        f"attribution), followed by {second} ({importance.iloc[1] / total:.0%}). One-hot columns "
        "are summed into their source feature so each bar is one business concept."
    )


def waterfall_takeaway(ex: Explained, grouped: pd.DataFrame, row: int, label: str) -> str:
    """One or two sentences describing a waterfall."""
    contributions = grouped.loc[row]
    up = contributions.idxmax()
    down = contributions.idxmin()
    return (
        f"Starting from the average model output ({ex.base_value:.2f}), the largest push up for "
        f"this {label}-cost claim comes from {up} ({factor(contributions[up])}) and the largest "
        f"push down from {down} ({factor(contributions[down])}). Values are on the log scale of "
        "the model target."
    )
