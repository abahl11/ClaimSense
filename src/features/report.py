"""Builds reports/features.md: feature catalogue, keyword-taxonomy coverage, model matrix.

Everything is computed on training claims only (accidents before the time-split cutoff): the
evidence behind the placeholder-estimate flag and every coverage figure.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.summaries import CLAIMS, MEDIAN, SHARE, group_summary, readable
from src.config import Config
from src.data.load import load_clean
from src.features.build import (
    build_imputer,
    build_preprocessor,
    feature_names,
    model_input_columns,
)
from src.features.tabular import engineer_features
from src.features.text_rules import TAXONOMY_FIELDS, coverage
from src.models.split import train_test_split_by_time
from src.reporting.outputs import (
    GENERATED_NOTE,
    figure_markdown,
    markdown_table,
    percent_columns,
    write_json,
    write_text,
)
from src.reporting.plots import hbar, new_figure, save_figure

logger = logging.getLogger(__name__)

# Plain-language definition and rationale for every model input (logical names).
# Values in braces are filled from the config when the report is written.
FEATURE_DOCS: dict[str, tuple[str, str]] = {
    "age": (
        "Claimant age at the accident.",
        "Recovery tends to take longer with age; EDA section 4 shows the pattern here.",
    ),
    "dependent_children": ("Number of dependent children.", "Dependants can raise benefits."),
    "dependents_other": ("Number of other dependants.", "As above; recorded separately."),
    "total_dependents": (
        "Children plus other dependants.",
        "One measure of family obligations; the raw counts are sparse.",
    ),
    "hours_per_week": (
        "Weekly hours (invalid values imputed).",
        "Exposure to risk and the base for wage replacement.",
    ),
    "days_per_week": ("Days worked per week (invalid values imputed).", "Shift pattern."),
    "hours_per_day": ("Hours per week / days per week.", "Long shifts are linked with fatigue."),
    "accident_year": (
        "Calendar year of the accident.",
        "Carries the cost trend. Trees cannot extrapolate beyond the last training year; "
        "Phase 5 tests explicit ways to handle drift.",
    ),
    "accident_hour": (
        "Hour of the accident (0-23).",
        "Time of day relates to shift type and fatigue.",
    ),
    "log_weekly_wages": (
        "log(1 + weekly wage).",
        "Wage replacement scales with pay; the log tames the long right tail.",
    ),
    "log_initial_estimate": (
        "log(1 + adjuster's initial estimate).",
        "The adjuster's view of severity, recorded at current prices (EDA sections 1 and 5).",
    ),
    "report_delay_days": (
        "Days from accident to report (log-transformed in the model).",
        "Reporting delay is a known cost driver; EDA section 3 shows the pattern here.",
    ),
    "hourly_wage": (
        "Weekly wage / hours, hours clipped up to {min_hours:g} (log-transformed).",
        "Pay rate independent of hours; clipping avoids dividing by near-zero hours.",
    ),
    "initial_to_wage_ratio": (
        "Initial estimate / weekly wage (log-transformed).",
        "Roughly how many weeks of pay the adjuster reserved, i.e. expected time off work.",
    ),
    "is_weekend": (
        "Accident on a Saturday or Sunday.",
        "Weekend work often means different staffing and tasks.",
    ),
    "is_night_shift": (
        "Accident after {night_after}:00 or before {night_before:02d}:00, "
        "excluding times of exactly 00:00.",
        "Night work is linked with fatigue. 00:00 is excluded because it mostly means "
        "'time unknown'.",
    ),
    "accident_time_unknown": (
        "Accident time recorded as exactly 00:00.",
        "A recording artifact concentrated in recent years (data quality report).",
    ),
    "is_part_time": (
        "Claimant is part-time.",
        "Part-time workers differ in hours, pay and job type.",
    ),
    "wage_is_default": (
        "Weekly wage equals a suspected system default ({heaped}).",
        "These values track recording eras, not pay (data quality report).",
    ),
    "initial_is_default": (
        "Initial estimate equals a placeholder value ({initial_defaults}).",
        "These estimates are less accurate than others (evidence in section 2).",
    ),
    "initial_is_nominal": (
        "Initial estimate of {nominal} or less.",
        "A nominal placeholder rather than an estimate.",
    ),
    "weekly_wages_was_missing": (
        "Weekly wage was missing or invalid and has been imputed.",
        "Lets the model treat imputed values differently.",
    ),
    "hours_per_week_was_missing": (
        "Hours were missing or invalid and have been imputed.",
        "As above.",
    ),
    "days_per_week_was_missing": ("Days were invalid and have been imputed.", "As above."),
    "gender": (
        "Recorded gender (M, F, U).",
        "Kept for now; Phase 7 compares the model with and without it.",
    ),
    "marital_status": (
        "M, S or U; unknown kept as its own category.",
        "Unknown can itself carry signal.",
    ),
    "age_band": (
        "Age in bands (edges in config).",
        "Lets linear models capture a non-linear age effect.",
    ),
    "accident_month": ("Month of the accident.", "Seasonal patterns, e.g. outdoor work."),
    "accident_weekday": ("Day of the week of the accident.", "Staffing and task mix vary by day."),
    "body_part": (
        "Keyword taxonomy: injured body part, 'multiple' if several areas.",
        "Cost varies widely by body part (EDA section 2).",
    ),
    "injury_nature": (
        "Keyword taxonomy: nature of injury, most severe mention wins.",
        "Cost varies widely by nature of injury (EDA section 2).",
    ),
    "injury_cause": (
        "Keyword taxonomy: how the injury happened.",
        "Mechanism relates to severity (falls, vehicles, lifting).",
    ),
    "num_body_parts_mentioned": (
        "Distinct body parts mentioned (after the finger/hand rule).",
        "More injured areas usually means more treatment.",
    ),
    "description_length": (
        "Characters in the description.",
        "Longer descriptions can signal more complex injuries.",
    ),
}


def small_pct(value: float) -> str:
    """Percentage with one decimal, showing tiny non-zero shares as '<0.1%'."""
    return "<0.1%" if 0 < value < 0.001 else f"{value:.1%}"


def fill_doc(text: str, cfg: Config) -> str:
    """Fill config values into a documentation string."""
    f, flags = cfg["features"], cfg["cleaning"]["flags"]
    return text.format(
        min_hours=f["min_hours_for_rate"],
        night_after=f["night_after_hour"],
        night_before=f["night_before_hour"],
        heaped=", ".join(f"{v:g}" for v in flags["heaped_wage_values"]),
        initial_defaults=", ".join(f"{v:g}" for v in f["initial_default_values"]),
        nominal=flags["nominal_initial_estimate_max"],
    )


# ---------------------------------------------------------------------------
# Feature catalogue
# ---------------------------------------------------------------------------


def catalogue(train_features: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """One row per model input: branch, definition, rationale and training statistics."""
    f = cfg["features"]
    branches = {
        "numeric": f["numeric"],
        "log numeric": f["log_numeric"],
        "binary": f["binary"],
        "categorical": f["categorical"],
        "text rules (L1)": f["text_rule_categorical"] + f["text_rule_numeric"],
    }
    imputed = build_imputer(cfg).fit_transform(train_features)  # adds the missing indicators
    rows = []
    for branch, names in branches.items():
        for name in names:
            column = cfg.resolve([name])[0]
            before = train_features[column] if column in train_features else imputed[column]
            definition, rationale = (fill_doc(text, cfg) for text in FEATURE_DOCS[name])
            rows.append(
                {
                    "Feature": name,
                    "Branch": branch,
                    "Definition": definition,
                    "Why": rationale,
                    "Missing before imputation": small_pct(before.isna().mean()),
                    "Share = 1 (flags)": (
                        small_pct(imputed[column].mean()) if branch == "binary" else ""
                    ),
                }
            )
    return pd.DataFrame(rows)


def catalogue_section(train_features: pd.DataFrame, cfg: Config) -> list[str]:
    """Markdown for the feature catalogue."""
    return [
        "## 1. Feature catalogue",
        "",
        "Every model input, grouped by how the pipeline preprocesses it. Statistics are on "
        "training claims before imputation. The target and anything derived from it are never "
        "inputs.",
        "",
        markdown_table(catalogue(train_features, cfg)),
        "",
    ]


# ---------------------------------------------------------------------------
# Placeholder initial estimates
# ---------------------------------------------------------------------------


def estimate_flag_evidence(train: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Accuracy of initial estimates for candidate 'placeholder' definitions."""
    initial, target = train[cfg.col("initial_estimate")], train[cfg.col("target")]
    log_error = (np.log(target) - np.log(initial)).abs()
    defaults = cfg["features"]["initial_default_values"]
    candidates = {
        "Multiple of 500": initial % 500 == 0,
        "Multiple of 1,000": initial % 1000 == 0,
        f"Equal to {', '.join(f'{v:g}' for v in defaults)} (chosen)": initial.isin(defaults),
    }
    rows = [
        {
            "Definition": name,
            "Share of claims": mask.mean(),
            "Median abs log error, flagged": log_error[mask].median(),
            "Median abs log error, others": log_error[~mask].median(),
        }
        for name, mask in candidates.items()
    ]
    return pd.DataFrame(rows)


def estimate_flag_section(train: pd.DataFrame, cfg: Config) -> list[str]:
    """Why only specific initial-estimate values are flagged as placeholders."""
    table = estimate_flag_evidence(train, cfg)
    return [
        "## 2. Which initial estimates behave like placeholders?",
        "",
        "Adjusters often enter round numbers (EDA section 1), so a natural idea is to flag "
        "round estimates as unreliable. The table tests that on training claims: a flag is only "
        "useful if flagged estimates are further from the final cost (higher error).",
        "",
        markdown_table(percent_columns(table, ["Share of claims"]), floatfmt=".3f"),
        "",
        "Round numbers in general are *not* less accurate, so a 'round number' flag would add "
        "noise. Only the most common single value is clearly worse than the rest, so the "
        "`initial_is_default` flag covers exactly the configured value(s).",
        "",
    ]


# ---------------------------------------------------------------------------
# Keyword taxonomy
# ---------------------------------------------------------------------------


def taxonomy_section(
    train_features: pd.DataFrame, cfg: Config
) -> tuple[list[str], dict[str, float]]:
    """Coverage, category counts and unmatched examples for each taxonomy field."""
    rules = cfg["text_rules"]
    other = rules["other_label"]
    covered = coverage(train_features, other)
    coverage_table = pd.DataFrame(
        {"Field": list(covered.index), "Labelled (not 'other')": [f"{v:.1%}" for v in covered]}
    )
    blocks = [
        "## 3. Keyword taxonomy",
        "",
        "Rules in `src/features/text_rules.py`; keyword lists in `config/config.yaml`. Keywords "
        "match whole words. For nature and cause the categories are in priority order, so a "
        "description mentioning both a laceration and a fracture is labelled a fracture. A "
        "finger mentioned together with its hand counts as a finger, so *multiple* means "
        "separate areas of the body.",
        "",
        "The dictionaries started from the categories in the project brief and were extended by "
        "reading unmatched descriptions: foreign body, hernia, inflammation, infection, soft "
        "tissue, psychological, bite/sting, dermatitis, amputation and pain were added to "
        "nature; vehicle, assault, bending/twisting, tool use, exposure and animal/insect to "
        "cause; trunk to body part. Descriptions that name an injury but not how it happened "
        "(for example 'STRAIN LOWER BACK') legitimately stay *other* for cause.",
        "",
        "**Coverage on training claims**",
        "",
        figure_markdown(
            plot_coverage(covered, cfg),
            cfg.path("features_report"),
            "Keyword taxonomy coverage",
            coverage_takeaway(covered, other),
        ),
        markdown_table(coverage_table),
        "",
    ]
    text = train_features[cfg.col("description")]
    target = cfg.col("target")
    for field in TAXONOMY_FIELDS:
        summary = group_summary(train_features, field, target, np.inf)
        summary = summary.sort_values(CLAIMS, ascending=False)[[field, CLAIMS, SHARE, MEDIAN]]
        summary[field] = readable(summary[field])
        unmatched = text[train_features[field] == other]
        examples = unmatched.sample(
            min(rules["unmatched_examples"], len(unmatched)), random_state=cfg.random_seed
        )
        title = readable(pd.Series([field]))[0].capitalize()
        blocks += [
            f"### {title}",
            "",
            markdown_table(percent_columns(summary, [SHARE]), floatfmt=",.0f"),
            "",
            f"Sample of descriptions still labelled *{other}*:",
            "",
            *[f"- `{example}`" for example in examples],
            "",
        ]
    return blocks, {f"coverage_{k}": float(v) for k, v in covered.items()}


def plot_coverage(covered: pd.Series, cfg: Config) -> Path:
    """Share of training descriptions that get a label, for each taxonomy field."""
    fig, ax = new_figure(cfg)
    ordered = covered.sort_values()
    hbar(ax, ordered, readable(pd.Series(ordered.index)), "{:.1%}", cfg)
    ax.set_title("Keyword taxonomy: share of training descriptions labelled (not 'other')")
    return save_figure(fig, cfg, "features_taxonomy_coverage")


def coverage_takeaway(covered: pd.Series, other: str) -> str:
    """Name the best- and worst-covered fields."""
    best, worst = covered.idxmax(), covered.idxmin()
    text = (
        f"The rules label {covered[best]:.0%} of descriptions for {best.replace('_', ' ')} "
        f"but only {covered[worst]:.0%} for {worst.replace('_', ' ')}; the rest stay "
        f"*{other}*."
    )
    if worst == "injury_cause":
        text += (
            " Cause is the hardest field because many descriptions name the injury but not how "
            "it happened, which no keyword list can fix."
        )
    return text


# ---------------------------------------------------------------------------
# Output matrix and safeguards
# ---------------------------------------------------------------------------


def matrix_section(train: pd.DataFrame, cfg: Config) -> tuple[list[str], dict[str, Any]]:
    """Fit the full preprocessor on training rows and count output columns per branch."""
    preprocessor = build_preprocessor(cfg).fit(train)
    names = feature_names(preprocessor)
    counts = pd.Series([n.split("__")[0] for n in names]).value_counts()
    table = counts.rename_axis("Branch").reset_index(name="Columns")
    target = cfg.col("target")
    inputs = model_input_columns(cfg)
    if target in inputs or any(target in name for name in names):
        raise AssertionError("The target leaked into the model inputs")
    return [
        "## 4. Model matrix",
        "",
        f"With the keyword-rule features switched on, the fitted pipeline turns each claim into "
        f"**{len(names)} columns**:",
        "",
        markdown_table(table),
        "",
        "The `text_layers.rules` switch removes the keyword-rule columns for the Phase 5 "
        f"ablation. The target (`{target}`) is not among the {len(inputs)} input columns the "
        "pipeline reads, a check that also runs in the test suite.",
        "",
    ], {"n_model_columns": len(names), "columns_per_branch": counts.to_dict()}


def safeguards_blocks() -> list[str]:
    """How this design avoids leakage."""
    return [
        "## 5. Leakage safeguards",
        "",
        "- **Nothing learned outside the pipeline.** Group medians, scalers and one-hot "
        "categories are all steps of one scikit-learn Pipeline. They are fit on training rows "
        "only, and cross-validation refits them in every fold.",
        "- **Fixed rules only before the split.** Cleaning, feature formulas and the keyword "
        "taxonomy use thresholds fixed in the config and learn nothing from the data.",
        "- **Design decisions on training claims only.** The placeholder-estimate value and the "
        "keyword-list iterations both used training claims.",
        "- **No target in the inputs.** The target and anything derived from it (such as the "
        "ultimate/initial ratio) are never model inputs.",
        "",
    ]


def run_features(cfg: Config) -> None:
    """Write ``reports/features.md`` and ``reports/metrics/features.json``."""
    claims = load_clean(cfg)
    train, _, cutoff = train_test_split_by_time(claims, cfg)
    train_features = engineer_features(train, cfg)
    taxonomy_blocks, taxonomy_metrics = taxonomy_section(train_features, cfg)
    matrix_blocks, matrix_metrics = matrix_section(train, cfg)
    blocks = [
        "# Features",
        "",
        GENERATED_NOTE.format(stage="features"),
        "",
        f"All statistics use the {len(train):,} training claims (accidents before "
        f"{cutoff:%d %b %Y}).",
        "",
        *catalogue_section(train_features, cfg),
        *estimate_flag_section(train, cfg),
        *taxonomy_blocks,
        *matrix_blocks,
        *safeguards_blocks(),
    ]
    write_text("\n".join(blocks), cfg.path("features_report"))
    write_json(
        {**taxonomy_metrics, **matrix_metrics},
        cfg.path("features_metrics"),
    )
