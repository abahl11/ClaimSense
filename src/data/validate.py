"""Data quality audit: schema, dtypes, missing values, duplicates, cleaning rules, target.

Writes ``reports/data_quality_report.md``, its figures and ``reports/metrics/data_quality.json``.
Every number in the report is computed here from the raw file; nothing is typed by hand.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config
from src.data.clean import CleaningResult, clean_claims, parse_types
from src.data.load import compare_columns, load_raw, log_overview
from src.reporting.outputs import (
    GENERATED_NOTE,
    figure_markdown,
    markdown_table,
    percent_columns,
    write_json,
    write_text,
)
from src.reporting.plots import (
    new_figure,
    percent_axis,
    save_figure,
    series_color,
    thousands_axis,
)

logger = logging.getLogger(__name__)

FIG_TARGET = "dq_target_distribution"
FIG_HOURS = "dq_accident_hour"
FIG_ARTIFACTS = "dq_recording_artifacts_by_year"
YEAR = "Accident year"
CLAIMS = "Claims"


@dataclass
class QualityAudit:
    """Everything the data quality report needs, computed once."""

    raw_shape: tuple[int, int]
    schema: dict[str, list[str]]
    overview: pd.DataFrame
    duplicates: dict[str, int]
    cleaning: CleaningResult
    target_stats: dict[str, float]
    target_percentiles: pd.DataFrame
    hour_counts: pd.Series
    artifacts: pd.DataFrame


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_schema(raw: pd.DataFrame, cfg: Config) -> dict[str, list[str]]:
    """Compare the file's columns with the config and fail if any are missing.

    Raises:
        ValueError: If a documented column is missing from the file.
    """
    diff = compare_columns(raw.columns, cfg.expected_columns)
    if diff["unexpected"]:
        logger.warning("Columns in the file but not in the config: %s", diff["unexpected"])
    if diff["missing"]:
        raise ValueError(f"Columns in the config but missing from the file: {diff['missing']}")
    logger.info("All %d expected columns are present.", len(cfg.expected_columns))
    return diff


def column_overview(raw: pd.DataFrame, typed: pd.DataFrame) -> pd.DataFrame:
    """Per-column dtype before and after parsing, missing and unparseable counts."""
    raw_missing = raw.isna().sum()
    return pd.DataFrame(
        {
            "Column": raw.columns,
            "Raw dtype": raw.dtypes.astype(str).to_numpy(),
            "Parsed dtype": typed[raw.columns].dtypes.astype(str).to_numpy(),
            "Missing": raw_missing.to_numpy(),
            "Unparseable": (typed[raw.columns].isna().sum() - raw_missing).to_numpy(),
            "Unique values": raw.nunique().to_numpy(),
        }
    )


def duplicate_counts(raw: pd.DataFrame, id_column: str) -> dict[str, int]:
    """Count exact duplicates, duplicate IDs and duplicates that differ only by ID."""
    return {
        "Exact duplicate rows": int(raw.duplicated().sum()),
        "Duplicate claim numbers": int(raw[id_column].duplicated().sum()),
        "Rows identical apart from the claim number": int(
            raw.drop(columns=id_column).duplicated().sum()
        ),
    }


def percentile_label(fraction: float) -> str:
    """Format 0.999 as '99.9%' and 0.5 as '50%'."""
    return f"{fraction * 100:g}%"


def target_percentiles(target: pd.Series, fractions: list[float]) -> pd.DataFrame:
    """Target percentiles on the raw and log1p scale."""
    values = target.quantile(fractions).to_numpy()
    return pd.DataFrame(
        {
            "Percentile": [percentile_label(f) for f in fractions],
            "Ultimate cost": values,
            "log1p(cost)": np.log1p(values),
        }
    )


def top_share(target: pd.Series, fraction: float) -> float:
    """Share of total cost carried by the largest ``fraction`` of claims."""
    n_top = max(1, math.ceil(len(target) * fraction))
    return float(target.nlargest(n_top).sum() / target.sum())


def target_stats(target: pd.Series, top_fractions: list[float]) -> dict[str, float]:
    """Summary statistics of the target, including skewness and cost concentration."""
    stats = {
        "count": int(target.count()),
        "min": float(target.min()),
        "median": float(target.median()),
        "mean": float(target.mean()),
        "max": float(target.max()),
        "std": float(target.std()),
        "skew_raw": float(target.skew()),
        "skew_log1p": float(np.log1p(target).skew()),
    }
    for fraction in top_fractions:
        stats[f"top_{fraction * 100:g}pct_cost_share"] = top_share(target, fraction)
    return stats


def accident_hour_counts(typed: pd.DataFrame, cfg: Config) -> pd.Series:
    """Number of claims per hour of the accident timestamp (0-23)."""
    hours = typed[cfg.col("accident_date")].dt.hour
    return hours.value_counts().reindex(range(24), fill_value=0).sort_index()


def yearly_artifacts(typed: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Share of claims per accident year carrying each suspected recording default."""
    flags = cfg["cleaning"]["flags"]
    accident, wage = cfg.cols(["accident_date", "weekly_wages"])
    hour = flags["suspect_accident_hour"]
    frame = pd.DataFrame({YEAR: typed[accident].dt.year})
    for value in flags["heaped_wage_values"]:
        frame[f"Wage = {value:g}"] = typed[wage] == value
    frame[f"Time {hour:02d}:00"] = typed[accident].dt.hour == hour
    table = frame.groupby(YEAR).mean()
    table.insert(0, CLAIMS, frame.groupby(YEAR).size())
    return table.reset_index()


def run_audit(raw: pd.DataFrame, cfg: Config) -> QualityAudit:
    """Run every check on the raw table and collect the results."""
    schema = check_schema(raw, cfg)
    typed = parse_types(raw, cfg)
    reporting = cfg["reporting"]
    target = typed[cfg.col("target")]
    return QualityAudit(
        raw_shape=raw.shape,
        schema=schema,
        overview=column_overview(raw, typed),
        duplicates=duplicate_counts(raw, cfg.col("claim_id")),
        cleaning=clean_claims(raw, cfg),
        target_stats=target_stats(target, reporting["top_share_fractions"]),
        target_percentiles=target_percentiles(target, reporting["target_percentiles"]),
        hour_counts=accident_hour_counts(typed, cfg),
        artifacts=yearly_artifacts(typed, cfg),
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_target_distribution(target: pd.Series, cfg: Config) -> Path:
    """Histogram of the target on the raw scale (clipped) and on a log scale."""
    p, r = cfg["plots"], cfg["reporting"]
    fig, (ax_raw, ax_log) = new_figure(cfg, ncols=2)
    color = series_color(cfg, 0)
    clip = target.quantile(r["raw_hist_clip_percentile"])
    bar_style = {"color": color, "edgecolor": p["surface"], "linewidth": 1}

    ax_raw.hist(target[target <= clip], bins=p["hist_bins"], **bar_style)
    ax_raw.set_title(
        f"Raw scale (smallest {percentile_label(r['raw_hist_clip_percentile'])} of claims)"
    )
    thousands_axis(ax_raw, "x")

    ax_log.hist(target, bins=np.geomspace(target.min(), target.max(), p["hist_bins"]), **bar_style)
    ax_log.set_xscale("log")
    ax_log.set_title("Log scale (all claims)")
    thousands_axis(ax_log, "x")
    top = ax_log.get_ylim()[1]
    for fraction in r["reference_percentiles"]:
        value = target.quantile(fraction)
        ax_log.axvline(value, color=p["ink_muted"], linewidth=1)
        ax_log.text(
            value,
            top,
            f" {percentile_label(fraction)}\n {value:,.0f}",
            color=p["ink_secondary"],
            va="top",
            fontsize=p["font_size"] - 1,
            bbox={"facecolor": p["surface"], "edgecolor": "none", "pad": 1},
        )
    for ax in (ax_raw, ax_log):
        ax.set_xlabel("Ultimate claim cost")
        ax.set_ylabel(CLAIMS)
        thousands_axis(ax, "y")
    return save_figure(fig, cfg, FIG_TARGET)


def plot_accident_hours(counts: pd.Series, cfg: Config) -> Path:
    """Bar chart of claims per accident hour with the suspect hour highlighted."""
    p = cfg["plots"]
    hour = cfg["cleaning"]["flags"]["suspect_accident_hour"]
    fig, ax = new_figure(cfg)
    colors = [series_color(cfg, 0) if h == hour else p["de_emphasis"] for h in counts.index]
    ax.bar(counts.index, counts.to_numpy(), width=0.7, color=colors)
    ax.text(
        hour,
        counts[hour],
        f"{counts[hour]:,}",
        ha="center",
        va="bottom",
        color=p["ink_primary"],
        fontsize=p["font_size"] - 1,
    )
    ax.set_xticks(counts.index, [f"{h:02d}" for h in counts.index])
    ax.set_title(f"Claims by hour of accident ({hour:02d}:00 highlighted)")
    ax.set_xlabel("Hour of accident")
    ax.set_ylabel(CLAIMS)
    thousands_axis(ax)
    return save_figure(fig, cfg, FIG_HOURS)


def plot_yearly_artifacts(table: pd.DataFrame, cfg: Config) -> Path:
    """Line chart of the share of claims with each recording default, by accident year."""
    p = cfg["plots"]
    fig, ax = new_figure(cfg)
    series = [c for c in table.columns if c not in (YEAR, CLAIMS)]
    years = table[YEAR]
    for slot, name in enumerate(series):
        ax.plot(
            years,
            table[name],
            color=series_color(cfg, slot),
            label=name,
            marker="o",
            markersize=5,
            markeredgecolor=p["surface"],
            markeredgewidth=1.5,
        )
        ax.text(
            years.iloc[-1] + 0.3,
            table[name].iloc[-1],
            f"{name}: {table[name].iloc[-1]:.0%}",
            color=p["ink_secondary"],
            va="center",
            fontsize=p["font_size"] - 1,
        )
    ax.set_xlim(years.min() - 0.5, years.max() + 3)
    ax.set_xticks(years[::2])
    ax.set_title("Share of claims with a suspected recording default, by accident year")
    ax.set_xlabel(YEAR)
    ax.set_ylabel("Share of claims")
    percent_axis(ax)
    ax.legend(loc="upper left")
    return save_figure(fig, cfg, FIG_ARTIFACTS)


# ---------------------------------------------------------------------------
# Written takeaways (templated from computed values)
# ---------------------------------------------------------------------------


def target_takeaway(stats: dict[str, float], top_fractions: list[float]) -> str:
    """Two to three sentences describing the target distribution."""
    shares = " and ".join(
        f"the largest {percentile_label(f)} carry {stats[f'top_{f * 100:g}pct_cost_share']:.0%}"
        for f in top_fractions
    )
    return (
        f"Ultimate cost runs from {stats['min']:,.0f} to {stats['max']:,.0f}. The mean "
        f"({stats['mean']:,.0f}) is {stats['mean'] / stats['median']:.1f} times the median "
        f"({stats['median']:,.0f}), and skewness falls from {stats['skew_raw']:.1f} on the raw "
        f"scale to {stats['skew_log1p']:.2f} after a log transform. Of all claim cost, {shares}, "
        "so any error measured in currency is dominated by a small number of very large claims."
    )


def hour_takeaway(counts: pd.Series, typed: pd.DataFrame, cfg: Config) -> str:
    """Describe how unusual the suspect accident hour is and when it occurs."""
    r, hour = cfg["reporting"], cfg["cleaning"]["flags"]["suspect_accident_hour"]
    lo, hi = r["hour_comparison_window"]
    accident = typed[cfg.col("accident_date")]
    years = accident.dt.year
    recent = sorted(years.unique())[-r["recent_years"] :]
    at_hour = accident.dt.hour == hour
    share_recent = (at_hour & years.isin(recent)).sum() / at_hour.sum()
    return (
        f"{counts[hour]:,} accidents are timed at exactly {hour:02d}:00, against an average of "
        f"{counts.loc[lo:hi].mean():,.0f} per hour between {lo:02d}:00 and {hi:02d}:00. "
        f"{share_recent:.0%} of the {hour:02d}:00 claims come from the last {len(recent)} "
        f"accident years ({recent[0]}-{recent[-1]}), which points to a change in how unknown "
        "times were recorded rather than a real spike in midnight injuries."
    )


def _series_trend(table: pd.DataFrame, name: str) -> str:
    """One clause describing how a share moves from the first year, via its peak, to the last."""
    first, last = table.iloc[0], table.iloc[-1]
    peak = table.loc[table[name].idxmax()]
    start = f"{name} goes from {first[name]:.0%} of claims in {int(first[YEAR])}"
    if int(peak[YEAR]) == int(last[YEAR]):
        return f"{start} to {last[name]:.0%} in {int(last[YEAR])}"
    return (
        f"{start} to a peak of {peak[name]:.0%} in {int(peak[YEAR])}, "
        f"then {last[name]:.0%} in {int(last[YEAR])}"
    )


def artifacts_takeaway(table: pd.DataFrame) -> str:
    """Describe how each suspected default changes over accident years."""
    parts = [_series_trend(table, c) for c in table.columns if c not in (YEAR, CLAIMS)]
    return (
        "; ".join(parts) + ". These defaults come and go with the accident year, so they differ "
        "between the earlier training period and the most recent test period. Models need an "
        "indicator for each one rather than treating the value as real."
    )


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------


def format_pct(value: float) -> str:
    """Format a percentage with two decimals, showing tiny non-zero values as '<0.01'."""
    return "<0.01" if 0 < value < 0.01 else f"{value:.2f}"


def _rule_table(log: pd.DataFrame) -> str:
    """Render cleaning log rows as a markdown table."""
    log = log.assign(pct_rows=log["pct_rows"].map(format_pct))
    table = log.rename(
        columns={
            "column": "Column",
            "issue": "Issue",
            "rows": "Rows",
            "pct_rows": "% of rows",
            "action": "Action",
            "rationale": "Rationale",
        }
    )[["Column", "Issue", "Rows", "% of rows", "Action", "Rationale"]]
    return markdown_table(table, floatfmt=("", "", ",.0f", "", "", ""))


def summary_section(audit: QualityAudit) -> str:
    """Headline facts about the file and the cleaning."""
    n_rows, n_cols = audit.raw_shape
    log = audit.cleaning.log
    changed = log[log["changes_data"] & (log["rows"] > 0)]
    flagged = log[~log["changes_data"] & (log["rows"] > 0)]
    dropped = n_rows - len(audit.cleaning.data)
    missing = audit.overview.loc[audit.overview["Missing"] > 0, ["Column", "Missing"]]
    missing_text = (
        ", ".join(f"{row.Column} ({row.Missing:,})" for row in missing.itertuples())
        if len(missing)
        else "none"
    )
    duplicates = sum(audit.duplicates.values())
    unexpected = audit.schema["unexpected"]
    unexpected_text = ", ".join(unexpected) if unexpected else "no unexpected columns"
    return "\n".join(
        [
            "## Summary",
            "",
            f"- The raw file has **{n_rows:,} claims and {n_cols} columns**; every expected "
            f"column is present, with {unexpected_text}.",
            f"- Columns with missing values in the raw file: {missing_text}.",
            f"- Duplicates found: {duplicates:,} (exact rows, claim numbers, or rows identical "
            "apart from the claim number).",
            f"- **{len(changed)} cleaning rules changed data**, making {changed['rows'].sum():,} "
            f"value edits; {dropped:,} rows were dropped, leaving {len(audit.cleaning.data):,}.",
            f"- **{len(flagged)} further checks** found unusual but plausible values that were "
            "kept as recorded.",
            "",
        ]
    )


def schema_section(audit: QualityAudit) -> str:
    """Column dtypes, missing values and duplicates."""
    duplicates = pd.DataFrame(
        {"Check": list(audit.duplicates), "Rows": list(audit.duplicates.values())}
    )
    return "\n".join(
        [
            "## 1. Schema, dtypes and missing values",
            "",
            "Dates arrive as text and are parsed with a fixed format; numeric columns are "
            "coerced to numbers. *Unparseable* counts values that were present but could not be "
            "parsed.",
            "",
            markdown_table(audit.overview),
            "",
            "### Duplicates",
            "",
            markdown_table(duplicates),
            "",
        ]
    )


def rules_section(audit: QualityAudit) -> str:
    """The cleaning rules that changed data and the checks that did not."""
    log = audit.cleaning.log
    return "\n".join(
        [
            "## 2. Cleaning rules",
            "",
            "Rules run in the order listed; each count is the number of rows affected at that "
            "step, as a share of all raw rows. Every threshold is fixed in `config/config.yaml` "
            "from domain limits, never from the relationship with the target, and nothing is "
            "learned from the data, so the same function is applied unchanged to new claims in "
            "the app.",
            "",
            "**Scope of this audit.** It inspects all rows, including the most recent period "
            "that later becomes the test set. That is acceptable for fixed validity rules like "
            "these (they describe whether a value is possible, not how it relates to cost), but "
            "no statistic learned here is ever used by a model.",
            "",
            "**Why values are set to missing instead of filled here.** Filling a value with a "
            "median or mode learns something from the data. If that were done on the full "
            "dataset, information from the test period would leak into training. So this stage "
            "only marks bad values as missing; Phase 3 imputes them inside the model pipeline, "
            "fitted on training rows only.",
            "",
            "### Rules that change the data",
            "",
            _rule_table(log[log["changes_data"]]),
            "",
            "### Checks that leave the data unchanged",
            "",
            _rule_table(log[~log["changes_data"]]),
            "",
        ]
    )


def examples_section(audit: QualityAudit, cfg: Config) -> str:
    """Sample rows affected by each rule that changed data."""
    log = audit.cleaning.log
    changed = log.loc[log["changes_data"] & (log["rows"] > 0), "issue"]
    lines = [
        "### Example rows affected by each data-changing rule",
        "",
        f"Up to {cfg['cleaning']['examples_per_rule']} rows per rule, shown as they were before "
        "the rule ran. *Row* is the 0-based data row in `train.csv` (header excluded).",
        "",
    ]
    for issue in changed:
        sample = audit.cleaning.examples[issue].reset_index(names="Row")
        lines += [f"**{issue}**", "", markdown_table(sample), ""]
    return "\n".join(lines)


def artifacts_section(
    audit: QualityAudit, typed: pd.DataFrame, figures: dict[str, Path], cfg: Config
) -> str:
    """Suspected recording defaults: the midnight time and the heaped wage values."""
    report = cfg.path("data_quality_report")
    shares = [c for c in audit.artifacts.columns if c not in (YEAR, CLAIMS)]
    return "\n".join(
        [
            "## 3. Recording artifacts",
            "",
            "Some values look like system defaults rather than real measurements. They are not "
            "errors to delete, but a model must not take them at face value.",
            "",
            figure_markdown(
                figures[FIG_HOURS],
                report,
                "Claims by hour of accident",
                hour_takeaway(audit.hour_counts, typed, cfg),
            ),
            figure_markdown(
                figures[FIG_ARTIFACTS],
                report,
                "Recording defaults by accident year",
                artifacts_takeaway(audit.artifacts),
            ),
            markdown_table(percent_columns(audit.artifacts, shares), floatfmt=",.0f"),
            "",
        ]
    )


def target_section(audit: QualityAudit, figures: dict[str, Path], cfg: Config) -> str:
    """Distribution of the target on the raw and log scale."""
    report = cfg.path("data_quality_report")
    return "\n".join(
        [
            f"## 4. Target: `{cfg.col('target')}`",
            "",
            figure_markdown(
                figures[FIG_TARGET],
                report,
                "Distribution of ultimate claim cost",
                target_takeaway(audit.target_stats, cfg["reporting"]["top_share_fractions"]),
            ),
            markdown_table(audit.target_percentiles, floatfmt=("", ",.0f", ".2f")),
            "",
            "**The target is not capped.** The largest claims are real and are exactly the ones "
            "the high-cost classifier must find, so removing or capping them would hide the "
            "problem the project exists to solve. The skew is handled by the modelling choices "
            "instead: log-scale targets and Tweedie/Gamma losses (Phase 5), and metrics reported "
            "on both scales.",
            "",
        ]
    )


def implications_section(audit: QualityAudit) -> str:
    """What the audit means for the next phases."""
    log = audit.cleaning.log
    set_missing = log[
        log["changes_data"] & (log["rows"] > 0) & log["action"].str.contains("missing")
    ]
    per_column = set_missing.groupby("column")["rows"].sum()
    missing_text = ", ".join(f"{column} ({rows:,})" for column, rows in per_column.items())
    stats = audit.target_stats
    return "\n".join(
        [
            "## 5. What this means for modelling",
            "",
            f"- **Imputation happens later, on training rows only.** Values set to missing: "
            f"{missing_text}. Phase 3 imputes them inside the scikit-learn pipeline (median "
            "within the part-time / full-time group) and adds missing-value indicators.",
            f"- **The target needs a log scale or a skew-aware loss.** Skewness is "
            f"{stats['skew_raw']:.1f} raw versus {stats['skew_log1p']:.2f} on log1p. "
            "Original-scale RMSE will be dominated by the largest claims, so it is reported "
            "alongside log-scale metrics.",
            "- **Recording defaults need indicator features.** The heaped wage values and the "
            "00:00 accident time change with the accident year, so they will look different in "
            "the time-based test period than in training.",
            "- **Initial estimates of 1 are kept as recorded**, because the adjuster baseline must "
            "use what adjusters actually entered.",
            "",
        ]
    )


def render_report(
    audit: QualityAudit, typed: pd.DataFrame, figures: dict[str, Path], cfg: Config
) -> str:
    """Assemble the full data quality report."""
    return "\n".join(
        [
            "# Data Quality Report",
            "",
            GENERATED_NOTE.format(stage="validate"),
            "",
            summary_section(audit),
            schema_section(audit),
            rules_section(audit),
            examples_section(audit, cfg),
            artifacts_section(audit, typed, figures, cfg),
            target_section(audit, figures, cfg),
            implications_section(audit),
        ]
    )


def audit_metrics(audit: QualityAudit) -> dict[str, object]:
    """Machine-readable summary of the audit for reports/metrics."""
    log = audit.cleaning.log.drop(columns="rationale")
    return {
        "raw_rows": audit.raw_shape[0],
        "raw_columns": audit.raw_shape[1],
        "clean_rows": len(audit.cleaning.data),
        "schema": audit.schema,
        "duplicates": audit.duplicates,
        "rules": log.to_dict(orient="records"),
        "target": audit.target_stats,
        "target_percentiles": dict(
            zip(
                audit.target_percentiles["Percentile"],
                audit.target_percentiles["Ultimate cost"],
                strict=True,
            )
        ),
    }


def run_validation(cfg: Config) -> None:
    """Run the full audit and write the report, figures and metrics."""
    raw = load_raw(cfg)
    log_overview(raw, head_rows=cfg["data"]["inspect_head_rows"])
    audit = run_audit(raw, cfg)
    typed = parse_types(raw, cfg)
    figures = {
        FIG_TARGET: plot_target_distribution(typed[cfg.col("target")], cfg),
        FIG_HOURS: plot_accident_hours(audit.hour_counts, cfg),
        FIG_ARTIFACTS: plot_yearly_artifacts(audit.artifacts, cfg),
    }
    write_text(render_report(audit, typed, figures, cfg), cfg.path("data_quality_report"))
    write_json(audit_metrics(audit), cfg.path("data_quality_metrics"))
