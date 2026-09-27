"""Exploratory data analysis: answers the Phase 2 questions with figures and a narrative.

Sections 1-4 use training claims only (accidents before the time-split cutoff), so no design
choice is informed by the test period. Section 5 shows every accident year because its job
is to describe how the test period differs from training; nothing in it is used to fit or
tune a model. Every number and takeaway in ``reports/eda.md`` is computed here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from sklearn.linear_model import LinearRegression

from src.config import Config
from src.data.load import load_clean
from src.features.tabular import (
    accident_year,
    age_band,
    delay_band,
    delay_days,
    dependents_group,
    wage_band,
)
from src.features.text_rules import BODY_PART, INJURY_CAUSE, INJURY_NATURE, extract_text_rules
from src.models.split import holdout_mask
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
    sequential_cmap,
    series_color,
    thousands_axis,
    year_axis,
)

logger = logging.getLogger(__name__)

YEAR = "accident_year"
DELAY = "report_delay_days"
RATIO = "ultimate_to_initial"
IS_TEST = "is_test_period"
CLAIMS = "Claims"
SHARE = "Share of claims"
MEDIAN = "Median cost"
MEAN = "Mean cost"
COST_SHARE = "Share of total cost"
HIGH_RATE = "High-cost rate"
SHARE_COLUMNS = [SHARE, COST_SHARE, HIGH_RATE]
MAX_FLAT_TICKS = 6  # more tick labels than this are rotated so they do not overlap
NEGLIGIBLE_SHARE = 0.05  # a share of growth below this is described as "essentially none"


@dataclass
class EdaData:
    """Enriched claims plus the facts every section needs."""

    claims: pd.DataFrame
    train: pd.DataFrame
    cutoff: pd.Timestamp
    high_cost_threshold: float
    target: str
    initial: str


@dataclass
class Section:
    """One report section: markdown blocks, key-finding bullets and metrics."""

    blocks: list[str]
    findings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Data preparation and shared summaries
# ---------------------------------------------------------------------------


def enrich(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Add derived columns (year, delay, bands, text labels, ratio, period flag)."""
    is_test, cutoff = holdout_mask(df, cfg)
    derived = pd.concat(
        [
            accident_year(df, cfg),
            delay_days(df, cfg),
            age_band(df, cfg),
            wage_band(df, cfg),
            delay_band(df, cfg),
            dependents_group(df, cfg),
            extract_text_rules(df[cfg.col("description")], cfg),
        ],
        axis=1,
    )
    out = pd.concat([df, derived], axis=1)
    out[RATIO] = out[cfg.col("target")] / out[cfg.col("initial_estimate")]
    out[IS_TEST] = is_test
    return out, cutoff


def load_eda_data(cfg: Config) -> EdaData:
    """Load the cleaned claims, enrich them and fix the high-cost threshold on training."""
    claims, cutoff = enrich(load_clean(cfg), cfg)
    train = claims.loc[~claims[IS_TEST]]
    target = cfg.col("target")
    threshold = float(train[target].quantile(cfg["labels"]["high_cost_quantile"]))
    logger.info("EDA on %d training claims (cutoff %s)", len(train), cutoff)
    return EdaData(claims, train, cutoff, threshold, target, cfg.col("initial_estimate"))


def resolve_column(name: str, cfg: Config) -> str:
    """Map a logical source-column name to its real name; derived names pass through."""
    return cfg.col(name) if name in cfg["data"]["columns"] else name


def group_summary(df: pd.DataFrame, by: str, target: str, threshold: float) -> pd.DataFrame:
    """Claim count, cost statistics and high-cost rate for each group of ``by``."""
    flagged = df.assign(_high=df[target] >= threshold)
    grouped = flagged.groupby(by, observed=True)
    summary = pd.DataFrame(
        {
            CLAIMS: grouped.size(),
            SHARE: grouped.size() / len(df),
            MEDIAN: grouped[target].median(),
            MEAN: grouped[target].mean(),
            COST_SHARE: grouped[target].sum() / df[target].sum(),
            HIGH_RATE: grouped["_high"].mean(),
        }
    )
    return summary.rename_axis(by).reset_index()


def order_by_count(summary: pd.DataFrame, by: str, other_label: str) -> pd.DataFrame:
    """Sort groups by claim count, keeping the catch-all ``other`` group last."""
    is_other = summary[by] == other_label
    ordered = summary[~is_other].sort_values(CLAIMS, ascending=False)
    return pd.concat([ordered, summary[is_other]], ignore_index=True)


def readable(values: pd.Series) -> pd.Series:
    """Turn snake_case category labels into plain words for charts and tables."""
    return values.astype(str).str.replace("_", " ")


def summary_table(summary: pd.DataFrame, by: str, label: str) -> str:
    """Markdown table of a group summary with percentages formatted."""
    table = percent_columns(summary, SHARE_COLUMNS).rename(columns={by: label})
    table[label] = readable(table[label])
    return markdown_table(table, floatfmt=",.0f")


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def hbar(ax: Axes, values: pd.Series, labels: pd.Series, fmt: str, cfg: Config) -> None:
    """Horizontal bars in the first series colour with the value at each bar tip."""
    p = cfg["plots"]
    y = np.arange(len(values))
    ax.barh(y, values.to_numpy(), height=0.6, color=series_color(cfg, 0))
    for yi, value in zip(y, values, strict=True):
        ax.text(value, yi, " " + fmt.format(value), va="center", color=p["ink_secondary"])
    ax.set_yticks(y, labels.tolist())
    ax.set_xlim(0, values.max() * 1.22)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.spines["left"].set_visible(True)
    ax.tick_params(axis="x", labelbottom=False, length=0)


def vbar(ax: Axes, labels: pd.Series, values: pd.Series, fmt: str, cfg: Config) -> None:
    """Vertical bars in the first series colour with the value on each bar cap."""
    p = cfg["plots"]
    x = np.arange(len(values))
    ax.bar(x, values.to_numpy(), width=0.6, color=series_color(cfg, 0))
    for xi, value in zip(x, values, strict=True):
        ax.text(
            xi,
            value,
            fmt.format(value),
            ha="center",
            va="bottom",
            color=p["ink_secondary"],
            fontsize=p["font_size"] - 2,
        )
    ax.set_xticks(x, labels.tolist())


def fractional_year(timestamp: pd.Timestamp) -> float:
    """Timestamp as a decimal year, e.g. 2002-07-02 -> about 2002.5."""
    days = 366 if timestamp.is_leap_year else 365
    return timestamp.year + (timestamp.dayofyear - 1) / days


def mark_test_period(ax: Axes, cutoff: pd.Timestamp, cfg: Config) -> None:
    """Draw the train/test boundary on a year axis."""
    p = cfg["plots"]
    x = fractional_year(cutoff)
    ax.axvline(x, color=p["ink_muted"], linewidth=1)
    ax.text(
        x,
        1.0,
        " test period",
        transform=ax.get_xaxis_transform(),
        va="top",
        color=p["ink_secondary"],
        fontsize=p["font_size"] - 1,
    )


def end_labels_collide(ends: pd.Series, ax: Axes, min_gap: float) -> bool:
    """True if any two end values are closer than ``min_gap`` of the y-axis range."""
    low, high = ax.get_ylim()
    gaps = np.diff(np.sort(ends.to_numpy()))
    return bool(len(gaps) and gaps.min() < min_gap * (high - low))


def year_lines(ax: Axes, table: pd.DataFrame, columns: list[str], fmt: str, cfg: Config) -> None:
    """One line per column against accident year, with a legend and end-value labels.

    End labels are skipped when they would collide; the legend and the table in the
    report then carry the values instead of labels nudged away from their lines.
    """
    p = cfg["plots"]
    for slot, name in enumerate(columns):
        ax.plot(
            table.index,
            table[name],
            color=series_color(cfg, slot),
            label=name,
            marker="o",
            markersize=4,
            markeredgecolor=p["surface"],
            markeredgewidth=1.2,
        )
    ends = table[columns].iloc[-1]
    if not end_labels_collide(ends, ax, cfg["eda"]["label_collision_gap"]):
        for value in ends:
            ax.text(
                table.index[-1] + 0.3,
                value,
                fmt.format(value),
                va="center",
                color=p["ink_secondary"],
                fontsize=p["font_size"] - 1,
            )
    ax.set_xlim(table.index.min() - 0.5, table.index.max() + 2)
    ax.set_xlabel("Accident year")
    year_axis(ax)
    if len(columns) > 1:
        ax.legend(loc="upper left")


# ---------------------------------------------------------------------------
# Section 1: adjuster accuracy
# ---------------------------------------------------------------------------


def adjuster_accuracy(data: EdaData, cfg: Config) -> dict[str, float]:
    """How far the adjuster's initial estimate is from the ultimate cost (training claims)."""
    e = cfg["eda"]
    ultimate, initial = data.train[data.target], data.train[data.initial]
    ratio = ultimate / initial
    abs_pct_error = (initial - ultimate).abs() / ultimate
    quantile = cfg["labels"]["high_cost_quantile"]
    is_high = ultimate >= data.high_cost_threshold
    initial_high = initial >= initial.quantile(quantile)
    return {
        "median_ratio": float(ratio.median()),
        "median_abs_pct_error": float(abs_pct_error.median()),
        "typical_factor_off": float(np.exp(np.log(ratio).abs().median())),
        "share_within_tolerance": float((abs_pct_error <= e["adjuster_tolerance"]).mean()),
        "share_under_by_factor": float((ratio > e["adjuster_large_miss_factor"]).mean()),
        "share_over_by_factor": float((ratio < 1 / e["adjuster_large_miss_factor"]).mean()),
        "aggregate_coverage": float(initial.sum() / ultimate.sum()),
        "spearman": float(ultimate.corr(initial, method="spearman")),
        "high_cost_caught_by_initial": float(initial_high[is_high].mean()),
    }


def plot_initial_vs_ultimate(data: EdaData, cfg: Config) -> Path:
    """Log-log density of initial estimate against ultimate cost, with y = x."""
    p = cfg["plots"]
    fig, ax = new_figure(cfg)
    x, y = data.train[data.initial], data.train[data.target]
    density = ax.hexbin(
        x,
        y,
        xscale="log",
        yscale="log",
        gridsize=cfg["eda"]["hexbin_gridsize"],
        bins="log",
        mincnt=1,
        cmap=sequential_cmap(cfg),
        linewidths=0,
    )
    fig.colorbar(density, ax=ax, label="Claims per cell (log scale)")
    limits = [y.min(), max(x.max(), y.max())]
    ax.plot(limits, limits, color=p["ink_secondary"], linewidth=1)
    ax.text(limits[1], limits[1], "ultimate = initial ", ha="right", va="bottom")
    ax.set_ylim(y.min() * 0.8, y.max() * 1.25)
    ax.axhline(data.high_cost_threshold, color=p["ink_muted"], linewidth=1)
    ax.text(
        x.min(),
        data.high_cost_threshold,
        f" high-cost threshold ({data.high_cost_threshold:,.0f})",
        va="bottom",
        color=p["ink_secondary"],
    )
    ax.grid(False)
    ax.set_xlabel("Initial estimate (adjuster, at reporting)")
    ax.set_ylabel("Ultimate cost")
    ax.set_title("Initial estimate vs ultimate cost, training claims (log scales)")
    thousands_axis(ax, "x")
    thousands_axis(ax, "y")
    return save_figure(fig, cfg, "eda_initial_vs_ultimate")


def adjuster_section(data: EdaData, cfg: Config) -> Section:
    """Section 1: how good are initial estimates? This motivates the project."""
    stats = adjuster_accuracy(data, cfg)
    e = cfg["eda"]
    tol, factor = e["adjuster_tolerance"], e["adjuster_large_miss_factor"]
    top_decile = f"{1 - cfg['labels']['high_cost_quantile']:.0%}"
    missed = 1 - stats["high_cost_caught_by_initial"]
    common = data.train[data.initial].value_counts().head(e["common_values_shown"])
    common_share = common.sum() / len(data.train)
    common_text = ", ".join(f"{value:,.0f}" for value in common.index)
    figure = plot_initial_vs_ultimate(data, cfg)
    takeaway = (
        f"Points spread widely around the diagonal: the typical claim's initial estimate is off "
        f"by a factor of {stats['typical_factor_off']:.2f} (median absolute error "
        f"{stats['median_abs_pct_error']:.0%} of the final cost), and only "
        f"{stats['share_within_tolerance']:.0%} of estimates land within ±{tol:.0%}. The "
        f"vertical stripes are round-number estimates: the three most common values "
        f"({common_text}) alone make up {common_share:.0%} of claims, which suggests many "
        "initial reserves are set to standard amounts rather than assessed individually."
    )
    table = pd.DataFrame(
        {
            "Measure": [
                "Median ratio ultimate / initial",
                "Median absolute error, as % of ultimate",
                f"Share of estimates within ±{tol:.0%}",
                f"Share under-estimated by more than {factor:g}x",
                f"Share over-estimated by more than {factor:g}x",
                "Sum of initial estimates / sum of ultimate costs",
                "Rank correlation (Spearman) initial vs ultimate",
                "High-cost claims whose initial estimate was also top 10%",
            ],
            "Value": [
                f"{stats['median_ratio']:.2f}",
                f"{stats['median_abs_pct_error']:.0%}",
                f"{stats['share_within_tolerance']:.0%}",
                f"{stats['share_under_by_factor']:.0%}",
                f"{stats['share_over_by_factor']:.0%}",
                f"{stats['aggregate_coverage']:.0%}",
                f"{stats['spearman']:.2f}",
                f"{stats['high_cost_caught_by_initial']:.0%}",
            ],
        }
    )
    blocks = [
        "## 1. How good are adjusters' initial estimates?",
        "",
        "This is the question that motivates the project: if the estimate recorded at "
        "reporting time were already accurate, there would be nothing to improve.",
        "",
        figure_markdown(figure, cfg.path("eda_report"), "Initial vs ultimate cost", takeaway),
        markdown_table(table),
        "",
        f"The initial estimate is informative (rank correlation {stats['spearman']:.2f}) but "
        f"imprecise. Of the claims that end up in the top {top_decile} by ultimate cost, "
        f"{stats['high_cost_caught_by_initial']:.0%} already had an initial estimate in the top "
        f"{top_decile}; the other **{missed:.0%} looked ordinary at reporting time**. Those are "
        "the claims that get escalated late, and the ones the high-cost flag is meant to catch. "
        "(Formal baseline metrics come in Phase 5.)",
        "",
    ]
    finding = (
        f"**Adjusters' initial estimates are informative but imprecise.** The typical estimate "
        f"is off by a factor of {stats['typical_factor_off']:.2f}, only "
        f"{stats['share_within_tolerance']:.0%} are within ±{tol:.0%}, and {missed:.0%} of "
        f"eventual high-cost claims did not have a top-{top_decile} initial estimate."
    )
    return Section(blocks, [finding], {"adjuster": stats})


# ---------------------------------------------------------------------------
# Section 2: injury types
# ---------------------------------------------------------------------------


def plot_category(summary: pd.DataFrame, by: str, title: str, cfg: Config, name: str) -> Path:
    """Paired horizontal bars: share of claims and median cost, in the same order."""
    fig, (ax_share, ax_cost) = new_figure(cfg, ncols=2, sharey=True)
    labels = readable(summary[by])
    hbar(ax_share, summary[SHARE], labels, "{:.0%}", cfg)
    hbar(ax_cost, summary[MEDIAN], labels, "{:,.0f}", cfg)
    ax_share.invert_yaxis()
    ax_share.set_title(f"{title}: share of training claims")
    ax_cost.set_title(f"{title}: median ultimate cost")
    return save_figure(fig, cfg, name)


def category_takeaway(summary: pd.DataFrame, by: str, noun: str, data: EdaData, cfg: Config) -> str:
    """Contrast the most frequent group with the most expensive one."""
    named = summary[summary[by] != cfg["text_rules"]["other_label"]]
    frequent = named.loc[named[CLAIMS].idxmax()]
    eligible = named[named[CLAIMS] >= cfg["eda"]["min_group_size"]]
    expensive = eligible.loc[eligible[MEDIAN].idxmax()]
    biggest = named.loc[named[COST_SHARE].idxmax()]
    overall = data.train[data.target].median()

    def name(row: pd.Series) -> str:
        return str(row[by]).replace("_", " ")

    frequent_text = f"{frequent[SHARE]:.0%} of claims"
    expensive_text = (
        f"median {expensive[MEDIAN]:,.0f}, {expensive[MEDIAN] / overall:.1f}x the overall "
        f"median of {overall:,.0f}"
    )
    if frequent[by] == expensive[by]:
        text = (
            f"**{name(frequent).capitalize()}** is both the most frequent {noun} "
            f"({frequent_text}) and the "
            f"most expensive by median ({expensive_text})."
        )
    else:
        text = (
            f"The most frequent {noun} is **{name(frequent)}** ({frequent_text}, median "
            f"{frequent[MEDIAN]:,.0f}), but the most expensive is **{name(expensive)}** "
            f"({expensive_text}). Frequency and severity rank differently."
        )
    text += (
        f" **{name(biggest).capitalize()}** carries the largest share of total cost "
        f"({biggest[COST_SHARE]:.0%})."
    )
    cheapest = eligible.loc[eligible[MEDIAN].idxmin()]
    text += (
        f" At the other end, **{name(cheapest)}** claims have a median of only "
        f"{cheapest[MEDIAN]:,.0f}."
    )
    return text


def injury_section(data: EdaData, cfg: Config) -> Section:
    """Section 2: most frequent vs most expensive injury types."""
    other = cfg["text_rules"]["other_label"]
    report = cfg.path("eda_report")
    blocks = [
        "## 2. Most frequent vs most expensive injury types",
        "",
        "Injury types come from the rule-based keyword taxonomy on the claim description "
        "(`src/features/text_rules.py`; keyword lists in `config/config.yaml`). This is a first "
        "pass: Phase 3 measures its coverage and refines it, and this report is regenerated "
        "with the refined rules. Groups are sorted by frequency with *other* (no keyword "
        f"matched) last; 'most expensive' only considers groups with at least "
        f"{cfg['eda']['min_group_size']} claims.",
        "",
    ]
    findings, metrics = [], {}
    fields = [
        (INJURY_NATURE, "Injury nature", "injury nature"),
        (BODY_PART, "Body part", "body part"),
    ]
    for by, title, noun in fields:
        summary = order_by_count(
            group_summary(data.train, by, data.target, data.high_cost_threshold), by, other
        )
        figure = plot_category(summary, by, title, cfg, f"eda_{by}")
        takeaway = category_takeaway(summary, by, noun, data, cfg)
        blocks += [
            f"### {title}",
            "",
            figure_markdown(figure, report, title, takeaway),
            summary_table(summary, by, title),
            "",
        ]
        findings.append(f"**{title}.** {takeaway}")
        metrics[by] = summary.set_index(by)[[CLAIMS, MEDIAN, MEAN]].to_dict(orient="index")
        eligible = summary[
            (summary[by] != other) & (summary[CLAIMS] >= cfg["eda"]["min_group_size"])
        ]
        metrics[f"{by.removeprefix('injury_')}_median_spread"] = float(
            eligible[MEDIAN].max() / eligible[MEDIAN].min()
        )
    causes = order_by_count(
        group_summary(data.train, INJURY_CAUSE, data.target, data.high_cost_threshold),
        INJURY_CAUSE,
        other,
    )
    blocks += ["### Injury cause", "", summary_table(causes, INJURY_CAUSE, "Injury cause"), ""]
    return Section(blocks, findings, metrics)


# ---------------------------------------------------------------------------
# Section 3: reporting delay
# ---------------------------------------------------------------------------


def plot_delay(summary: pd.DataFrame, cfg: Config) -> Path:
    """Median cost and high-cost rate by reporting-delay band (two panels, one axis each)."""
    fig, (ax_cost, ax_rate) = new_figure(cfg, ncols=2)
    labels = summary["delay_band"].astype(str)
    vbar(ax_cost, labels, summary[MEDIAN], "{:,.0f}", cfg)
    vbar(ax_rate, labels, summary[HIGH_RATE], "{:.0%}", cfg)
    ax_cost.set_title("Median ultimate cost")
    ax_rate.set_title("Share of claims that are high-cost")
    thousands_axis(ax_cost)
    percent_axis(ax_rate)
    for ax in (ax_cost, ax_rate):
        ax.set_xlabel("Days from accident to report")
        ax.tick_params(axis="x", labelsize=cfg["plots"]["font_size"] - 1)
    return save_figure(fig, cfg, "eda_report_delay")


def delay_section(data: EdaData, cfg: Config) -> Section:
    """Section 3: does cost rise with reporting delay?"""
    summary = group_summary(data.train, "delay_band", data.target, data.high_cost_threshold)
    first, last = summary.iloc[0], summary.iloc[-1]
    middle = summary.iloc[1:-1]
    peak = middle.loc[middle[MEDIAN].idxmax()]
    rho = data.train[DELAY].corr(data.train[data.target], method="spearman")
    strength = "weak" if abs(rho) < cfg["eda"]["weak_correlation"] else "clear"
    takeaway = (
        f"The relationship is not a straight line. Median cost is {first[MEDIAN]:,.0f} for "
        f"claims reported in under a week, peaks at {peak[MEDIAN]:,.0f} for "
        f"{peak['delay_band']} days, eases off, then jumps to {last[MEDIAN]:,.0f} for the "
        f"{last[SHARE]:.1%} of claims reported after more than a year. The high-cost rate climbs "
        f"more steadily, from {first[HIGH_RATE]:.0%} to {last[HIGH_RATE]:.0%}. The overall rank "
        f"correlation is {strength} ({rho:.2f}) because the effect sits in the tails, which a "
        "tree model can pick up but a single linear term would miss."
    )
    figure = plot_delay(summary, cfg)
    blocks = [
        "## 3. Cost vs reporting delay",
        "",
        figure_markdown(figure, cfg.path("eda_report"), "Cost by reporting delay", takeaway),
        summary_table(summary, "delay_band", "Days to report"),
        "",
    ]
    finding = f"**Reporting delay matters mainly at the extremes.** {takeaway}"
    metrics = {
        "delay_spearman": float(rho),
        "delay_last_to_first_median": float(last[MEDIAN] / first[MEDIAN]),
    }
    return Section(blocks, [finding], metrics)


# ---------------------------------------------------------------------------
# Section 4: claimant and employment characteristics
# ---------------------------------------------------------------------------


def claimant_groups(cfg: Config) -> list[tuple[str, str]]:
    """(column, title) pairs compared in the claimant section."""
    return [
        ("age_band", "Age band"),
        ("wage_band", "Weekly wage band"),
        (cfg.col("part_time"), "Part-time / full-time"),
        ("dependents_group", "Total dependents"),
    ]


def plot_claimant(summaries: dict[str, pd.DataFrame], cfg: Config) -> Path:
    """Small multiples: median cost by each claimant characteristic, one shared y-axis."""
    fig, axes = new_figure(cfg, ncols=2, nrows=2, sharey=True)
    names = {code: name.capitalize() for code, name in part_time_names(cfg).items()}
    for ax, (by, title) in zip(axes.flat, claimant_groups(cfg), strict=True):
        summary = summaries[by]
        labels = summary[by].astype(str).replace(names)
        vbar(ax, labels, summary[MEDIAN], "{:,.0f}", cfg)
        ax.set_title(title)
        ax.tick_params(axis="x", labelsize=cfg["plots"]["font_size"] - 2)
        if len(labels) > MAX_FLAT_TICKS:
            ax.tick_params(axis="x", labelrotation=30)
        thousands_axis(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("Median ultimate cost")
    return save_figure(fig, cfg, "eda_claimant_characteristics")


def spread_clause(
    summary: pd.DataFrame, by: str, title: str, names: dict[str, str]
) -> tuple[str, float]:
    """Describe the lowest and highest median group; return the clause and max/min ratio."""
    low, high = summary.loc[summary[MEDIAN].idxmin()], summary.loc[summary[MEDIAN].idxmax()]
    ratio = high[MEDIAN] / low[MEDIAN]
    low_name, high_name = (names.get(str(g), str(g)) for g in (low[by], high[by]))
    clause = (
        f"{title.lower()} from {low[MEDIAN]:,.0f} ({low_name}) to {high[MEDIAN]:,.0f} "
        f"({high_name}), {ratio:.1f}x"
    )
    return clause, ratio


def part_time_names(cfg: Config) -> dict[str, str]:
    """Readable names for the part-time / full-time codes."""
    codes = cfg["data"]["codes"]
    return {codes["part_time"]: "part-time", codes["full_time"]: "full-time"}


def default_wage_table(data: EdaData, cfg: Config) -> pd.DataFrame:
    """Median cost of claims carrying each suspected default wage versus all others."""
    wage = data.train[cfg.col("weekly_wages")]
    defaults = cfg["cleaning"]["flags"]["heaped_wage_values"]
    groups = pd.Series("Other wages", index=data.train.index)
    for value in defaults:
        groups = groups.mask(wage == value, f"Exactly {value:g}")
    summary = group_summary(
        data.train.assign(wage_group=groups), "wage_group", data.target, data.high_cost_threshold
    )
    return summary


def claimant_section(data: EdaData, cfg: Config) -> Section:
    """Section 4: cost by age, wage, part-time status and dependents."""
    summaries = {
        by: group_summary(data.train, by, data.target, data.high_cost_threshold)
        for by, _ in claimant_groups(cfg)
    }
    names = part_time_names(cfg)
    groups = claimant_groups(cfg)
    clauses = [spread_clause(summaries[by], by, title, names) for by, title in groups]
    ranked = sorted(zip(clauses, groups, strict=True), key=lambda item: item[0][1], reverse=True)
    widest, narrowest = ranked[:2], ranked[-2:]

    def listed(items: list) -> str:
        return " and ".join(f"{title.lower()} ({ratio:.1f}x)" for (_, ratio), (_, title) in items)

    takeaway = (
        "Median cost ranges by "
        + "; by ".join(clause for clause, _ in clauses)
        + f". The widest spreads are by {listed(widest)}; the narrowest by {listed(narrowest)}. "
        "These are unadjusted comparisons: groups can also differ in injury type, which the "
        "model untangles by using all features together."
    )
    strongest = ranked[0]
    figure = plot_claimant(summaries, cfg)
    blocks = [
        "## 4. Cost vs age, wage, part-time status and dependents",
        "",
        figure_markdown(
            figure, cfg.path("eda_report"), "Cost by claimant characteristics", takeaway
        ),
    ]
    for by, title in claimant_groups(cfg):
        blocks += [f"**{title}**", "", summary_table(summaries[by], by, title), ""]
    blocks += [
        "**Suspected default wages.** Claims whose wage equals a suspected system default "
        "(see the data quality report) are compared with all other claims below. If their cost "
        "differs markedly, the default value is carrying information about the recording era "
        "rather than about the worker's pay, which is why Phase 3 adds an indicator for it.",
        "",
        summary_table(default_wage_table(data, cfg), "wage_group", "Weekly wage"),
        "",
    ]
    finding = f"**Claimant characteristics.** {takeaway.split(' These are unadjusted')[0]}"
    return Section(blocks, [finding], {"claimant_max_spread": float(strongest[0][1])})


# ---------------------------------------------------------------------------
# Section 5: trends over accident year (all claims)
# ---------------------------------------------------------------------------


def yearly_summary(data: EdaData) -> pd.DataFrame:
    """Claims, median and mean ultimate cost per accident year (all claims)."""
    grouped = data.claims.groupby(YEAR)[data.target]
    return pd.DataFrame({CLAIMS: grouped.size(), MEDIAN: grouped.median(), MEAN: grouped.mean()})


def plot_yearly(table: pd.DataFrame, data: EdaData, cfg: Config) -> Path:
    """Claims per year and cost per year, as two panels (no dual axis)."""
    fig, (ax_count, ax_cost) = new_figure(cfg, ncols=2)
    ax_count.bar(table.index, table[CLAIMS], width=0.6, color=series_color(cfg, 0))
    ax_count.set_title("Claims per accident year")
    ax_count.set_xlabel("Accident year")
    year_axis(ax_count)
    thousands_axis(ax_count)
    year_lines(ax_cost, table, [MEDIAN, MEAN], "{:,.0f}", cfg)
    ax_cost.set_title("Ultimate cost per accident year")
    thousands_axis(ax_cost)
    for ax in (ax_count, ax_cost):
        mark_test_period(ax, data.cutoff, cfg)
    return save_figure(fig, cfg, "eda_yearly_counts_costs")


def cost_index(df: pd.DataFrame, target: str, controls: list[str]) -> pd.Series:
    """Geometric-mean cost per accident year relative to the first year, holding controls fixed.

    Fits ``log(cost) ~ year + controls`` by least squares. The exponentiated year effects
    give each year's typical cost relative to the first year for a claim with the same
    controls. With no controls this is simply the ratio of geometric means.
    """
    years = df[YEAR].astype(int)
    year_dummies = pd.get_dummies(
        years.astype("category"), prefix="y", drop_first=True, dtype=float
    )
    parts = [year_dummies]
    if controls:
        parts.append(pd.get_dummies(df[controls].astype(str), drop_first=True, dtype=float))
    design = pd.concat(parts, axis=1)
    model = LinearRegression().fit(design, np.log(df[target]))
    effects = model.coef_[: year_dummies.shape[1]]
    index_years = sorted(years.unique())
    return pd.Series(np.exp(np.r_[0.0, effects]), index=index_years)


def plot_cost_index(index: pd.DataFrame, data: EdaData, cfg: Config) -> Path:
    """Raw vs mix-adjusted cost index by accident year."""
    fig, ax = new_figure(cfg)
    year_lines(ax, index, list(index.columns), "{:.2f}x", cfg)
    ax.axhline(1.0, color=cfg["plots"]["baseline"], linewidth=1)
    ax.set_title("Typical claim cost relative to the first accident year")
    ax.set_ylabel("Cost index (first year = 1.0)")
    mark_test_period(ax, data.cutoff, cfg)
    return save_figure(fig, cfg, "eda_cost_index")


def index_takeaway(index: pd.DataFrame) -> tuple[str, float]:
    """Compare raw and mix-adjusted growth; return the text and the share due to mix."""
    raw_col, adj_col = index.columns
    raw, adjusted = index[raw_col].iloc[-1], index[adj_col].iloc[-1]
    first, last = index.index[0], index.index[-1]
    mix_share = 1 - np.log(adjusted) / np.log(raw) if raw > 1 else float("nan")
    if mix_share < NEGLIGIBLE_SHARE:
        explained = "essentially none of the growth"
    else:
        explained = f"about {mix_share:.0%} of the (log) growth"
    verdict = (
        "the rise behaves like a change in price level (inflation), not a shift towards more "
        "serious claims"
        if mix_share < 0.5
        else "the rise is mainly a shift towards more serious claims, not a change in price level"
    )
    low_year = index[raw_col].idxmin()
    text = (
        f"The typical claim in {last} costs {raw:.2f}x a {first} claim. Holding injury type, "
        f"body part, cause, age band and part-time status fixed, it still costs {adjusted:.2f}x, "
        f"so changes in claim mix explain {explained}: {verdict}. The path is uneven, with a low "
        f"of {index[raw_col].min():.2f}x in {low_year} before a steep climb."
    )
    return text, float(mix_share)


def anchor_trends(data: EdaData, cfg: Config) -> pd.DataFrame:
    """Per-year medians of ultimate cost, initial estimate and (non-default) wage, indexed."""
    wage = cfg.col("weekly_wages")
    defaults = cfg["cleaning"]["flags"]["heaped_wage_values"]
    real_wage = data.claims[wage].where(~data.claims[wage].isin(defaults))
    grouped = data.claims.assign(_wage=real_wage).groupby(YEAR)
    medians = pd.DataFrame(
        {
            "Ultimate cost": grouped[data.target].median(),
            "Initial estimate": grouped[data.initial].median(),
            "Weekly wage (defaults excluded)": grouped["_wage"].median(),
        }
    )
    return medians / medians.iloc[0]


def plot_anchors(indexed: pd.DataFrame, ratio: pd.Series, data: EdaData, cfg: Config) -> Path:
    """Indexed medians of cost, estimate and wage, and the median ultimate/initial ratio."""
    fig, (ax_index, ax_ratio) = new_figure(cfg, ncols=2)
    year_lines(ax_index, indexed, list(indexed.columns), "{:.1f}x", cfg)
    ax_index.set_title("Median by accident year (first year = 1.0)")
    ax_index.axhline(1.0, color=cfg["plots"]["baseline"], linewidth=1)
    year_lines(
        ax_ratio,
        ratio.to_frame("Median ultimate / initial"),
        ["Median ultimate / initial"],
        "{:.2f}",
        cfg,
    )
    ax_ratio.axhline(1.0, color=cfg["plots"]["baseline"], linewidth=1)
    ax_ratio.set_title("Median ratio of ultimate cost to initial estimate")
    for ax in (ax_index, ax_ratio):
        mark_test_period(ax, data.cutoff, cfg)
    return save_figure(fig, cfg, "eda_price_anchors")


def anchor_takeaway(
    indexed: pd.DataFrame, ratio: pd.Series, cost_range: tuple[float, float], data: EdaData
) -> str:
    """Growth of each anchor and how (in)stable the ultimate/initial ratio is."""
    last = indexed.iloc[-1]
    by_period = data.claims.groupby(IS_TEST)[RATIO].median()
    first_year, last_year, low_year = ratio.index[0], ratio.index[-1], ratio.idxmin()
    return (
        f"From {first_year} to {last_year} the median ultimate cost grows {last.iloc[0]:.1f}x, "
        f"the median initial estimate {last.iloc[1]:.1f}x (jumpy, because estimates cluster on "
        f"round numbers) and the median non-default wage only {last.iloc[2]:.1f}x. So the "
        "initial estimate carries most of the price level, but it is not a perfect anchor: the "
        f"median ultimate/initial ratio falls from {ratio.iloc[0]:.2f} in {first_year} to "
        f"{ratio.min():.2f} in {low_year}, then recovers to {ratio.iloc[-1]:.2f} in {last_year}. "
        f"That range ({ratio.min():.2f}-{ratio.max():.2f}) is much narrower than the cost "
        f"index range ({cost_range[0]:.2f}-{cost_range[1]:.2f}x), and the training and test "
        f"periods have similar medians ({by_period[False]:.2f} vs {by_period[True]:.2f}). "
        "Modelling the ratio (approach B) should reduce the drift problem but will not remove "
        "it, which is why approach C (explicit trending) is also tested."
    )


def trend_section(data: EdaData, cfg: Config) -> Section:
    """Section 5: counts, costs, price vs mix, and price anchors over accident years."""
    report = cfg.path("eda_report")
    yearly = yearly_summary(data)
    first, last = yearly.iloc[0], yearly.iloc[-1]
    yearly_text = (
        f"Claim counts are flat (about {yearly[CLAIMS].mean():,.0f} a year), but costs are not: "
        f"the median rises from {first[MEDIAN]:,.0f} in {yearly.index[0]} to {last[MEDIAN]:,.0f} "
        f"in {yearly.index[-1]} ({last[MEDIAN] / first[MEDIAN]:.1f}x) and the mean from "
        f"{first[MEAN]:,.0f} to {last[MEAN]:,.0f}. The test period (right of the line) is the "
        "most expensive part of the data, so a model trained on earlier years risks "
        "under-predicting it unless it is anchored to current prices."
    )
    controls = [resolve_column(name, cfg) for name in cfg["eda"]["index_controls"]]
    index = pd.DataFrame(
        {
            "Raw (all claims)": cost_index(data.claims, data.target, []),
            "Mix-adjusted": cost_index(data.claims, data.target, controls),
        }
    )
    index_text, mix_share = index_takeaway(index)
    indexed = anchor_trends(data, cfg)
    ratio = data.claims.groupby(YEAR)[RATIO].median()
    raw_index = index.iloc[:, 0]
    anchors_text = anchor_takeaway(indexed, ratio, (raw_index.min(), raw_index.max()), data)
    table = pd.concat([yearly, index, ratio.rename("Median ultimate / initial")], axis=1)
    table = table.rename_axis("Accident year").reset_index()
    blocks = [
        "## 5. Trends over accident year (all claims)",
        "",
        "This section uses every accident year, including the test period, because its purpose "
        "is to describe how the test period differs from training. Nothing here is used to fit "
        "or tune a model; the choice between modelling approaches is made with time-ordered "
        "cross-validation inside the training period (Phase 4).",
        "",
        "### Claim counts and costs",
        "",
        figure_markdown(
            plot_yearly(yearly, data, cfg), report, "Counts and costs by year", yearly_text
        ),
        "### Price change or mix change?",
        "",
        "Costs can rise because the same injury costs more (price change, which needs trending "
        "or anchoring) or because more serious injuries are being claimed (mix change, which "
        "the injury features already capture). A regression of log cost on accident year, with "
        "and without controls for the claim mix, separates the two.",
        "",
        figure_markdown(plot_cost_index(index, data, cfg), report, "Cost index", index_text),
        "### Does the initial estimate track the price level?",
        "",
        figure_markdown(
            plot_anchors(indexed, ratio, data, cfg), report, "Price anchors", anchors_text
        ),
        markdown_table(table, floatfmt=("g", ",.0f", ",.0f", ",.0f", ".2f", ".2f", ".2f")),
        "",
    ]
    findings = [
        f"**Costs rise over time.** {yearly_text.split(' The test period')[0]}",
        f"**Price or mix?** {index_text}",
        f"**Anchoring.** {anchors_text}",
    ]
    metrics = {
        "yearly_median_growth": float(last[MEDIAN] / first[MEDIAN]),
        "cost_index_last_year": index.iloc[-1].to_dict(),
        "mix_share_of_growth": mix_share,
        "anchor_growth_last_year": indexed.iloc[-1].to_dict(),
        "ratio_median_by_period": {
            "train": float(data.train[RATIO].median()),
            "test": float(data.claims.loc[data.claims[IS_TEST], RATIO].median()),
        },
    }
    return Section(blocks, findings, metrics)


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def intro_blocks(data: EdaData, cfg: Config) -> list[str]:
    """Title, provenance and the scope of the analysis."""
    train_dates = data.train[cfg.col("accident_date")]
    n_test = int(data.claims[IS_TEST].sum())
    return [
        "# Exploratory Data Analysis",
        "",
        GENERATED_NOTE.format(stage="eda"),
        "",
        "**Scope.** Sections 1-4 use the training period only: "
        f"{len(data.train):,} claims with accidents from {train_dates.min():%d %b %Y} to "
        f"{train_dates.max():%d %b %Y}. The {n_test:,} most recent claims (accidents from "
        f"{data.cutoff:%d %b %Y}) are the test set and are left out, so that what is learned "
        "here cannot leak into choices evaluated on them. Section 5 is the exception (see there).",
        "",
        f"**High-cost claims** are those at or above the {cfg['labels']['high_cost_quantile']:.0%} "
        f"percentile of ultimate cost in the training period: "
        f"**{data.high_cost_threshold:,.0f}**.",
        "",
    ]


def implications_blocks(metrics: dict[str, Any]) -> list[str]:
    """What the findings mean for feature engineering and modelling (built from metrics)."""
    kind = (
        "mostly a price-level change"
        if metrics["mix_share_of_growth"] < 0.5
        else "mostly a change in claim mix"
    )
    return [
        "## Implications for modelling",
        "",
        f"- **Validation mirrors deployment.** Median cost grows "
        f"{metrics['yearly_median_growth']:.1f}x across the period and the rise is {kind}, so "
        "model choices are made with time-ordered cross-validation inside the training period "
        "(train on earlier years, validate on later ones), never on the test set.",
        "- **Three ways to handle price drift are compared in Phase 5:** (A) predict log cost "
        "directly; (B) predict the log ratio of ultimate cost to the initial estimate, which is "
        "recorded at current prices; (C) deflate costs with a trend fitted on training years, "
        "then re-inflate predictions. The winner is chosen by cross-validation and all three "
        "are reported on the test set.",
        f"- **Text features earn their place.** Median cost differs "
        f"{metrics['nature_median_spread']:.1f}x between the cheapest and dearest injury natures "
        "(groups with enough claims), information that exists only in the free-text description.",
        f"- **Non-linear effects favour tree models.** Delay has a rank correlation of only "
        f"{metrics['delay_spearman']:.2f} with cost, yet the longest-delay band has "
        f"{metrics['delay_last_to_first_median']:.1f}x the median cost of the shortest. Effects "
        "concentrated in the tails like this are captured by gradient-boosted trees without "
        "hand-made thresholds.",
        "- **Recording defaults get indicator features** (default wages, 00:00 accident times), "
        "as planned in the data quality report.",
        "",
    ]


def run_eda(cfg: Config) -> None:
    """Run every EDA section and write ``reports/eda.md`` and ``reports/metrics/eda.json``."""
    data = load_eda_data(cfg)
    sections = [
        adjuster_section(data, cfg),
        injury_section(data, cfg),
        delay_section(data, cfg),
        claimant_section(data, cfg),
        trend_section(data, cfg),
    ]
    all_findings = (finding for section in sections for finding in section.findings)
    findings = [f"{i}. {text}" for i, text in enumerate(all_findings, start=1)]
    metrics: dict[str, Any] = {
        "n_train": len(data.train),
        "n_test": int(data.claims[IS_TEST].sum()),
        "cutoff": data.cutoff,
        "high_cost_threshold": data.high_cost_threshold,
    }
    for section in sections:
        metrics.update(section.metrics)
    blocks = intro_blocks(data, cfg)
    for section in sections:
        blocks += section.blocks
    blocks += ["## Key findings", "", *findings, "", *implications_blocks(metrics)]
    write_text("\n".join(blocks), cfg.path("eda_report"))
    write_json(metrics, cfg.path("eda_metrics"))
