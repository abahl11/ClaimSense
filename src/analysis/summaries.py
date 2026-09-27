"""Group summaries of claim cost shared by the EDA and feature reports."""

from __future__ import annotations

import pandas as pd

from src.reporting.outputs import markdown_table, percent_columns

CLAIMS = "Claims"
SHARE = "Share of claims"
MEDIAN = "Median cost"
MEAN = "Mean cost"
COST_SHARE = "Share of total cost"
HIGH_RATE = "High-cost rate"
SHARE_COLUMNS = [SHARE, COST_SHARE, HIGH_RATE]


def group_summary(df: pd.DataFrame, by: str, target: str, threshold: float) -> pd.DataFrame:
    """Claim count, cost statistics and high-cost rate for each group of ``by``.

    Args:
        df: Claims to summarise.
        by: Column to group on.
        target: Cost column.
        threshold: Cost at or above which a claim counts as high-cost.

    Returns:
        One row per group.
    """
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
