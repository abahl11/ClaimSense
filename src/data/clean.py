"""Deterministic, rule-based cleaning of the raw claims table.

Every rule uses a fixed threshold from ``config.yaml`` and learns nothing from the data, so
the same function cleans the training data and new claims at prediction time. Invalid
values are set to missing here; they are imputed later inside the model pipeline
(Phase 3), where the imputer is fit on training rows only.

The same rule list drives both the cleaning and the data quality report, so the report
always describes exactly what the cleaning does.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.config import Config

logger = logging.getLogger(__name__)

Detector = Callable[[pd.DataFrame], pd.Series]
Fixer = Callable[[pd.DataFrame, pd.Series], pd.DataFrame]

FLAG_ONLY = "None (kept as recorded)"
IMPUTE_LATER = "Set to missing (imputed in the model pipeline)"
DAYS_PER_WEEK = 7


@dataclass(frozen=True)
class Rule:
    """One data quality rule: how to detect an issue and what to do about it.

    Attributes:
        column: Column the rule concerns. The rule is skipped if the column is absent
            (for example the target column when scoring new claims).
        issue: Plain-language description of the problem.
        action: What the cleaning does to affected rows.
        rationale: Why this action was chosen.
        detect: Returns a boolean mask of affected rows.
        fix: Applies the action to the masked rows; ``None`` means flag only.
        show: Columns displayed for example rows in the report (defaults to ``column``).
    """

    column: str
    issue: str
    action: str
    rationale: str
    detect: Detector
    fix: Fixer | None = None
    show: tuple[str, ...] = ()


@dataclass
class CleaningResult:
    """Output of :func:`clean_claims`.

    Attributes:
        data: The cleaned table.
        log: One row per rule with the number and share of rows affected.
        examples: Sample affected rows (before the fix) keyed by rule issue.
    """

    data: pd.DataFrame
    log: pd.DataFrame
    examples: dict[str, pd.DataFrame] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Small helpers shared by the rules
# ---------------------------------------------------------------------------


def set_missing(column: str) -> Fixer:
    """Return a fixer that sets ``column`` to missing on the masked rows."""

    def fix(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
        out = df.copy()
        out[column] = out[column].mask(mask)
        return out

    return fix


def set_value(column: str, value: Any) -> Fixer:
    """Return a fixer that sets ``column`` to ``value`` on the masked rows."""

    def fix(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
        out = df.copy()
        out[column] = out[column].mask(mask, value)
        return out

    return fix


def drop_rows(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Remove the masked rows."""
    return df.loc[~mask]


def normalize_text(text: pd.Series, bad_codepoints: Iterable[int]) -> pd.Series:
    """Replace bad characters with spaces, collapse whitespace, strip and upper-case.

    Args:
        text: Free-text column.
        bad_codepoints: Unicode code points to replace (for example 0xFFFD).

    Returns:
        The normalised text.
    """
    bad = "".join(re.escape(chr(code)) for code in bad_codepoints)
    if bad:
        text = text.str.replace(f"[{bad}]", " ", regex=True)
    return text.str.replace(r"\s+", " ", regex=True).str.strip().str.upper()


def normalize_column(column: str, bad_codepoints: Sequence[int]) -> Fixer:
    """Return a fixer that normalises the text in ``column`` on the masked rows."""

    def fix(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
        out = df.copy()
        out.loc[mask, column] = normalize_text(out.loc[mask, column], bad_codepoints)
        return out

    return fix


def report_delay_days(accident: pd.Series, reported: pd.Series) -> pd.Series:
    """Whole days between the accident date and the report date.

    The report date has no time of day, so both are compared at day level.
    """
    return (reported.dt.normalize() - accident.dt.normalize()).dt.days


# ---------------------------------------------------------------------------
# Rule groups. Order matters: later rules see the result of earlier ones.
# ---------------------------------------------------------------------------


def _row_rules(cfg: Config) -> list[Rule]:
    """Rules that drop whole rows."""
    claim_id, target, accident = cfg.cols(["claim_id", "target", "accident_date"])
    return [
        Rule(
            column=claim_id,
            issue="Duplicate claim number",
            action="Dropped (first occurrence kept)",
            rationale="A claim number identifies one claim. A duplicate would be counted twice "
            "and could land in both the training and the test set.",
            detect=lambda df: df[claim_id].duplicated(keep="first"),
            fix=drop_rows,
        ),
        Rule(
            column=target,
            issue="Ultimate cost missing or not positive",
            action="Dropped",
            rationale="A claim without a positive final cost cannot be used to train or "
            "evaluate a cost model.",
            detect=lambda df: df[target].isna() | (df[target] <= 0),
            fix=drop_rows,
        ),
        Rule(
            column=accident,
            issue="Accident date missing or unparseable",
            action="Dropped",
            rationale="The accident date orders claims for the time-based split and drives the "
            "calendar features, so a claim cannot be used without it.",
            detect=lambda df: df[accident].isna(),
            fix=drop_rows,
        ),
    ]


def _date_rules(cfg: Config) -> list[Rule]:
    """Rules on the report date."""
    accident, reported = cfg.cols(["accident_date", "report_date"])
    long_delay = cfg["cleaning"]["flags"]["long_report_delay_days"]

    def delay(df: pd.DataFrame) -> pd.Series:
        return report_delay_days(df[accident], df[reported])

    return [
        Rule(
            column=reported,
            issue="Report date missing or unparseable",
            action="Left missing (report delay imputed in the model pipeline)",
            rationale="The claim is still usable; only the report delay is unknown.",
            detect=lambda df: df[reported].isna(),
        ),
        Rule(
            column=reported,
            issue="Report date earlier than accident date",
            action="Report date set to missing",
            rationale="A claim cannot be reported before the accident. Rather than guess which "
            "date is wrong, the report date is treated as unknown.",
            detect=lambda df: delay(df) < 0,
            fix=set_missing(reported),
            show=(accident, reported),
        ),
        Rule(
            column=reported,
            issue=f"Reported more than {long_delay} days after the accident",
            action=FLAG_ONLY,
            rationale="Late reporting is real and is a known cost driver in workers' "
            "compensation, so it is kept as information, not treated as an error.",
            detect=lambda df: delay(df) > long_delay,
        ),
    ]


def _hours_rules(cfg: Config) -> list[Rule]:
    """Rules on hours and days worked per week."""
    c = cfg["cleaning"]
    hours, days, part_time = cfg.cols(["hours_per_week", "days_per_week", "part_time"])
    max_hours, max_per_day = c["hours_per_week_max"], c["hours_per_day_max"]
    min_days, max_days = c["days_per_week_range"]
    shown = (hours, days, part_time)
    return [
        Rule(
            column=hours,
            issue="Hours worked per week is 0 or negative",
            action=IMPUTE_LATER,
            rationale="Every claimant was working when injured, so 0 hours is a missing value "
            "recorded as 0. It is imputed later with the median of the same part-time / "
            "full-time group, learned from training rows only.",
            detect=lambda df: df[hours] <= 0,
            fix=set_missing(hours),
            show=shown,
        ),
        Rule(
            column=hours,
            issue=f"Hours worked per week above {max_hours}",
            action=IMPUTE_LATER,
            rationale=f"More than {max_hours} hours a week means over "
            f"{max_hours // DAYS_PER_WEEK} hours every day. Several values look like a "
            "misplaced decimal point (380 for 38.0), but guessing the correction is riskier "
            "than imputing a typical value.",
            detect=lambda df: df[hours] > max_hours,
            fix=set_missing(hours),
            show=shown,
        ),
        Rule(
            column=days,
            issue=f"Days worked per week outside {min_days}-{max_days}",
            action=IMPUTE_LATER,
            rationale="A week has seven days and every claimant worked at least one.",
            detect=lambda df: df[days].notna() & ~df[days].between(min_days, max_days),
            fix=set_missing(days),
            show=shown,
        ),
        Rule(
            column=days,
            issue=f"More than {max_per_day} hours per working day (hours / days)",
            action="Days worked set to missing (imputed in the model pipeline)",
            rationale=f"More than {max_per_day} hours in a day is impossible, so one of the two "
            "fields is wrong. Most of these rows are a standard full-time week (such as 38 "
            "hours) recorded against a single working day, so the days value is treated as "
            "the error.",
            detect=lambda df: df[hours] / df[days] > max_per_day,
            fix=set_missing(days),
            show=shown,
        ),
    ]


def _wage_rules(cfg: Config) -> list[Rule]:
    """Rules on weekly wages."""
    c = cfg["cleaning"]
    wage, hours, part_time = cfg.cols(["weekly_wages", "hours_per_week", "part_time"])
    min_weekly, min_hourly = c["weekly_wage_min"], c["hourly_wage_min"]
    shown = (wage, hours, part_time)
    return [
        Rule(
            column=wage,
            issue=f"Weekly wage below {min_weekly}",
            action=IMPUTE_LATER,
            rationale=f"No real job pays under {min_weekly} a week, whatever the hours. Values "
            "such as 1.0 look like placeholders entered when the wage was unknown.",
            detect=lambda df: df[wage] < min_weekly,
            fix=set_missing(wage),
            show=shown,
        ),
        Rule(
            column=wage,
            issue=f"Implied hourly wage below {min_hourly} (weekly wage / hours)",
            action="Weekly wage set to missing (imputed in the model pipeline)",
            rationale="An hourly rate this low is far below any minimum or junior wage in the "
            "period, so the wage was mis-recorded for the hours worked.",
            detect=lambda df: df[wage] / df[hours] < min_hourly,
            fix=set_missing(wage),
            show=shown,
        ),
    ]


def _unknown_flag_rule(column: str, unknown: str) -> Rule:
    """Flag rows already recorded with the unknown code."""
    return Rule(
        column=column,
        issue=f"{column} recorded as {unknown} (unknown)",
        action=FLAG_ONLY,
        rationale="A legitimate category. It is kept as its own level rather than imputed, "
        "so the model can learn whether 'unknown' itself carries signal.",
        detect=lambda df: df[column] == unknown,
    )


def _category_rule(column: str, valid: list[str], fill: str | None) -> Rule:
    """Map missing or unrecognised category codes to ``fill`` (or to missing)."""
    if fill is None:
        action = IMPUTE_LATER
        rationale = "No safe default exists, so the value is imputed in the model pipeline."
        fixer = set_missing(column)
    else:
        action = f"Set to {fill}"
        rationale = (
            f"{fill} (unknown) is already a category, so blanks and unrecognised codes join it "
            "instead of being guessed."
        )
        fixer = set_value(column, fill)
    return Rule(
        column=column,
        issue=f"{column} missing or not one of {', '.join(valid)}",
        action=action,
        rationale=rationale,
        detect=lambda df: ~df[column].isin(valid),
        fix=fixer,
    )


def _category_rules(cfg: Config) -> list[Rule]:
    """Rules on the categorical columns."""
    c = cfg["cleaning"]
    unknown = c["unknown_category"]
    rules: list[Rule] = []
    for name, spec in c["categories"].items():
        column = cfg.col(name)
        if unknown in spec["valid"]:
            rules.append(_unknown_flag_rule(column, unknown))
        rules.append(_category_rule(column, spec["valid"], spec["fill"]))
    return rules


def _text_rules(cfg: Config) -> list[Rule]:
    """Rules on the free-text claim description."""
    c = cfg["cleaning"]
    text, bad = cfg.col("description"), c["text_bad_codepoints"]
    return [
        Rule(
            column=text,
            issue="Claim description missing or empty",
            action="Filled with an empty string",
            rationale="Text features need a string; an empty description simply matches no "
            "keywords.",
            detect=lambda df: df[text].isna() | (df[text].str.strip() == ""),
            fix=set_value(text, c["text_missing_fill"]),
        ),
        Rule(
            column=text,
            issue="Claim description has encoding errors, extra whitespace or lower case",
            action="Normalised (bad characters replaced, whitespace collapsed, upper-cased)",
            rationale="The keyword rules match exact words, so every description needs one "
            "consistent form.",
            detect=lambda df: df[text].notna() & (df[text] != normalize_text(df[text], bad)),
            fix=normalize_column(text, bad),
        ),
    ]


def _heaped_wage_rule(wage: str, value: float) -> Rule:
    """Flag a weekly wage value that appears suspiciously often."""
    return Rule(
        column=wage,
        issue=f"Weekly wage exactly {value:g}",
        action=FLAG_ONLY,
        rationale="An exact value this common, concentrated in particular accident years (see "
        "the figure below), looks like a system default rather than a real wage. It is too "
        "common to discard; Phase 3 adds an indicator so models can learn that the wage may "
        "not be real.",
        detect=lambda df: df[wage] == value,
    )


def _plausibility_rules(cfg: Config) -> list[Rule]:
    """Flag-only checks: unusual but possible values that are kept as recorded."""
    f, codes = cfg["cleaning"]["flags"], cfg["data"]["codes"]
    age, wage, hours, part_time, accident, initial = cfg.cols(
        ["age", "weekly_wages", "hours_per_week", "part_time", "accident_date", "initial_estimate"]
    )
    hour = f["suspect_accident_hour"]
    label_note = (
        "The part-time label may describe the contract rather than the hours, so both "
        "fields are kept as recorded."
    )
    return [
        Rule(
            column=age,
            issue=f"Age below {f['young_age_below']}",
            action=FLAG_ONLY,
            rationale="Plausible: school-age part-time work is legal in some jurisdictions.",
            detect=lambda df: df[age] < f["young_age_below"],
        ),
        Rule(
            column=age,
            issue=f"Age above {f['old_age_above']}",
            action=FLAG_ONLY,
            rationale="Plausible: some people keep working past typical retirement age.",
            detect=lambda df: df[age] > f["old_age_above"],
        ),
        Rule(
            column=wage,
            issue=f"Weekly wage above {f['high_weekly_wage_above']:,}",
            action=FLAG_ONLY,
            rationale="Plausible for senior or specialist roles. Linear models use log wages, "
            "so these values do not dominate, and tree models are unaffected by scale.",
            detect=lambda df: df[wage] > f["high_weekly_wage_above"],
        ),
        *[_heaped_wage_rule(wage, value) for value in f["heaped_wage_values"]],
        Rule(
            column=part_time,
            issue=f"Part-time with at least {f['part_time_hours_at_least']} hours a week",
            action=FLAG_ONLY,
            rationale=label_note,
            detect=lambda df: (df[part_time] == codes["part_time"])
            & (df[hours] >= f["part_time_hours_at_least"]),
        ),
        Rule(
            column=part_time,
            issue=f"Full-time with fewer than {f['full_time_hours_below']} hours a week",
            action=FLAG_ONLY,
            rationale=label_note,
            detect=lambda df: (df[part_time] == codes["full_time"])
            & (df[hours] < f["full_time_hours_below"]),
        ),
        Rule(
            column=accident,
            issue=f"Accident time recorded as exactly {hour:02d}:00",
            action=FLAG_ONLY,
            rationale=f"Far more accidents are timed at {hour:02d}:00 than in the hours around "
            "it (see the figure below), which suggests 'time unknown' stored as midnight. "
            "Phase 3 adds a flag for it.",
            detect=lambda df: df[accident].dt.hour == hour,
        ),
        Rule(
            column=initial,
            issue=f"Initial estimate of {f['nominal_initial_estimate_max']} or less",
            action=FLAG_ONLY,
            rationale="A nominal placeholder rather than a cost estimate. Kept as recorded "
            "because the initial-estimate baseline must reflect what adjusters actually "
            "entered. Phase 3 adds an indicator.",
            detect=lambda df: df[initial] <= f["nominal_initial_estimate_max"],
        ),
    ]


def build_rules(cfg: Config) -> list[Rule]:
    """Return every cleaning rule and plausibility check, in the order they run."""
    return [
        *_row_rules(cfg),
        *_date_rules(cfg),
        *_hours_rules(cfg),
        *_wage_rules(cfg),
        *_category_rules(cfg),
        *_text_rules(cfg),
        *_plausibility_rules(cfg),
    ]


# ---------------------------------------------------------------------------
# Running the rules
# ---------------------------------------------------------------------------


def parse_types(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Parse date columns and coerce numeric columns; unparseable values become missing.

    Args:
        df: Raw claims table.
        cfg: Loaded project configuration.

    Returns:
        A copy with datetime and numeric dtypes applied.
    """
    d = cfg["data"]
    out = df.copy()
    for column in cfg.cols(d["date_columns"]):
        out[column] = pd.to_datetime(out[column], format=d["datetime_format"], errors="coerce")
    for column in cfg.cols(d["numeric_columns"]):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def apply_rules(df: pd.DataFrame, rules: Sequence[Rule], n_examples: int) -> CleaningResult:
    """Run rules in order, fixing data and recording how many rows each one affected.

    Args:
        df: Typed claims table (output of :func:`parse_types`).
        rules: Rules to apply, in order.
        n_examples: Number of affected rows to keep as examples per rule.

    Returns:
        The cleaned data, a per-rule log and example rows.
    """
    n_rows = len(df)
    records, examples = [], {}
    for rule in rules:
        if rule.column not in df.columns:
            logger.debug("Skipping rule '%s': column %s not present", rule.issue, rule.column)
            continue
        mask = rule.detect(df).astype("boolean").fillna(False).astype(bool)
        count = int(mask.sum())
        if count:
            examples[rule.issue] = df.loc[mask, list(rule.show or (rule.column,))].head(n_examples)
        if count and rule.fix is not None:
            df = rule.fix(df, mask)
        records.append(
            {
                "column": rule.column,
                "issue": rule.issue,
                "rows": count,
                "pct_rows": 100 * count / n_rows if n_rows else 0.0,
                "action": rule.action,
                "changes_data": rule.fix is not None,
                "rationale": rule.rationale,
            }
        )
        logger.info("%-68s %6d rows", rule.issue, count)
    return CleaningResult(
        data=df.reset_index(drop=True), log=pd.DataFrame(records), examples=examples
    )


def clean_claims(df: pd.DataFrame, cfg: Config) -> CleaningResult:
    """Parse types and apply every cleaning rule.

    Works on training data and on new claims without a target column.

    Args:
        df: Raw claims table.
        cfg: Loaded project configuration.

    Returns:
        The cleaned data with the rule log and example rows.
    """
    typed = parse_types(df, cfg)
    return apply_rules(typed, build_rules(cfg), cfg["cleaning"]["examples_per_rule"])


def save_clean(df: pd.DataFrame, cfg: Config) -> None:
    """Write the cleaned table to ``paths.clean_data`` as Parquet (keeps dtypes)."""
    path = cfg.path("clean_data")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Saved %d cleaned rows to %s", len(df), path)
