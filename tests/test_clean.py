"""Tests for the deterministic cleaning rules in src/data/clean.py."""

import pandas as pd
import pytest

from src.data.clean import clean_claims, normalize_text, report_delay_days

REPLACEMENT_CHAR = "�"


def rule_rows(result, issue_start: str) -> int:
    """Rows affected by the rule whose issue text starts with ``issue_start``."""
    matches = result.log[result.log["issue"].str.startswith(issue_start)]
    assert len(matches) == 1, f"expected one rule starting with {issue_start!r}"
    return int(matches["rows"].iloc[0])


def test_valid_claim_is_unchanged(cfg, make_claims):
    result = clean_claims(make_claims(), cfg)
    changed = result.log[result.log["changes_data"]]
    assert changed["rows"].sum() == 0
    assert len(result.data) == 1
    assert result.data[cfg.col("weekly_wages")].iloc[0] == 450.0


@pytest.mark.parametrize("hours", [0.0, -5.0])
def test_non_positive_hours_set_to_missing(cfg, make_claims, hours):
    result = clean_claims(make_claims({"hours_per_week": hours}), cfg)
    assert result.data[cfg.col("hours_per_week")].isna().all()
    assert rule_rows(result, "Hours worked per week is 0") == 1


def test_hours_above_max_set_to_missing(cfg, make_claims):
    too_many = cfg["cleaning"]["hours_per_week_max"] + 1
    at_limit = cfg["cleaning"]["hours_per_week_max"]
    result = clean_claims(
        make_claims({"hours_per_week": too_many}, {"hours_per_week": at_limit}), cfg
    )
    hours = result.data[cfg.col("hours_per_week")]
    assert hours.isna().tolist() == [True, False]


@pytest.mark.parametrize("days", [0, 8])
def test_days_outside_week_set_to_missing(cfg, make_claims, days):
    result = clean_claims(make_claims({"days_per_week": days}), cfg)
    assert result.data[cfg.col("days_per_week")].isna().all()


def test_impossible_hours_per_day_sets_days_missing(cfg, make_claims):
    result = clean_claims(make_claims({"hours_per_week": 38.0, "days_per_week": 1}), cfg)
    assert result.data[cfg.col("days_per_week")].isna().all()
    assert result.data[cfg.col("hours_per_week")].iloc[0] == 38.0


def test_hours_per_day_check_runs_after_hours_cleaning(cfg, make_claims):
    # 380 hours is removed by the hours rule first, so days must not also be blanked.
    result = clean_claims(make_claims({"hours_per_week": 380.0, "days_per_week": 5}), cfg)
    assert result.data[cfg.col("days_per_week")].iloc[0] == 5


def test_weekly_wage_below_minimum_set_to_missing(cfg, make_claims):
    result = clean_claims(make_claims({"weekly_wages": 1.0}), cfg)
    assert result.data[cfg.col("weekly_wages")].isna().all()


def test_hourly_wage_below_minimum_sets_wage_missing(cfg, make_claims):
    result = clean_claims(make_claims({"weekly_wages": 15.0, "hours_per_week": 35.0}), cfg)
    assert result.data[cfg.col("weekly_wages")].isna().all()
    assert rule_rows(result, "Implied hourly wage") == 1


def test_low_wage_for_few_hours_is_kept(cfg, make_claims):
    result = clean_claims(make_claims({"weekly_wages": 45.0, "hours_per_week": 5.0}), cfg)
    assert result.data[cfg.col("weekly_wages")].iloc[0] == 45.0


@pytest.mark.parametrize("value", [None, "X"])
def test_missing_or_invalid_marital_status_becomes_unknown(cfg, make_claims, value):
    result = clean_claims(make_claims({"marital_status": value}), cfg)
    assert result.data[cfg.col("marital_status")].iloc[0] == cfg["cleaning"]["unknown_category"]


def test_unknown_gender_is_kept_and_flagged(cfg, make_claims):
    result = clean_claims(make_claims({"gender": "U"}), cfg)
    assert result.data[cfg.col("gender")].iloc[0] == "U"
    assert rule_rows(result, "Gender recorded as U") == 1


def test_invalid_part_time_code_set_to_missing(cfg, make_claims):
    result = clean_claims(make_claims({"part_time": "Z"}), cfg)
    assert result.data[cfg.col("part_time")].isna().all()


def test_report_before_accident_sets_report_date_missing(cfg, make_claims):
    claims = make_claims({"report_date": "2001-03-01T00:00:00Z"})
    result = clean_claims(claims, cfg)
    assert result.data[cfg.col("report_date")].isna().all()
    assert result.data[cfg.col("accident_date")].notna().all()


def test_same_day_report_is_valid(cfg, make_claims):
    claims = make_claims({"report_date": "2001-03-05T00:00:00Z"})
    result = clean_claims(claims, cfg)
    assert result.data[cfg.col("report_date")].notna().all()


def test_duplicate_claim_numbers_are_dropped(cfg, make_claims):
    claims = make_claims({}, {"age": 50})
    claims[cfg.col("claim_id")] = "WC0000001"
    result = clean_claims(claims, cfg)
    assert len(result.data) == 1
    assert result.data[cfg.col("age")].iloc[0] == 35


@pytest.mark.parametrize("target", [0.0, -10.0, None])
def test_non_positive_or_missing_target_is_dropped(cfg, make_claims, target):
    result = clean_claims(make_claims({}, {"target": target}), cfg)
    assert len(result.data) == 1


def test_unparseable_accident_date_is_dropped(cfg, make_claims):
    result = clean_claims(make_claims({}, {"accident_date": "not a date"}), cfg)
    assert len(result.data) == 1


def test_description_is_normalised(cfg, make_claims):
    raw_text = f"caught {REPLACEMENT_CHAR}{REPLACEMENT_CHAR} bar  and mould "
    result = clean_claims(make_claims({"description": raw_text}), cfg)
    assert result.data[cfg.col("description")].iloc[0] == "CAUGHT BAR AND MOULD"


@pytest.mark.parametrize("text", [None, "   "])
def test_missing_description_filled_with_empty_string(cfg, make_claims, text):
    result = clean_claims(make_claims({"description": text}), cfg)
    assert result.data[cfg.col("description")].iloc[0] == cfg["cleaning"]["text_missing_fill"]


def test_flag_only_checks_do_not_change_values(cfg, make_claims):
    flags = cfg["cleaning"]["flags"]
    unusual = {
        "weekly_wages": float(flags["heaped_wage_values"][0]),
        "age": flags["young_age_below"] - 1,
        "accident_date": f"2001-03-05T{flags['suspect_accident_hour']:02d}:00:00Z",
        "initial_estimate": flags["nominal_initial_estimate_max"],
    }
    claims = make_claims(unusual)
    result = clean_claims(claims, cfg)
    for name in ("weekly_wages", "age", "initial_estimate"):
        assert result.data[cfg.col(name)].iloc[0] == claims[cfg.col(name)].iloc[0]
    flagged = result.log[~result.log["changes_data"] & (result.log["rows"] > 0)]
    assert len(flagged) >= len(unusual)


def test_cleaning_works_without_target_column(cfg, make_claims):
    claims = make_claims({"hours_per_week": 0.0}).drop(columns=cfg.col("target"))
    result = clean_claims(claims, cfg)
    assert cfg.col("target") not in result.log["column"].tolist()
    assert result.data[cfg.col("hours_per_week")].isna().all()


def test_cleaning_does_not_modify_input(cfg, make_claims):
    claims = make_claims({"hours_per_week": 0.0, "marital_status": None})
    before = claims.copy()
    clean_claims(claims, cfg)
    pd.testing.assert_frame_equal(claims, before)


def test_cleaning_is_deterministic(cfg, make_claims):
    claims = make_claims({"hours_per_week": 0.0}, {"weekly_wages": 1.0}, {})
    first, second = clean_claims(claims, cfg), clean_claims(claims, cfg)
    pd.testing.assert_frame_equal(first.data, second.data)
    pd.testing.assert_frame_equal(first.log, second.log)


def test_dates_are_parsed_to_datetimes(cfg, make_claims):
    result = clean_claims(make_claims(), cfg)
    for name in cfg["data"]["date_columns"]:
        assert pd.api.types.is_datetime64_any_dtype(result.data[cfg.col(name)])


def test_report_delay_days_counts_whole_days():
    accident = pd.Series(pd.to_datetime(["2001-03-05 23:00", "2001-03-05 01:00"]))
    reported = pd.Series(pd.to_datetime(["2001-03-06", "2001-03-05"]))
    assert report_delay_days(accident, reported).tolist() == [1, 0]


def test_normalize_text_without_bad_characters():
    text = pd.Series(["  slipped   on wet floor "])
    assert normalize_text(text, []).iloc[0] == "SLIPPED ON WET FLOOR"
