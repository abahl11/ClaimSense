"""Tests for the rule-based keyword taxonomy in src/features/text_rules.py."""

import pandas as pd
import pytest

from src.features.text_rules import (
    BODY_PART,
    INJURY_CAUSE,
    INJURY_NATURE,
    NUM_BODY_PARTS,
    coverage,
    extract_text_rules,
)


def labels_for(cfg, *texts: str) -> pd.DataFrame:
    return extract_text_rules(pd.Series(list(texts)), cfg)


@pytest.mark.parametrize(
    ("text", "part"),
    [
        ("LIFTING BOX STRAINED LOWER BACK", "back"),
        ("CUT LEFT INDEX FINGER", "finger"),
        ("FOREIGN BODY RIGHT EYE", "eye"),
        ("BRUISED RIGHT FOREARM", "arm"),
    ],
)
def test_single_body_part(cfg, text, part):
    assert labels_for(cfg, text)[BODY_PART].iloc[0] == part


def test_two_body_parts_are_multiple(cfg):
    result = labels_for(cfg, "STRAIN NECK AND SHOULDER")
    assert result[BODY_PART].iloc[0] == cfg["text_rules"]["multiple_label"]
    assert result[NUM_BODY_PARTS].iloc[0] == 2


def test_multiple_keyword_is_multiple(cfg):
    result = labels_for(cfg, "MULTIPLE INJURIES")
    assert result[BODY_PART].iloc[0] == cfg["text_rules"]["multiple_label"]


def test_whole_word_matching(cfg):
    # FOREARM must not also count as ARM, and CUTTING is not the nature CUT.
    result = labels_for(cfg, "CUTTING TIMBER BRUISED FOREARM")
    assert result[NUM_BODY_PARTS].iloc[0] == 1
    assert result[INJURY_NATURE].iloc[0] == "contusion"


def test_nature_priority_prefers_more_severe(cfg):
    # Fracture is listed before laceration, so it wins when both are mentioned.
    result = labels_for(cfg, "LACERATION AND FRACTURE LEFT HAND")
    assert result[INJURY_NATURE].iloc[0] == "fracture"


def test_multi_word_keyword(cfg):
    assert labels_for(cfg, "FOREIGN BODY IN EYE")[INJURY_NATURE].iloc[0] == "foreign_body"


def test_cause_labels(cfg):
    result = labels_for(cfg, "SLIPPED ON WET FLOOR", "LIFTING CRATE", "HAND CAUGHT IN ROLLER")
    assert result[INJURY_CAUSE].tolist() == ["slip_fall", "lifting", "caught_in"]


def test_unmatched_text_is_other(cfg):
    other = cfg["text_rules"]["other_label"]
    result = labels_for(cfg, "WORK STRESS")
    assert result[BODY_PART].iloc[0] == other
    assert result[INJURY_NATURE].iloc[0] == other
    assert result[NUM_BODY_PARTS].iloc[0] == 0


def test_missing_text_is_handled(cfg):
    result = extract_text_rules(pd.Series([None, ""]), cfg)
    assert (result["description_length"] == 0).all()


def test_coverage_counts_non_other_share(cfg):
    other = cfg["text_rules"]["other_label"]
    result = labels_for(cfg, "STRAINED LOWER BACK", "WORK STRESS")
    assert coverage(result, other)[BODY_PART] == 0.5
