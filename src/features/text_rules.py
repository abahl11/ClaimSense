"""Rule-based keyword taxonomy for the free-text claim description.

Keyword lists live in ``config.yaml`` under ``text_rules``. Matching is on whole words, so
``ARM`` does not match ``FOREARM`` and ``CUT`` does not match ``CUTTING``. The rules are
fixed in advance and learn nothing from the data, so they are safe to apply before the
train/test split.
"""

from __future__ import annotations

import re

import pandas as pd

from src.config import Config

Patterns = dict[str, re.Pattern[str]]

BODY_PART = "body_part"
INJURY_NATURE = "injury_nature"
INJURY_CAUSE = "injury_cause"
NUM_BODY_PARTS = "num_body_parts_mentioned"
DESCRIPTION_LENGTH = "description_length"
TAXONOMY_FIELDS = (BODY_PART, INJURY_NATURE, INJURY_CAUSE)


def compile_keywords(keywords: list[str]) -> re.Pattern[str]:
    """Compile keywords into one whole-word regex (longer phrases tried first)."""
    ordered = sorted(keywords, key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(re.escape(k) for k in ordered) + r")\b")


def compile_taxonomy(categories: dict[str, list[str]]) -> Patterns:
    """Compile one regex per category, keeping the config order (the priority order)."""
    return {label: compile_keywords(words) for label, words in categories.items()}


def keyword_matches(text: pd.Series, patterns: Patterns) -> pd.DataFrame:
    """Boolean table: one column per category, True where the text mentions it."""
    return pd.DataFrame(
        {label: text.str.contains(pattern) for label, pattern in patterns.items()},
        index=text.index,
    ).astype(bool)


def label_by_priority(matches: pd.DataFrame, other_label: str) -> pd.Series:
    """Pick the first matching category in column (priority) order, else ``other_label``."""
    first = matches.idxmax(axis=1)
    return first.where(matches.any(axis=1), other_label)


def apply_part_hierarchy(matches: pd.DataFrame, parents: dict[str, str]) -> pd.DataFrame:
    """Drop a parent part where its child is also mentioned ("HAND ... FINGER" -> finger)."""
    out = matches.copy()
    for child, parent in parents.items():
        out[parent] &= ~out[child]
    return out


def label_body_part(text: pd.Series, cfg: Config) -> tuple[pd.Series, pd.Series]:
    """Label the body part and count how many distinct body parts are mentioned.

    Args:
        text: Normalised claim descriptions.
        cfg: Loaded project configuration.

    Returns:
        The body-part label (a part, ``multiple`` or ``other``) and the count of parts,
        after the child/parent hierarchy in ``text_rules.body_part_parents`` is applied.
    """
    rules = cfg["text_rules"]
    raw_matches = keyword_matches(text, compile_taxonomy(rules[BODY_PART]))
    matches = apply_part_hierarchy(raw_matches, rules["body_part_parents"])
    count = matches.sum(axis=1)
    says_multiple = text.str.contains(compile_keywords(rules["multiple_keywords"]))
    label = label_by_priority(matches, rules["other_label"])
    label = label.mask((count > 1) | says_multiple, rules["multiple_label"])
    return label, count


def label_category(text: pd.Series, cfg: Config, field: str) -> pd.Series:
    """Label a priority-ordered field such as ``injury_nature`` or ``injury_cause``."""
    rules = cfg["text_rules"]
    matches = keyword_matches(text, compile_taxonomy(rules[field]))
    return label_by_priority(matches, rules["other_label"])


def extract_text_rules(text: pd.Series, cfg: Config) -> pd.DataFrame:
    """Derive every rule-based text feature from the claim description.

    Args:
        text: Normalised claim descriptions (output of the cleaning stage).
        cfg: Loaded project configuration.

    Returns:
        A table with body part, injury nature, injury cause, number of body parts
        mentioned and description length, indexed like ``text``.
    """
    text = text.fillna("")
    body_part, count = label_body_part(text, cfg)
    return pd.DataFrame(
        {
            BODY_PART: body_part,
            INJURY_NATURE: label_category(text, cfg, INJURY_NATURE),
            INJURY_CAUSE: label_category(text, cfg, INJURY_CAUSE),
            NUM_BODY_PARTS: count,
            DESCRIPTION_LENGTH: text.str.len(),
        },
        index=text.index,
    )


def coverage(labels: pd.DataFrame, other_label: str) -> pd.Series:
    """Share of rows labelled with something other than ``other_label``, per field."""
    return pd.Series({field: (labels[field] != other_label).mean() for field in TAXONOMY_FIELDS})
