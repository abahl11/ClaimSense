"""Tests for report helpers: README block replacement and number formatting."""

import pytest

from src.reporting.build_results import money, pct, relative_change
from src.reporting.documents import replace_block


def test_replace_block_keeps_markers_and_surrounding_text():
    text = "intro\n<!-- BEGIN GENERATED: x -->\nold\n<!-- END GENERATED: x -->\noutro"
    updated = replace_block(text, "x", "new")
    assert updated.startswith("intro\n<!-- BEGIN GENERATED: x -->")
    assert "new" in updated and "old" not in updated
    assert updated.endswith("<!-- END GENERATED: x -->\noutro")


def test_replace_block_is_idempotent():
    text = "<!-- BEGIN GENERATED: x --><!-- END GENERATED: x -->"
    once = replace_block(text, "x", "content")
    assert replace_block(once, "x", "content") == once


def test_replace_block_requires_markers():
    with pytest.raises(ValueError, match="no generated block"):
        replace_block("no markers here", "x", "content")


def test_formatting_helpers():
    assert money(1234567.8) == "1,234,568"
    assert pct(0.1234) == "12.3%"
    assert relative_change(90, 100) == pytest.approx(-0.1)
