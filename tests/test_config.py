"""Tests for the configuration loader."""

import pytest

from src.config import PROJECT_ROOT, load_config


def test_config_loads_with_integer_seed():
    cfg = load_config()
    assert isinstance(cfg.random_seed, int)


def test_paths_resolve_inside_project():
    cfg = load_config()
    for key in cfg["paths"]:
        assert cfg.path(key).is_relative_to(PROJECT_ROOT)


def test_key_columns_are_in_expected_schema():
    data = load_config()["data"]
    for key in ("id_column", "target", "initial_estimate", "text_column"):
        assert data[key] in data["expected_columns"]


def test_missing_section_raises(tmp_path):
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text("random_seed: 1\n", encoding="utf-8")
    with pytest.raises(KeyError, match="missing required sections"):
        load_config(bad_config)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does_not_exist.yaml")
