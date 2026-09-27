"""Load and expose the project configuration stored in ``config/config.yaml``.

All tunable values (paths, seed, split settings, hyperparameters, keyword
dictionaries) live in the YAML file. Modules receive a :class:`Config` object
instead of hardcoding values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config" / "config.yaml"
REQUIRED_SECTIONS: tuple[str, ...] = ("random_seed", "paths", "logging", "data", "split")


@dataclass(frozen=True, repr=False)
class Config:
    """Read-only view of the parsed YAML configuration.

    Attributes:
        settings: The parsed YAML as a nested dictionary.
        root: Directory that relative paths in ``settings["paths"]`` are resolved against.
    """

    settings: dict[str, Any]
    root: Path

    def __repr__(self) -> str:
        """Short form, so pipelines that hold a Config print readably."""
        return f"Config(root={self.root})"

    def __getitem__(self, key: str) -> Any:
        """Return a top-level config section, for example ``cfg["data"]``."""
        return self.settings[key]

    @property
    def random_seed(self) -> int:
        """Return the single random seed shared by every stochastic component."""
        return int(self.settings["random_seed"])

    def path(self, key: str) -> Path:
        """Return the absolute path for an entry in the ``paths`` section.

        Args:
            key: Name of the entry, for example ``"raw_data"``.

        Returns:
            The entry resolved against the project root.
        """
        return self.root / self.settings["paths"][key]

    def col(self, name: str) -> str:
        """Return the source column name for a logical name, e.g. ``"weekly_wages"``."""
        return self.settings["data"]["columns"][name]

    def cols(self, names: list[str]) -> list[str]:
        """Return the source column names for a list of logical names."""
        return [self.col(name) for name in names]

    def resolve(self, names: list[str]) -> list[str]:
        """Map logical source-column names to real names; engineered names pass through."""
        columns = self.settings["data"]["columns"]
        return [columns.get(name, name) for name in names]

    @property
    def expected_columns(self) -> list[str]:
        """Every source column the project expects in the raw file."""
        return list(self.settings["data"]["columns"].values())


def load_config(config_path: Path | None = None, root: Path = PROJECT_ROOT) -> Config:
    """Parse the YAML config file and check that required sections exist.

    Args:
        config_path: YAML file to read. Defaults to ``config/config.yaml``.
        root: Directory used to resolve relative paths. Defaults to the project root.

    Returns:
        The loaded configuration.

    Raises:
        FileNotFoundError: If the config file does not exist.
        KeyError: If a required top-level section is missing.
    """
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        settings = yaml.safe_load(handle) or {}
    missing = [section for section in REQUIRED_SECTIONS if section not in settings]
    if missing:
        raise KeyError(f"Config file {path} is missing required sections: {missing}")
    return Config(settings=settings, root=Path(root))
