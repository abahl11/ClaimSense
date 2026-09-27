"""Configure the ``logging`` module for every pipeline run."""

from __future__ import annotations

import logging
import sys

from src.config import Config


def setup_logging(cfg: Config, level: str | None = None) -> None:
    """Send log records to the console and to the log file named in the config.

    Args:
        cfg: Loaded project configuration.
        level: Optional override for the level set in ``cfg["logging"]["level"]``.
    """
    log_cfg = cfg["logging"]
    log_file = cfg.path("log_file")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
    logging.basicConfig(
        level=(level or log_cfg["level"]).upper(),
        format=log_cfg["format"],
        datefmt=log_cfg["datefmt"],
        handlers=handlers,
        force=True,
    )
    for name in log_cfg.get("quiet_loggers", []):
        logging.getLogger(name).setLevel(logging.WARNING)
