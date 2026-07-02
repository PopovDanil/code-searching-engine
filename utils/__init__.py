"""Shared utilities."""

import logging
import sys
from typing import Optional


def setup_logging(level: Optional[str] = None) -> None:
    """Configure root logger with a clean format.

    Parameters
    ----------
    level:
        Logging level string (``"DEBUG"``, ``"INFO"``, …).  Defaults to
        ``"INFO"``.
    """
    lvl = getattr(logging, (level or "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s │ %(name)-30s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
