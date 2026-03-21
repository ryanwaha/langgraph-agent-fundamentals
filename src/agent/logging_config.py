"""Logging configuration for the agent project.

Call setup_logging() once at process startup (e.g. in telegram_bot.main()).
Writes to logs/bot.log with rotation; terminal output is suppressed.

Third-party libraries are capped at WARNING to avoid noise.
"""

import logging
import logging.handlers
from pathlib import Path

_LOG_DIR = Path("logs")
_LOG_FILE = _LOG_DIR / "bot.log"

# Noisy third-party loggers to quiet down
_QUIET_LOGGERS = [
    "httpx",
    "httpcore",
    "telegram",
    "apscheduler",
    "google",
    "langchain",
    "langgraph",
]


def setup_logging(level: int = logging.INFO) -> None:
    """Configure file-only logging for the bot process.

    Args:
        level: Root log level for our code. Defaults to INFO.
    """
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    handler = logging.handlers.RotatingFileHandler(
        _LOG_FILE,
        maxBytes=5 * 1024 * 1024,  # 5 MB per file
        backupCount=3,              # keep bot.log, bot.log.1, bot.log.2, bot.log.3
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    # Terminal: WARNING+ only (keeps terminal clean while surfacing real problems)
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    console.setFormatter(logging.Formatter(
        fmt="%(levelname)-8s %(name)s: %(message)s",
    ))

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)
    root.addHandler(console)

    # Suppress noisy third-party output
    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
