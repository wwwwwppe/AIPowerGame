from __future__ import annotations

import logging


def setup_logging(level: int = logging.INFO) -> None:
    if logging.getLogger().handlers:
        return

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
