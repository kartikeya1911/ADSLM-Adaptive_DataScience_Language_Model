"""
app/utils/logger.py
===================
Centralised logging utility for the ADSLM system.
Every module imports this to get a consistent, timestamped logger.
"""

import logging
import sys
import io

# Force UTF-8 output on Windows (avoids UnicodeEncodeError with special chars)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger with console output.

    Args:
        name: Typically __name__ from the calling module.

    Returns:
        logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
