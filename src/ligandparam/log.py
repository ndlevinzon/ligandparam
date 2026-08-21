"""Logging helpers for the ligandparam package."""

from __future__ import annotations

import logging
from pathlib import Path

from . import __logging_name__


def get_logger() -> logging.Logger:
    """Return the package logger (null handler until configured)."""
    logger = logging.getLogger(__logging_name__)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


def set_stream_logger(logging_level: int = logging.INFO) -> logging.Logger:
    """Attach stdout (INFO) / stderr (WARNING+) handlers via ``ffpopt.runtime.Console``."""
    from ffpopt.runtime.Console import attach_console_handlers

    logger = logging.getLogger(__logging_name__)
    logger.setLevel(logging_level)
    logger.handlers = [
        h for h in logger.handlers if not isinstance(h, logging.NullHandler)
    ]
    attach_console_handlers(logger, tag="ligandparam", level=logging_level)
    return logger


def set_file_logger(
    logfilename: Path,
    logname: str = None,
    filemode: str = "a",
    *,
    also_console: bool = True,
) -> logging.Logger:
    """Log to a file; optionally mirror to the console (Slurm-friendly)."""
    from ffpopt.runtime.Console import attach_console_handlers, console_formatter

    if logname is None:
        logname = __logging_name__
    logger = logging.getLogger(logname)
    logger.setLevel(logging.INFO)
    logger.handlers = [
        h for h in logger.handlers if not isinstance(h, logging.NullHandler)
    ]
    file_handler = logging.FileHandler(filename=logfilename, mode=filemode)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(console_formatter("ligandparam"))
    logger.addHandler(file_handler)
    if also_console:
        attach_console_handlers(logger, tag="ligandparam")
    return logger
