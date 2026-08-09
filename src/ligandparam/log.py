"""Logging helpers for the ligandparam package."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from . import __logging_name__


def get_logger() -> logging.Logger:
    """
    Get a logger with a null handler.

    Returns
    -------
    logging.Logger
        A logger instance with a null handler.
    """
    logger = logging.getLogger(__logging_name__)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


def _console_formatter(tag: str = "ligandparam") -> logging.Formatter:
    return logging.Formatter(
        f"{{asctime}} [{tag}] {{levelname}}: {{message}}",
        style="{",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


class _MaxLevelFilter(logging.Filter):
    def __init__(self, max_level: int) -> None:
        super().__init__()
        self.max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= self.max_level


def _attach_console_handlers(
    logger: logging.Logger,
    *,
    tag: str = "ligandparam",
    logging_level: int = logging.INFO,
) -> None:
    marker = f"ligandparam.console:{tag}"
    for handler in logger.handlers:
        if getattr(handler, "_lp_console_marker", None) == marker:
            return

    fmt = _console_formatter(tag)

    out = logging.StreamHandler(sys.stdout)
    out.setLevel(logging_level)
    out.addFilter(_MaxLevelFilter(logging.INFO))
    out.setFormatter(fmt)
    out._lp_console_marker = marker  # type: ignore[attr-defined]
    logger.addHandler(out)

    err = logging.StreamHandler(sys.stderr)
    err.setLevel(logging.WARNING)
    err.setFormatter(fmt)
    err._lp_console_marker = marker  # type: ignore[attr-defined]
    logger.addHandler(err)


def set_stream_logger(logging_level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger to output to stdout (INFO) and stderr (WARNING+).

    Parameters
    ----------
    logging_level : int, optional
        The logging level to set for the logger, by default logging.INFO.

    Returns
    -------
    logging.Logger
        A logger instance configured for console output with timestamps and a
        ``[ligandparam]`` tag.
    """
    logger = logging.getLogger(__logging_name__)
    logger.setLevel(logging_level)
    # Drop null-only setups so recipes get real console handlers.
    logger.handlers = [
        h
        for h in logger.handlers
        if not isinstance(h, logging.NullHandler)
    ]
    _attach_console_handlers(
        logger, tag="ligandparam", logging_level=logging_level
    )
    logger.propagate = False
    return logger


def set_file_logger(
    logfilename: Path,
    logname: str = None,
    filemode: str = "a",
    *,
    also_console: bool = True,
) -> logging.Logger:
    """
    Set up a logger to output to a file (and optionally the console).

    Parameters
    ----------
    logfilename : Path
        The path to the log file.
    logname : str, optional
        The name of the logger, by default None. If None, the module's logging name is used.
    filemode : str, optional
        The mode to open the log file, by default 'a'.
    also_console : bool, optional
        If True (default), also mirror INFO+ to stdout and WARNING+ to stderr
        so Slurm ``.out`` / ``.err`` capture the same messages.

    Returns
    -------
    logging.Logger
        A logger instance configured to output to the specified file.
    """
    if logname is None:
        logname = __logging_name__
    logger = logging.getLogger(logname)
    logger.setLevel(logging.INFO)
    logger.handlers = [
        h
        for h in logger.handlers
        if not isinstance(h, logging.NullHandler)
    ]
    formatter = logging.Formatter(
        "{asctime} [ligandparam] {levelname}: {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(filename=logfilename, mode=filemode)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    if also_console:
        _attach_console_handlers(logger, tag="ligandparam")
    logger.propagate = False
    return logger
