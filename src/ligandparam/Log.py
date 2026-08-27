"""Logging helpers for the ligandparam package."""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path
from typing import Optional

from . import __logging_name__

_QUOTES_FILE = Path(__file__).resolve().parent / "pkgdata" / "quotes.txt"


def default_quotes_path() -> Path:
    """Packaged quotes file (one quote per line)."""
    return _QUOTES_FILE


def load_quotes(path: Path | None = None) -> list[str]:
    """Read non-empty, non-comment lines from the quotes file."""
    quotes_path = Path(path) if path is not None else default_quotes_path()
    if not quotes_path.is_file():
        return []
    out: list[str] = []
    try:
        text = quotes_path.read_text(encoding="utf-8")
    except OSError:
        return []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if len(line) >= 2 and line[0] == line[-1] and line[0] in {'"', "'"}:
            line = line[1:-1].strip()
        if line:
            out.append(line)
    return out


def format_reminder_line(quote: str) -> str:
    """Return the success reminder line for one quote.

    The quotes file already wraps spoken text in ``"..."``; this does not
    add a second pair around the whole line.
    """
    from ffpopt.runtime.Console import ascii_for_stdio

    return ascii_for_stdio(f"LIGANDPARAM reminds you: {quote}")


def log_success_quote(
    logger: logging.Logger | None = None,
    *,
    quotes_path: Path | None = None,
) -> Optional[str]:
    """Pick a random quote and print it with the console log prefix.

    Writes one stdout line::

        YYYY-mm-dd HH:MM:SS [ligandparam] INFO: LIGANDPARAM reminds you: ...

    ``logger`` is accepted for call-site compatibility but is not used to
    emit a second copy. No-op when the quotes file is missing or empty.
    """
    quotes = load_quotes(quotes_path)
    if not quotes:
        return None
    from ffpopt.runtime.Console import format_console_line

    _ = logger
    quote = random.choice(quotes)
    line = format_reminder_line(quote)
    print(
        format_console_line(f"INFO: {line}", tag="ligandparam"),
        end="",
        file=sys.stdout,
        flush=True,
    )
    return quote


def dihed_correct_ok(result, *, dry_run: bool = False) -> bool:
    """True when a dihedral-correct result wrote a frcmod with no failed jobs."""
    if dry_run or not isinstance(result, dict):
        return False
    out = result.get("merged_frcmod") or result.get("out_frcmod")
    if not out or not Path(out).is_file():
        return False
    for frag in result.get("fragments") or []:
        if not isinstance(frag, dict):
            continue
        status = str(frag.get("status") or "").lower()
        if status == "failed" or frag.get("error"):
            return False
    report = result.get("merge_report") or {}
    if isinstance(report, dict) and report.get("errors"):
        return False
    return True


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
