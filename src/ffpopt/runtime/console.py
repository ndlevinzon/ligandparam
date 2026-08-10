"""Console formatting helpers for Slurm-friendly stdout/stderr logging.

Prefixes console lines with a timestamp and a package/command tag so parallel
workers remain readable in ``.out`` / ``.err`` while still writing detail logs
to per-job files.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO


def console_timestamp() -> str:
    """Return a local wall-clock timestamp suitable for console lines."""
    return time.strftime("%Y-%m-%d %H:%M:%S")


def format_console_line(message: str, *, tag: str) -> str:
    """Prefix ``message`` with ``YYYY-mm-dd HH:MM:SS [tag]`` (per line)."""
    stamp = console_timestamp()
    prefix = f"{stamp} [{tag}] "
    text = message if message.endswith("\n") else f"{message}\n"
    if text == "\n":
        return prefix + "\n"
    parts = text.splitlines(keepends=True)
    return "".join(
        prefix + (part if part.endswith("\n") else part + "\n") for part in parts
    )


class TeeTextIO:
    """Write plain text to a log file and mirror tagged lines to a console stream."""

    def __init__(
        self,
        file_stream: TextIO,
        console_stream: TextIO,
        *,
        tag: str,
    ) -> None:
        self.file_stream = file_stream
        self.console_stream = console_stream
        self.tag = tag
        self.encoding = getattr(file_stream, "encoding", "utf-8") or "utf-8"
        self._buf = ""

    def write(self, data: str) -> int:
        if not data:
            return 0
        self.file_stream.write(data)
        self._buf += data
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self.console_stream.write(
                format_console_line(line + "\n", tag=self.tag)
            )
        return len(data)

    def flush(self) -> None:
        if self._buf:
            self.console_stream.write(
                format_console_line(self._buf, tag=self.tag)
            )
            self._buf = ""
        try:
            self.file_stream.flush()
        except OSError:
            pass
        try:
            self.console_stream.flush()
        except OSError:
            pass

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        return self.file_stream.fileno()

    def writable(self) -> bool:
        return True

    def close(self) -> None:
        self.flush()


class _MaxLevelFilter(logging.Filter):
    """Allow records with ``levelno <= max_level``."""

    def __init__(self, max_level: int) -> None:
        super().__init__()
        self.max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= self.max_level


def console_formatter(tag: str) -> logging.Formatter:
    """Formatter: ``YYYY-mm-dd HH:MM:SS [tag] LEVEL: message``."""
    return logging.Formatter(
        f"{{asctime}} [{tag}] {{levelname}}: {{message}}",
        style="{",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def attach_console_handlers(
    logger: logging.Logger,
    *,
    tag: str,
    level: int = logging.INFO,
) -> None:
    """Attach stdout (INFO) and stderr (WARNING+) handlers if not already present."""
    marker = f"ligandparam.console:{tag}"
    for handler in logger.handlers:
        if getattr(handler, "_lp_console_marker", None) == marker:
            return

    fmt = console_formatter(tag)

    out = logging.StreamHandler(sys.stdout)
    out.setLevel(level)
    out.addFilter(_MaxLevelFilter(logging.INFO))
    out.setFormatter(fmt)
    out._lp_console_marker = marker  # type: ignore[attr-defined]
    logger.addHandler(out)

    err = logging.StreamHandler(sys.stderr)
    err.setLevel(logging.WARNING)
    err.setFormatter(fmt)
    err._lp_console_marker = marker  # type: ignore[attr-defined]
    logger.addHandler(err)

    logger.setLevel(min(logger.level or level, level))
    # Avoid duplicate lines if a parent also has console handlers.
    logger.propagate = False


@contextmanager
def tee_stdio_to_file(
    log_path: Path,
    *,
    tag: str = "ffpopt",
) -> Iterator[None]:
    """Write stdout/stderr to ``log_path`` and mirror tagged lines to the console.

    Useful for wavefront ``print`` spam under parallel fragment pools: the
    per-fragment ``.log`` stays complete, and Slurm ``.out`` / ``.err`` still
    receive the same content with timestamps and a command tag.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_path, "a", encoding="utf-8", buffering=1)
    old_out, old_err = sys.stdout, sys.stderr
    tee_out = TeeTextIO(fh, old_out, tag=tag)
    tee_err = TeeTextIO(fh, old_err, tag=tag)
    try:
        sys.stdout = tee_out  # type: ignore[assignment]
        sys.stderr = tee_err  # type: ignore[assignment]
        yield
    finally:
        try:
            tee_out.flush()
            tee_err.flush()
        except OSError:
            pass
        sys.stdout = old_out
        sys.stderr = old_err
        try:
            fh.close()
        except OSError:
            pass
