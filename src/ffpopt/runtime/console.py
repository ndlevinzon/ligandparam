"""Console formatting helpers for Slurm-friendly stdout/stderr logging.

Format (single timestamp, hierarchical brackets)::

    YYYY-mm-dd HH:MM:SS [ffpopt:fragment_10] [frag-twist] INFO: message

Leading ``[scope]`` tokens in the message body are peeled into the bracket
hierarchy so callers can keep writing ``log.info("[frag-twist] ...")`` without
duplicating tags. Console handlers always write to the real process streams
(``sys.__stdout__`` / ``sys.__stderr__``) so teeing stdout for fragment logs
does not double-prefix already-formatted logger lines.
"""

from __future__ import annotations

import logging
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence, TextIO

# Lines that already carry our console prefix (logger → TeeTextIO).
_PREFIXED_LINE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \[")
# Leading ``[scope]`` tokens embedded in the message / print body.
_LEADING_SCOPE = re.compile(r"^\[([^\]]+)\]\s*")


def console_timestamp() -> str:
    """Return a local wall-clock timestamp suitable for console lines."""
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _normalize_tags(tag: str | Sequence[str] | None = None, *extra: str) -> list[str]:
    tags: list[str] = []
    if tag is None:
        pass
    elif isinstance(tag, str):
        if tag:
            tags.append(tag)
    else:
        tags.extend(str(t) for t in tag if t)
    tags.extend(str(t) for t in extra if t)
    # Preserve order, drop duplicates.
    seen: set[str] = set()
    out: list[str] = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def peel_leading_scopes(message: str) -> tuple[list[str], str]:
    """Split leading ``[scope]`` tokens from ``message``."""
    scopes: list[str] = []
    rest = message
    while True:
        match = _LEADING_SCOPE.match(rest)
        if not match:
            break
        scopes.append(match.group(1))
        rest = rest[match.end() :]
    return scopes, rest


def format_bracket_tags(*tags: str) -> str:
    """Return ``[a] [b]`` (no trailing space); empty if no tags."""
    return " ".join(f"[{t}]" for t in tags if t)


def format_console_line(
    message: str,
    *,
    tag: str | Sequence[str] | None = None,
    tags: Sequence[str] | None = None,
) -> str:
    """Prefix ``message`` once: ``TIMESTAMP [tag…] [peeled…] rest``."""
    if _PREFIXED_LINE.match(message.lstrip("\r")):
        # Already formatted (e.g. logging handler wrote into a TeeTextIO).
        return message if message.endswith("\n") else f"{message}\n"

    base = _normalize_tags(tag)
    if tags:
        base = _normalize_tags(base + list(tags))

    stamp = console_timestamp()
    text = message if message.endswith("\n") else f"{message}\n"
    if text == "\n":
        bracket = format_bracket_tags(*base)
        prefix = f"{stamp} {bracket} " if bracket else f"{stamp} "
        return prefix + "\n"

    parts = text.splitlines(keepends=True)
    out: list[str] = []
    for part in parts:
        newline = part.endswith("\n")
        body = part[:-1] if newline else part
        scopes, rest = peel_leading_scopes(body)
        bracket = format_bracket_tags(*_normalize_tags(base + scopes))
        prefix = f"{stamp} {bracket} " if bracket else f"{stamp} "
        line = prefix + rest
        out.append(line + ("\n" if newline else "\n"))
    return "".join(out)


class TeeTextIO:
    """Write plain text to a log file and mirror tagged lines to a console stream."""

    def __init__(
        self,
        file_stream: TextIO,
        console_stream: TextIO,
        *,
        tag: str | Sequence[str],
    ) -> None:
        self.file_stream = file_stream
        self.console_stream = console_stream
        self.tags = _normalize_tags(tag)
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
                format_console_line(line + "\n", tags=self.tags)
            )
        return len(data)

    def flush(self) -> None:
        if self._buf:
            self.console_stream.write(
                format_console_line(self._buf, tags=self.tags)
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


class HierarchicalConsoleFormatter(logging.Formatter):
    """``TIMESTAMP [base…] [peeled…] LEVEL: message`` with one timestamp."""

    def __init__(self, *tags: str, datefmt: str = "%Y-%m-%d %H:%M:%S") -> None:
        super().__init__(datefmt=datefmt)
        self.base_tags = _normalize_tags(tags)

    def format(self, record: logging.LogRecord) -> str:
        scopes, rest = peel_leading_scopes(record.getMessage())
        bracket = format_bracket_tags(*_normalize_tags(self.base_tags + scopes))
        asctime = self.formatTime(record, self.datefmt)
        level = record.levelname
        if bracket:
            head = f"{asctime} {bracket} {level}: {rest}"
        else:
            head = f"{asctime} {level}: {rest}"
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            if head[-1:] != "\n":
                head += "\n"
            head += record.exc_text
        return head


def console_formatter(
    tag: str | Sequence[str] | None = None,
    *extra_tags: str,
) -> logging.Formatter:
    """Formatter: ``YYYY-mm-dd HH:MM:SS [tag…] LEVEL: message``."""
    tags = _normalize_tags(tag, *extra_tags)
    return HierarchicalConsoleFormatter(*tags)


def attach_console_handlers(
    logger: logging.Logger,
    *,
    tag: str | Sequence[str] = "ffpopt",
    level: int = logging.INFO,
) -> None:
    """Attach stdout (INFO) and stderr (WARNING+) handlers if not already present.

    Writes to ``sys.__stdout__`` / ``sys.__stderr__`` so temporary stdio tees
    used for per-fragment files cannot double-prefix logger output.
    """
    for handler in logger.handlers:
        if getattr(handler, "_lp_console_marker", None):
            return

    tags = _normalize_tags(tag)
    marker = "ffpopt.console:" + "/".join(tags)
    fmt = console_formatter(tags)

    out = logging.StreamHandler(sys.__stdout__)
    out.setLevel(level)
    out.addFilter(_MaxLevelFilter(logging.INFO))
    out.setFormatter(fmt)
    out._lp_console_marker = marker  # type: ignore[attr-defined]
    logger.addHandler(out)

    err = logging.StreamHandler(sys.__stderr__)
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
    tag: str | Sequence[str] = "ffpopt",
) -> Iterator[None]:
    """Write stdout/stderr to ``log_path`` and mirror tagged lines to the console.

    Useful for wavefront ``print`` spam under parallel fragment pools: the
    per-fragment ``.log`` stays complete, and Slurm ``.out`` / ``.err`` still
    receive the same content with a single timestamp and hierarchical tags.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_path, "a", encoding="utf-8", buffering=1)
    old_out, old_err = sys.stdout, sys.stderr
    # Mirror to the streams we are replacing (real console, or a test capture).
    # Logger console handlers use sys.__stdout__/__stderr__, so they are not
    # double-prefixed by this tee.
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
