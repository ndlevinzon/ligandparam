"""Live multi-fragment progress board for parallel dihedral twist runs.

Thin wrappers around :mod:`ffpopt.progress_board` with fragment-oriented names.
Workers write stage updates into a shared JSON status file. The parent process
renders an ASCII board (and ``FRAG_STATUS.txt``) so interleaved wavefront spam
does not obscure which fragment is doing what. Detailed per-fragment output is
intended to live in ``<frag_dir>/frag-twist.log``.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

from .progress_board import JobBoardWatcher, JobProgressStore, format_job_board

PathLike = str | Path

# Ordered stages used for display hints (unknown stages still render).
KNOWN_STAGES = (
    "queued",
    "prepare",
    "hl_scan",
    "orig_scan",
    "compare",
    "fit",
    "apply",
    "rescan",
    "finished",
    "failed",
)


class FragmentProgressStore(JobProgressStore):
    """Process-shared fragment status table backed by a JSON file."""

    def __init__(self, path: PathLike) -> None:
        super().__init__(
            path,
            collection_key="fragments",
            id_header="Fragment",
            title="Fragment dihedral twist — live status",
            empty_hint="no fragments registered yet",
            detail_hint_label="Per-fragment detail logs",
        )

    def register(
        self,
        fragment_id: str,
        *,
        bonds: int = 0,
        frag_dir: str | None = None,
        log_path: str | None = None,
    ) -> None:
        """Mark a fragment as queued before workers start."""
        detail = f"{bonds} bond(s)"
        if frag_dir:
            detail = f"{detail} · {frag_dir}"
        super().register(
            fragment_id,
            status="queued",
            stage="queued",
            detail=detail,
            bonds=bonds,
            log_path=log_path,
        )

    def update(
        self,
        fragment_id: str,
        *,
        status: Optional[str] = None,
        stage: Optional[str] = None,
        detail: Optional[str] = None,
        error: Optional[str] = None,
        bonds: Optional[int] = None,
        log_path: Optional[str] = None,
    ) -> None:
        """Merge fields for one fragment and rewrite the JSON store."""
        super().update(
            fragment_id,
            status=status,
            stage=stage,
            detail=detail,
            error=error,
            bonds=bonds,
            log_path=log_path,
        )


def format_fragment_board(
    fragments: Mapping[str, Mapping[str, Any]],
    *,
    title: str = "Fragment dihedral twist — live status",
    log_root_hint: str | None = None,
) -> str:
    """Format ``{id: {status, stage, detail, ...}}`` as a fixed-width board."""
    return format_job_board(
        fragments,
        title=title,
        id_header="Fragment",
        empty_hint="no fragments registered yet",
        detail_hint_label="Per-fragment detail logs",
        log_root_hint=log_root_hint,
    )


@contextmanager
def fragment_stdio_to_file(log_path: Path) -> Iterator[None]:
    """Redirect ``stdout`` / ``stderr`` to ``log_path`` for this process.

    Wavefront and fit-apply still use ``print``; under parallel fragment pools
    those lines would otherwise interleave on the parent console.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_path, "a", encoding="utf-8", buffering=1)
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout = fh  # type: ignore[assignment]
        sys.stderr = fh  # type: ignore[assignment]
        yield
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
        try:
            fh.flush()
            fh.close()
        except OSError:
            pass


def make_fragment_file_logger(fragment_id: str, log_path: Path):
    """Logger that writes only to the fragment log (does not propagate)."""
    import logging

    name = f"ffpopt.workflows.frag.{fragment_id}"
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter(
            "{asctime} - {levelname} - {message}",
            style="{",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    return logger


class FragmentBoardWatcher(JobBoardWatcher):
    """Background thread that refreshes ``FRAG_STATUS.txt`` and logs on change."""

    def __init__(
        self,
        store: FragmentProgressStore,
        *,
        board_path: Path,
        logger,
        interval_sec: float = 5.0,
        log_root_hint: str | None = None,
    ) -> None:
        super().__init__(
            store,
            board_path=board_path,
            logger=logger,
            interval_sec=interval_sec,
            log_root_hint=log_root_hint,
            thread_name="frag-progress-board",
        )
