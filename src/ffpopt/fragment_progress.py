"""Live multi-fragment progress board for parallel dihedral twist runs.

Workers write stage updates into a shared JSON status file. The parent process
renders an ASCII board (and ``FRAG_STATUS.txt``) so interleaved wavefront spam
does not obscure which fragment is doing what. Detailed per-fragment output is
intended to live in ``<frag_dir>/frag-twist.log``.
"""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

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


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


@contextmanager
def _dir_lock(lock_dir: Path, *, timeout_sec: float = 30.0) -> Iterator[None]:
    """Exclusive lock via ``mkdir`` (works on NFS / Windows without extras)."""
    deadline = time.monotonic() + timeout_sec
    while True:
        try:
            os.mkdir(lock_dir)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                # Stale lock: take over so a crashed worker cannot freeze the board.
                try:
                    os.rmdir(lock_dir)
                except OSError:
                    pass
                continue
            time.sleep(0.05)
    try:
        yield
    finally:
        try:
            os.rmdir(lock_dir)
        except OSError:
            pass


class FragmentProgressStore:
    """Process-shared fragment status table backed by a JSON file."""

    def __init__(self, path: PathLike) -> None:
        self.path = Path(path)
        self.lock_dir = self.path.with_suffix(self.path.suffix + ".lock")

    def _read_unlocked(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"fragments": {}}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"fragments": {}}
        if not isinstance(data, dict):
            return {"fragments": {}}
        frags = data.get("fragments")
        if not isinstance(frags, dict):
            data["fragments"] = {}
        return data

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """Return ``{fragment_id: status_dict}``."""
        with _dir_lock(self.lock_dir):
            data = self._read_unlocked()
        return dict(data.get("fragments") or {})

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
        self.update(
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
        with _dir_lock(self.lock_dir):
            data = self._read_unlocked()
            frags = data.setdefault("fragments", {})
            entry = dict(frags.get(fragment_id) or {})
            entry["fragment_id"] = fragment_id
            entry["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
            if status is not None:
                entry["status"] = str(status)
            if stage is not None:
                entry["stage"] = str(stage)
            if detail is not None:
                entry["detail"] = str(detail)
            if error is not None:
                entry["error"] = str(error)
            if bonds is not None:
                entry["bonds"] = int(bonds)
            if log_path is not None:
                entry["log_path"] = str(log_path)
            frags[fragment_id] = entry
            _atomic_write_text(self.path, json.dumps(data, indent=2) + "\n")

    def render_board(
        self,
        *,
        title: str = "Fragment dihedral twist — live status",
        log_root_hint: str | None = None,
    ) -> str:
        """Return an ASCII status table for the current snapshot."""
        return format_fragment_board(
            self.snapshot(), title=title, log_root_hint=log_root_hint
        )


def format_fragment_board(
    fragments: Mapping[str, Mapping[str, Any]],
    *,
    title: str = "Fragment dihedral twist — live status",
    log_root_hint: str | None = None,
) -> str:
    """Format ``{id: {status, stage, detail, ...}}`` as a fixed-width board."""
    ids = sorted(fragments.keys())
    col_id = max([len("Fragment")] + [len(i) for i in ids] + [10])
    col_st = max(
        [len("Status")]
        + [len(str(fragments[i].get("status", ""))) for i in ids]
        + [8]
    )
    col_sg = max(
        [len("Stage")]
        + [len(str(fragments[i].get("stage", ""))) for i in ids]
        + [8]
    )
    col_dt = 44

    def row(fid: str, status: str, stage: str, detail: str) -> str:
        d = (detail or "")[: col_dt - 1]
        return (
            f" {fid:<{col_id}}  {status:<{col_st}}  {stage:<{col_sg}}  {d:<{col_dt}}"
        )

    width = col_id + col_st + col_sg + col_dt + 8
    bar = "=" * width
    sep = "-" * width
    lines = [
        bar,
        f" {title}",
        bar,
        row("Fragment", "Status", "Stage", "Detail"),
        sep,
    ]

    counts = {"done": 0, "running": 0, "queued": 0, "failed": 0, "other": 0}
    for fid in ids:
        e = fragments[fid]
        status = str(e.get("status") or "?")
        stage = str(e.get("stage") or "—")
        detail = str(e.get("detail") or "")
        if e.get("error") and status == "failed":
            detail = (detail + " | " if detail else "") + str(e["error"])[:40]
        lines.append(row(fid, status, stage, detail))
        key = status if status in counts else "other"
        counts[key] = counts.get(key, 0) + 1

    if not ids:
        lines.append(row("(none)", "—", "—", "no fragments registered yet"))

    lines.append(sep)
    lines.append(
        f" {counts.get('done', 0)} done · {counts.get('running', 0)} running · "
        f"{counts.get('queued', 0)} queued · {counts.get('failed', 0)} failed"
        + (f" · {counts.get('other', 0)} other" if counts.get("other") else "")
    )
    if log_root_hint:
        lines.append(f" Per-fragment detail logs: {log_root_hint}")
    lines.append(bar)
    return "\n".join(lines) + "\n"


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


class FragmentBoardWatcher:
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
        import threading

        self.store = store
        self.board_path = Path(board_path)
        self.logger = logger
        self.interval_sec = float(interval_sec)
        self.log_root_hint = log_root_hint
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, name="frag-progress-board", daemon=True
        )
        self._last = ""

    def start(self) -> None:
        self._emit(force_log=True)
        self._thread.start()

    def stop(self, *, final: bool = True) -> None:
        self._stop.set()
        self._thread.join(timeout=max(1.0, self.interval_sec))
        if final:
            self._emit(force_log=True)

    def _emit(self, *, force_log: bool = False) -> None:
        text = self.store.render_board(log_root_hint=self.log_root_hint)
        try:
            _atomic_write_text(self.board_path, text)
        except OSError:
            pass
        if force_log or text != self._last:
            self.logger.info("\n%s", text.rstrip("\n"))
            self._last = text

    def _loop(self) -> None:
        while not self._stop.wait(timeout=self.interval_sec):
            self._emit(force_log=False)
