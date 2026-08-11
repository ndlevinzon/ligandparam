"""Shared CPU lease table for parallel fragment (and similar) workers.

Workers lease a fair share of a total core budget from a JSON file protected by
a mkdir lock. Finished workers release their lease so remaining work can grow
its ``nproc`` at the next lease boundary (fragment start or twist phase).

On each :meth:`CpuBudget.lease` call, leases for the active owner set are
recomputed with fair share + leftover distribution so cores are not left idle
among currently active workers. In-flight wavefront pools do not resize; the
new size applies at the next scan phase.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterable, Optional

PathLike = str | Path


def _atomic_write_text(path: Path, text: str) -> None:
    from .progress_board import atomic_write_text

    atomic_write_text(path, text)


class _DirLock:
    """Exclusive lock via ``mkdir`` (works on NFS / Windows without extras)."""

    def __init__(self, lock_dir: Path, *, timeout_sec: float = 30.0) -> None:
        from .progress_board import DirLock

        self._lock = DirLock(lock_dir, timeout_sec=timeout_sec)

    def __enter__(self) -> None:
        return self._lock.__enter__()

    def __exit__(self, *exc: object) -> None:
        return self._lock.__exit__(*exc)


def fair_share_leases(
    total: int,
    owners: Iterable[str],
    *,
    prefer: Optional[str] = None,
) -> dict[str, int]:
    """Distribute ``total`` cores across ``owners`` (floor 1 each when possible).

    Base share is ``total // n``; leftover cores are handed out one-by-one in
    sorted owner order, with ``prefer`` receiving the first leftover when set.
    """
    total = max(1, int(total))
    ids = sorted({str(o) for o in owners if o is not None and str(o)})
    if not ids:
        return {}
    n = len(ids)
    if total < n:
        out = {oid: 0 for oid in ids}
        ordered = list(ids)
        if prefer and prefer in out:
            ordered = [prefer] + [o for o in ordered if o != prefer]
        for oid in ordered[:total]:
            out[oid] = 1
        return {k: v for k, v in out.items() if v > 0}

    base = total // n
    rem = total % n
    out = {oid: base for oid in ids}
    ordered = list(ids)
    if prefer and prefer in out:
        ordered = [prefer] + [o for o in ordered if o != prefer]
    for oid in ordered[:rem]:
        out[oid] += 1
    return out


class CpuBudget:
    """Process-shared CPU lease table backed by a JSON file."""

    def __init__(
        self, path: PathLike, total: int, *, clear_leases: bool = False
    ) -> None:
        self.path = Path(path)
        self.total = max(1, int(total))
        self.lock_dir = self.path.with_suffix(self.path.suffix + ".lock")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _DirLock(self.lock_dir):
            data = self._read_unlocked()
            data["total"] = self.total
            if clear_leases:
                data["leases"] = {}
            else:
                data.setdefault("leases", {})
            data["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
            _atomic_write_text(self.path, json.dumps(data, indent=2) + "\n")

    def _read_unlocked(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"total": self.total, "leases": {}}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"total": self.total, "leases": {}}
        if not isinstance(data, dict):
            return {"total": self.total, "leases": {}}
        leases = data.get("leases")
        if not isinstance(leases, dict):
            data["leases"] = {}
        data["total"] = int(data.get("total") or self.total)
        return data

    def clear_leases(self) -> None:
        """Drop every lease (parent restart after kill / walltime timeout)."""
        with _DirLock(self.lock_dir):
            data = self._read_unlocked()
            data["leases"] = {}
            data["total"] = self.total
            data["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
            _atomic_write_text(self.path, json.dumps(data, indent=2) + "\n")

    def snapshot(self) -> dict[str, Any]:
        """Return ``{total, leases, used, free}``."""
        with _DirLock(self.lock_dir):
            data = self._read_unlocked()
        leases = {
            str(k): int(v)
            for k, v in (data.get("leases") or {}).items()
            if int(v) > 0
        }
        used = sum(leases.values())
        total = int(data.get("total") or self.total)
        return {
            "total": total,
            "leases": leases,
            "used": used,
            "free": max(0, total - used),
        }

    def release(self, owner_id: str) -> None:
        """Drop ``owner_id``'s lease (no-op if absent)."""
        owner_id = str(owner_id)
        with _DirLock(self.lock_dir):
            data = self._read_unlocked()
            leases = data.setdefault("leases", {})
            if owner_id in leases:
                del leases[owner_id]
                data["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
                _atomic_write_text(self.path, json.dumps(data, indent=2) + "\n")

    def lease(
        self,
        owner_id: str,
        *,
        active_owners: Optional[Iterable[str]] = None,
    ) -> int:
        """Claim a fair share for ``owner_id`` and return the leased core count.

        Recomputes leases for the full active set so leftovers are not left
        idle. ``active_owners`` defaults to current lease holders plus
        ``owner_id`` (finished workers should :meth:`release` first).
        """
        owner_id = str(owner_id)
        with _DirLock(self.lock_dir):
            data = self._read_unlocked()
            total = int(data.get("total") or self.total)
            leases = {
                str(k): int(v)
                for k, v in (data.get("leases") or {}).items()
                if int(v) > 0
            }
            if active_owners is None:
                owners = set(leases.keys()) | {owner_id}
            else:
                owners = {str(o) for o in active_owners if o is not None and str(o)}
                owners.add(owner_id)
            shares = fair_share_leases(total, owners, prefer=owner_id)
            data["leases"] = {
                k: int(shares[k]) for k in owners if shares.get(k, 0) > 0
            }
            data["total"] = total
            data["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
            _atomic_write_text(self.path, json.dumps(data, indent=2) + "\n")
            return int(data["leases"].get(owner_id, 1))
