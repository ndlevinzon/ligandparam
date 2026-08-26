"""Shared CPU lease table for parallel fragment (and similar) workers.

Workers lease a weighted share of a total core budget from a JSON file
protected by a mkdir lock. Correlated / AFFDO-style fragments (many rotors)
take a larger share than 1-D fragment jobs. Finished workers release so
remaining work can grow its ``nproc`` at the next lease boundary (scan phase
or the next sequential bond).

On each :meth:`CpuBudget.lease` call, leases for the active owner set are
recomputed so cores are not left idle among currently active workers.
In-flight wavefront pools do not resize; the new size applies at the next
scan phase or remaining-bond split.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

PathLike = str | Path


def _atomic_write_text(path: Path, text: str) -> None:
    from .ProgressBoard import atomic_write_text

    atomic_write_text(path, text)


class _DirLock:
    """Exclusive lock via ``mkdir`` (works on NFS / Windows without extras)."""

    def __init__(self, lock_dir: Path, *, timeout_sec: float = 30.0) -> None:
        from .ProgressBoard import DirLock

        self._lock = DirLock(lock_dir, timeout_sec=timeout_sec)

    def __enter__(self) -> None:
        return self._lock.__enter__()

    def __exit__(self, *exc: object) -> None:
        return self._lock.__exit__(*exc)


def cpu_lease_weight(n_bonds: int, *, correlated: bool) -> int:
    """Core-share weight for one fragment owner.

    1-D fragments (at most two fit bonds) use weight 1. Correlated /
    whole-ligand packing uses ``n_bonds``, capped at
    ``FFPOPT_WHOLE_MAX_BONDS_PER_TWIST`` (default 8).
    """
    if not correlated:
        return 1
    from ffpopt.runtime.EnvDefaults import env_int

    cap = max(1, int(env_int("FFPOPT_WHOLE_MAX_BONDS_PER_TWIST")))
    return max(1, min(int(n_bonds), cap))


def _owner_weight(weights: Mapping[str, Any] | None, owner_id: str) -> int:
    if not weights:
        return 1
    try:
        return max(1, int(weights.get(owner_id, 1)))
    except (TypeError, ValueError):
        return 1


def fair_share_leases(
    total: int,
    owners: Iterable[str],
    *,
    weights: Mapping[str, Any] | None = None,
    prefer: Optional[str] = None,
) -> dict[str, int]:
    """Distribute ``total`` cores across ``owners`` (floor 1 each when possible).

    Equal weights match the old ``total // n`` plus leftover. Non-unit
    ``weights`` use largest remainder so a correlated fragment (weight =
    n_bonds) gets more cores than a 1-D sibling. ``prefer`` receives the
    first leftover when remainders tie.
    """
    total = max(1, int(total))
    ids = sorted({str(o) for o in owners if o is not None and str(o)})
    if not ids:
        return {}
    n = len(ids)
    wmap = {oid: _owner_weight(weights, oid) for oid in ids}

    def _prefer_first(seq: list[str]) -> list[str]:
        if prefer and prefer in wmap:
            return [prefer] + [o for o in seq if o != prefer]
        return seq

    if total < n:
        ordered = _prefer_first(sorted(ids, key=lambda o: (-wmap[o], o)))
        out = {oid: 0 for oid in ids}
        for oid in ordered[:total]:
            out[oid] = 1
        return {k: v for k, v in out.items() if v > 0}

    wsum = sum(wmap.values())
    quotas = {oid: total * wmap[oid] / wsum for oid in ids}
    floors = {oid: int(quotas[oid]) for oid in ids}
    leftover = total - sum(floors.values())
    ordered_frac = _prefer_first(
        sorted(ids, key=lambda o: (-(quotas[o] - floors[o]), -wmap[o], o))
    )
    out = dict(floors)
    for oid in ordered_frac[:leftover]:
        out[oid] += 1
    if total >= n:
        zeros = [oid for oid in ids if out[oid] < 1]
        if zeros:
            donors = sorted(
                (oid for oid in ids if out[oid] > 1),
                key=lambda o: (out[o], wmap[o], o),
                reverse=True,
            )
            for zid, did in zip(zeros, donors):
                out[did] -= 1
                out[zid] = 1
    return {k: int(v) for k, v in out.items() if v > 0}


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
                data["weights"] = {}
            else:
                data.setdefault("leases", {})
                data.setdefault("weights", {})
            data["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
            _atomic_write_text(self.path, json.dumps(data, indent=2) + "\n")

    def _read_unlocked(self) -> dict[str, Any]:
        empty = {"total": self.total, "leases": {}, "weights": {}}
        if not self.path.is_file():
            return empty
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return empty
        if not isinstance(data, dict):
            return empty
        if not isinstance(data.get("leases"), dict):
            data["leases"] = {}
        if not isinstance(data.get("weights"), dict):
            data["weights"] = {}
        data["total"] = int(data.get("total") or self.total)
        return data

    def clear_leases(self) -> None:
        """Drop every lease (parent restart after kill / walltime timeout)."""
        with _DirLock(self.lock_dir):
            data = self._read_unlocked()
            data["leases"] = {}
            data["weights"] = {}
            data["total"] = self.total
            data["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
            _atomic_write_text(self.path, json.dumps(data, indent=2) + "\n")

    def snapshot(self) -> dict[str, Any]:
        """Return ``{total, leases, weights, used, free}``."""
        with _DirLock(self.lock_dir):
            data = self._read_unlocked()
        leases = {
            str(k): int(v)
            for k, v in (data.get("leases") or {}).items()
            if int(v) > 0
        }
        weights = {
            str(k): _owner_weight(data.get("weights"), str(k))
            for k in leases
        }
        used = sum(leases.values())
        total = int(data.get("total") or self.total)
        return {
            "total": total,
            "leases": leases,
            "weights": weights,
            "used": used,
            "free": max(0, total - used),
        }

    def release(self, owner_id: str) -> None:
        """Drop ``owner_id``'s lease and weight (no-op if absent)."""
        owner_id = str(owner_id)
        with _DirLock(self.lock_dir):
            data = self._read_unlocked()
            leases = data.setdefault("leases", {})
            weights = data.setdefault("weights", {})
            changed = False
            if owner_id in leases:
                del leases[owner_id]
                changed = True
            if owner_id in weights:
                del weights[owner_id]
                changed = True
            if changed:
                data["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
                _atomic_write_text(self.path, json.dumps(data, indent=2) + "\n")

    def lease(
        self,
        owner_id: str,
        *,
        weight: Optional[int] = None,
        active_owners: Optional[Iterable[str]] = None,
    ) -> int:
        """Claim a weighted share for ``owner_id`` and return leased cores.

        Recomputes leases for the full active set so leftovers are not left
        idle. ``weight`` is stored for this owner (default 1, or the last
        stored weight). ``active_owners`` defaults to current lease holders
        plus ``owner_id`` (finished workers should :meth:`release` first).
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
            stored_w = {
                str(k): _owner_weight(data.get("weights"), str(k))
                for k in (data.get("weights") or {})
            }
            if active_owners is None:
                owners = set(leases.keys()) | {owner_id}
            else:
                owners = {str(o) for o in active_owners if o is not None and str(o)}
                owners.add(owner_id)
            if weight is not None:
                stored_w[owner_id] = max(1, int(weight))
            elif owner_id not in stored_w:
                stored_w[owner_id] = 1
            w_for_share = {oid: stored_w.get(oid, 1) for oid in owners}
            shares = fair_share_leases(
                total, owners, weights=w_for_share, prefer=owner_id
            )
            data["leases"] = {
                k: int(shares[k]) for k in owners if shares.get(k, 0) > 0
            }
            data["weights"] = {
                k: int(w_for_share[k]) for k in data["leases"]
            }
            data["total"] = total
            data["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
            _atomic_write_text(self.path, json.dumps(data, indent=2) + "\n")
            return int(data["leases"].get(owner_id, 1))
