"""Shared CPU lease table for parallel fragment (and similar) workers.

Workers lease a weighted share of a total core budget from a JSON file
protected by a mkdir lock. A scan never starts on a single core when the
budget can spare ``FFPOPT_MIN_WF_NPROC`` (at least 2): extra owners wait
instead of starving. ``n_alive`` reserves cores for fragments that have
not leased yet, so the first scanner does not grab the whole node, and a
leftover scanner can take what remains after siblings finish or enter
fit. In-flight wavefront pools do not resize; the new size applies at
the next scan phase or remaining-bond split.
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


def cpu_min_lease(total: int | None = None) -> int:
    """Never start a scan on one core when the budget can spare more.

    Floor is ``max(2, FFPOPT_MIN_WF_NPROC)``, capped at ``total`` so
    ``-n 1`` still runs. A job that cannot get this many cores waits.
    """
    from ffpopt.runtime.EnvDefaults import env_int

    floor = max(2, int(env_int("FFPOPT_MIN_WF_NPROC")))
    if total is None:
        return floor
    return min(floor, max(1, int(total)))


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
    min_each: int = 1,
    virtual_units: int = 0,
) -> dict[str, int]:
    """Distribute ``total`` cores across ``owners``.

    Equal weights match ``total // n`` plus leftover. Non-unit ``weights``
    use largest remainder. ``min_each`` > 1 never assigns 1 core when the
    budget can give two: extra owners get 0 instead of starving on a
    single worker. ``virtual_units`` are reserved weight-1 slots for
    fragments that have not leased yet, so the first scanner does not
    grab the whole node.
    """
    total = max(1, int(total))
    min_each = max(1, int(min_each))
    virtual_units = max(0, int(virtual_units))
    ids = sorted({str(o) for o in owners if o is not None and str(o)})
    if not ids:
        return {}
    n = len(ids)
    wmap = {oid: _owner_weight(weights, oid) for oid in ids}

    def _prefer_first(seq: list[str]) -> list[str]:
        if prefer and prefer in wmap:
            return [prefer] + [o for o in seq if o != prefer]
        return seq

    if total < min_each:
        ordered = _prefer_first(sorted(ids, key=lambda o: (-wmap[o], o)))
        return {ordered[0]: total}

    if min_each > 1 or virtual_units > 0:
        w_real = sum(wmap.values())
        W = max(1, w_real + virtual_units)
        reserved = int(total * virtual_units / W) if virtual_units else 0
        available = max(0, total - reserved)
        if available < min_each:
            if total >= min_each:
                ordered = _prefer_first(sorted(ids, key=lambda o: (-wmap[o], o)))
                return {ordered[0]: min_each}
            return {}
        n_fit = min(n, available // min_each)
        chosen = _prefer_first(sorted(ids, key=lambda o: (-wmap[o], o)))[:n_fit]
        rem = available - min_each * len(chosen)
        w_ch = sum(wmap[o] for o in chosen) or len(chosen)
        quotas = {o: rem * wmap[o] / w_ch for o in chosen}
        floors = {o: int(quotas[o]) for o in chosen}
        leftover = rem - sum(floors.values())
        ordered_frac = _prefer_first(
            sorted(
                chosen,
                key=lambda o: (-(quotas[o] - floors[o]), -wmap[o], o),
            )
        )
        out = {o: min_each + floors[o] for o in chosen}
        for oid in ordered_frac[:leftover]:
            out[oid] += 1
        return {k: int(v) for k, v in out.items() if v > 0}

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
        self,
        path: PathLike,
        total: int,
        *,
        clear_leases: bool = False,
        n_alive: Optional[int] = None,
    ) -> None:
        self.path = Path(path)
        self.total = max(1, int(total))
        self.lock_dir = self.path.with_suffix(self.path.suffix + ".lock")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _DirLock(self.lock_dir):
            data = self._read_unlocked()
            prev_total = int(data.get("total") or 0)
            if clear_leases:
                data["total"] = self.total
                data["leases"] = {}
                data["weights"] = {}
                data["n_alive"] = max(0, int(n_alive or 0))
                data["started"] = []
            else:
                # Never shrink the parent budget if a worker passes a hint of 1.
                data["total"] = (
                    max(self.total, prev_total) if prev_total else self.total
                )
                data.setdefault("leases", {})
                data.setdefault("weights", {})
                data.setdefault("started", [])
                if n_alive is not None:
                    data["n_alive"] = max(0, int(n_alive))
                else:
                    data.setdefault("n_alive", 0)
            self.total = int(data["total"])
            data["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
            _atomic_write_text(self.path, json.dumps(data, indent=2) + "\n")

    def _read_unlocked(self) -> dict[str, Any]:
        empty = {
            "total": self.total,
            "leases": {},
            "weights": {},
            "n_alive": 0,
            "started": [],
        }
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
        try:
            data["n_alive"] = max(0, int(data.get("n_alive") or 0))
        except (TypeError, ValueError):
            data["n_alive"] = 0
        started = data.get("started")
        if not isinstance(started, list):
            data["started"] = []
        data["total"] = int(data.get("total") or self.total)
        return data

    def clear_leases(self) -> None:
        """Drop every lease (parent restart after kill / walltime timeout)."""
        with _DirLock(self.lock_dir):
            data = self._read_unlocked()
            data["leases"] = {}
            data["weights"] = {}
            data["n_alive"] = 0
            data["started"] = []
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
            "n_alive": int(data.get("n_alive") or 0),
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

    def dec_alive(self) -> int:
        """One unfinished fragment finished (success or fail). Returns n_alive."""
        with _DirLock(self.lock_dir):
            data = self._read_unlocked()
            n_alive = max(0, int(data.get("n_alive") or 0) - 1)
            data["n_alive"] = n_alive
            data["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
            _atomic_write_text(self.path, json.dumps(data, indent=2) + "\n")
            return n_alive

    def lease(
        self,
        owner_id: str,
        *,
        weight: Optional[int] = None,
        active_owners: Optional[Iterable[str]] = None,
    ) -> int:
        """Claim cores for ``owner_id`` without shrinking in-flight siblings.

        Returns 0 if fewer than ``cpu_min_lease`` cores are free (caller
        should wait). Unfinished fragments that have not leased yet are
        reserved via ``n_alive`` so the first scanner cannot take the
        whole node, and a leftover scanner can take what remains.
        """
        owner_id = str(owner_id)
        with _DirLock(self.lock_dir):
            data = self._read_unlocked()
            total = int(data.get("total") or self.total)
            min_each = cpu_min_lease(total)
            leases = {
                str(k): int(v)
                for k, v in (data.get("leases") or {}).items()
                if int(v) > 0
            }
            stored_w = {
                str(k): _owner_weight(data.get("weights"), str(k))
                for k in (data.get("weights") or {})
            }
            others = {k: v for k, v in leases.items() if k != owner_id}
            if active_owners is not None:
                keep = {str(o) for o in active_owners if o is not None and str(o)}
                keep.discard(owner_id)
                others = {k: v for k, v in others.items() if k in keep}
            if weight is not None:
                stored_w[owner_id] = max(1, int(weight))
            elif owner_id not in stored_w:
                stored_w[owner_id] = 1
            owners = set(others.keys()) | {owner_id}
            w_for_share = {oid: stored_w.get(oid, 1) for oid in owners}
            n_alive = int(data.get("n_alive") or 0)
            started = {str(x) for x in (data.get("started") or [])}
            virtual = max(0, n_alive - len(started | {owner_id}))
            free = max(0, total - sum(others.values()))
            if free < min_each:
                leases_out = dict(others)
                data["leases"] = leases_out
                data["weights"] = {
                    k: int(w_for_share.get(k, stored_w.get(k, 1)))
                    for k in leases_out
                }
                data["total"] = total
                data["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
                _atomic_write_text(self.path, json.dumps(data, indent=2) + "\n")
                return 0
            shares = fair_share_leases(
                total,
                owners,
                weights=w_for_share,
                prefer=owner_id,
                min_each=min_each,
                virtual_units=virtual,
            )
            want = int(shares.get(owner_id, 0))
            leased = min(free, want)
            if leased < min_each:
                leases_out = dict(others)
                data["leases"] = leases_out
                data["weights"] = {
                    k: int(w_for_share.get(k, stored_w.get(k, 1)))
                    for k in leases_out
                }
                data["total"] = total
                data["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
                _atomic_write_text(self.path, json.dumps(data, indent=2) + "\n")
                return 0
            leases_out = dict(others)
            leases_out[owner_id] = int(leased)
            data["leases"] = leases_out
            data["weights"] = {
                k: int(w_for_share.get(k, stored_w.get(k, 1)))
                for k in leases_out
            }
            data["weights"][owner_id] = int(stored_w[owner_id])
            started.add(owner_id)
            data["started"] = sorted(started)
            data["total"] = total
            data["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
            _atomic_write_text(self.path, json.dumps(data, indent=2) + "\n")
            return int(leased)
