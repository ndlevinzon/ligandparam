"""AIMNet2 ASE calculator factory (PyPI ``aimnet``, IsayevLab AIMNetCentral).

``--model aimnet2`` is a neural-net HL option next to ``xtb``. Energies /
forces are eV and eV/Ang (ASE). The published ``aimnet`` package needs
Python 3.11-3.13 and PyTorch 2.8+.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ffpopt.runtime.EnvDefaults import env_int, env_value

# Short CLI names -> AIMNetCentral registry aliases (member 0).
_FAMILY_ALIASES = {
    "aimnet": "aimnet2",
    "aimnet2": "aimnet2",
    "aimnet2-wb97m": "aimnet2",
    "aimnet2-wb97m-d3": "aimnet2",
    "aimnet2-2025": "aimnet2-2025",
    "aimnet2-b973c": "aimnet2-b973c",
    "aimnet2-b97-3c": "aimnet2-b973c",
    "aimnet2-nse": "aimnet2-nse",
    "aimnet2nse": "aimnet2-nse",
    "aimnet2-pd": "aimnet2-pd",
    "aimnet2pd": "aimnet2-pd",
    "aimnet2-rxn": "aimnet2-rxn",
    "aimnet2rxn": "aimnet2-rxn",
    "aimnet2-qr": "aimnet2-rxn",
}

_INSTALL_HINT = (
    "AIMNet2 is not installed. From this source tree, on Python 3.11-3.13: "
    "pip install torch --index-url https://download.pytorch.org/whl/cpu "
    '&& pip install -e ".[aimnet]". GPU: install a CUDA torch wheel first. '
    "Docs: https://isayevlab.github.io/aimnetcentral/"
)


def is_aimnet_model(model: str | None) -> bool:
    """True when ``--model`` should dispatch to AIMNet2."""
    m = (model or "").strip().lower().replace("_", "-")
    if not m:
        return False
    if "/" in m or m.endswith(".pt") or m.endswith(".jpt") or m.endswith(".safetensors"):
        return "aimnet" in m
    return "aimnet" in m


def resolve_aimnet_model_name(model: str) -> tuple[str, int]:
    """Map a CLI ``--model`` string to ``(registry_or_path, ensemble_member)``.

    Underscores become hyphens. Trailing ``-0`` .. ``-3`` select an ensemble
    member (default 0). Hugging Face repo IDs and filesystem paths pass through.
    """
    raw = (model or "").strip()
    if not raw:
        return "aimnet2", 0
    path = Path(raw)
    if path.exists() or raw.endswith((".pt", ".jpt", ".pth", ".safetensors")):
        return raw, 0
    if "/" in raw:
        # Hugging Face repo ids are lowercase; Struct/GenCalculator uppercases.
        return raw.replace("_", "-").lower(), 0

    key = raw.lower().replace("_", "-")
    member = 0
    for suffix in ("-0", "-1", "-2", "-3"):
        if key.endswith(suffix) and len(key) > len(suffix):
            member = int(suffix[-1])
            key = key[: -len(suffix)]
            break
    return _FAMILY_ALIASES.get(key, key), member


def aimnet_torch_device() -> Optional[str]:
    """``cpu`` / ``cuda`` / ``cuda:0``, or None for library auto-detect."""
    raw = env_value("FFPOPT_AIMNET_DEVICE")
    if raw is None:
        return None
    text = str(raw).strip().lower()
    return text or None


def visible_cuda_ids() -> list[str]:
    """GPU ids this process may use (Slurm ``CUDA_VISIBLE_DEVICES``)."""
    import os

    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        slurm = os.environ.get("SLURM_GPUS_ON_NODE")
        if slurm and str(slurm).strip().isdigit():
            n = int(slurm)
            return [str(i) for i in range(n)] if n > 0 else []
        return []
    text = str(raw).strip()
    if not text or text == "-1":
        return []
    return [p.strip() for p in text.split(",") if p.strip() and p.strip() != "-1"]


def _aimnet_wants_cpu(model: str | None) -> bool:
    if not is_aimnet_model(model):
        return True
    device = aimnet_torch_device()
    if device is not None and str(device).startswith("cpu"):
        return True
    return False


def cap_aimnet_nproc(nproc: int, model: str | None) -> int:
    """Limit concurrent AIMNet2 processes so they do not OOM one GPU.

    CPU AIMNet2 is unchanged. On GPU, default is
    ``n_gpu * FFPOPT_AIMNET_PER_GPU`` (default 4 per GPU), or
    ``FFPOPT_AIMNET_WORKERS`` if set.
    """
    nproc = max(1, int(nproc))
    if _aimnet_wants_cpu(model) or not is_aimnet_model(model):
        return nproc
    ids = visible_cuda_ids()
    n_gpu = len(ids)
    if n_gpu < 1:
        device = aimnet_torch_device()
        if device and str(device).startswith("cuda"):
            n_gpu = 1
        else:
            return nproc
    raw_workers = env_value("FFPOPT_AIMNET_WORKERS")
    if raw_workers is not None and str(raw_workers).strip() != "":
        cap = max(1, int(raw_workers))
    else:
        per = env_int("FFPOPT_AIMNET_PER_GPU", 4)
        cap = max(1, n_gpu * max(1, per))
    return min(nproc, cap)


def aimnet_gpu_plan_message(requested: int, capped: int, model: str | None) -> str:
    ids = visible_cuda_ids()
    per = env_int("FFPOPT_AIMNET_PER_GPU", 4)
    return (
        f"AIMNet2 GPU worker cap: nproc {requested} -> {capped} "
        f"({len(ids)} GPU(s), {per}/GPU; model={model}). "
        "Override with FFPOPT_AIMNET_WORKERS or FFPOPT_AIMNET_PER_GPU."
    )


def pin_aimnet_worker_cuda(worker_index: int) -> Optional[str]:
    """Restrict this process to one visible GPU (round-robin).

    Must run before PyTorch is imported in the worker. After pinning, the
    process sees that GPU as ``cuda:0``.
    """
    import os

    ids = visible_cuda_ids()
    if not ids:
        return None
    chosen = ids[int(worker_index) % len(ids)]
    os.environ["CUDA_VISIBLE_DEVICES"] = chosen
    os.environ["FFPOPT_AIMNET_DEVICE"] = "cuda"
    return chosen


def configure_aimnet_spawn_worker(los) -> None:
    """Pin CUDA + limit CPU threads in a wavefront spawn worker."""
    import multiprocessing
    import os

    model = getattr(getattr(los, "args", None), "model", None)
    if not is_aimnet_model(model):
        return
    ident = multiprocessing.current_process()._identity
    idx = int(ident[0]) - 1 if ident else 0
    if not _aimnet_wants_cpu(model):
        pin_aimnet_worker_cuda(idx)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        import torch

        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    except Exception:
        pass



def _import_aimnet():
    try:
        from aimnet.calculators import AIMNet2ASE, AIMNet2Calculator
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    return AIMNet2ASE, AIMNet2Calculator


def make_aimnet2_calculator(
    model: str,
    *,
    charge: Any = 0,
    spin: Any = 1,
    mfile: Optional[str] = None,
):
    """Build ``AIMNet2ASE`` for ``GenCalculator``.

    ``mfile`` (``--mfile``) overrides the registry name with a local artifact
    or Hugging Face-style directory.
    """
    AIMNet2ASE, AIMNet2Calculator = _import_aimnet()
    name, member = resolve_aimnet_model_name(mfile or model)
    device = aimnet_torch_device()
    try:
        mult = int(spin) if spin is not None else 1
    except (TypeError, ValueError):
        mult = 1
    if mult < 1:
        mult = 1
    try:
        q = int(round(float(charge))) if charge is not None else 0
    except (TypeError, ValueError):
        q = 0
    calc_kw: dict[str, Any] = {"ensemble_member": member}
    if device:
        calc_kw["device"] = device
    base = AIMNet2Calculator(name, **calc_kw)
    return AIMNet2ASE(base, charge=q, mult=mult)
