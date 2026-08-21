"""Process-pool runners for independent per-conformer ab initio ESP jobs."""

from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Optional, Sequence

from .AbInitioOptions import AbInitioOptions


_LOG = logging.getLogger("ffpopt.cpefit.ParallelEsp")


def split_core_budget(total_cores: int, n_jobs: int) -> tuple[int, int]:
    """Split a core budget across concurrent jobs and per-job threads.

    Canonical implementation: :func:`ffpopt.runtime.FastWavefront.split_core_budget`.
    """
    from ffpopt.runtime.FastWavefront import split_core_budget as _split

    return _split(total_cores, n_jobs)


def _aiopts_dict(aiopts: AbInitioOptions) -> dict:
    return {
        "program": aiopts.program,
        "theory": aiopts.theory,
        "charge": aiopts.charge,
        "mult": aiopts.mult,
        "mem": aiopts.mem,
        "nproc": aiopts.nproc,
    }


def _aiopts_from_dict(d: dict) -> AbInitioOptions:
    return AbInitioOptions(
        program=d["program"],
        theory=d["theory"],
        charge=d.get("charge", 0),
        mult=d.get("mult", 1),
        mem=d["mem"],
        nproc=d["nproc"],
    )


def needs_abinitio_esp(conf, aiopts: AbInitioOptions) -> bool:
    """True if the conformer's ESP ``.log`` still needs to be computed."""
    oname = Path(conf.GetBasename() + ".log")
    pname = Path(str(aiopts.program).split()[-1]).name.lower()
    if "psi4" in pname or "quick" in pname:
        return not oname.is_file()
    # Gaussian path also requires the input when the log is missing.
    return not oname.is_file()


def _set_job_scratch(basename: str) -> Path:
    """Give each concurrent ESP job a unique scratch directory."""
    scratch = Path.cwd() / "tmp" / f"esp_{basename}"
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ["PSI_SCRATCH"] = str(scratch)
    os.environ["GAUSS_SCRDIR"] = str(scratch)
    return scratch


def _run_abinitio_esp_job(job: dict) -> str:
    """Worker: run one conformer's ESP QM (spawn-pool picklable)."""
    conf = job["conf"]
    aiopts = _aiopts_from_dict(job["aiopts"])
    _set_job_scratch(conf.GetBasename())
    conf.RunAbInitioEspIfNeeded(aiopts)
    return conf.GetBasename()


def _run_cosmo_harmonics_job(job: dict) -> object:
    """Worker: cosmo + surface-harmonic ESP set for one conformer."""
    conf = job["conf"]
    aiopts = _aiopts_from_dict(job["aiopts"])
    _set_job_scratch(conf.GetBasename())
    conf.MakeCosmoAndSurfaceHarmonics(
        job["lmax"],
        job["qatoms"],
        aiopts,
        onlypos=bool(job.get("onlypos", False)),
    )
    return conf


def run_abinitio_esp_conformers(
    confs: Sequence,
    aiopts: AbInitioOptions,
    *,
    total_nproc: Optional[int] = None,
    logger: logging.Logger | None = None,
) -> None:
    """Run independent per-conformer ESP QM jobs, then load results in-place.

    ``aiopts.nproc`` (or ``total_nproc``) is treated as a **total** core budget
    and split across concurrent conformers. After the pool finishes, each
    conformer is reloaded via :meth:`Conformer.RunAbInitioEspIfNeeded` so the
    parent objects receive ``espvals`` (workers operate on pickled copies).
    The subsequent charge / CPE fit remains the caller's responsibility and
    should stay serial.
    """
    log = logger or _LOG
    confs = list(confs)
    if not confs:
        return

    total = int(total_nproc if total_nproc is not None else aiopts.nproc)
    pending = [c for c in confs if needs_abinitio_esp(c, aiopts)]
    if not pending:
        log.info("[esp] all %s conformer ESP log(s) present - loading", len(confs))
        for conf in confs:
            conf.RunAbInitioEspIfNeeded(aiopts)
        return

    n_workers, nproc_job = split_core_budget(total, len(pending))
    job_opts = copy.copy(aiopts)
    job_opts.nproc = int(nproc_job)
    opts_dict = _aiopts_dict(job_opts)

    log.info(
        "[esp] parallel conformer ESP: %s pending of %s, nproc=%s -> "
        "%s worker(s) x threads=%s",
        len(pending),
        len(confs),
        total,
        n_workers,
        nproc_job,
    )

    jobs = [{"conf": c, "aiopts": opts_dict} for c in pending]
    if n_workers == 1:
        for job in jobs:
            _run_abinitio_esp_job(job)
    else:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            pool.map(_run_abinitio_esp_job, jobs)

    # Reload into parent conformer objects (logs now on disk).
    for conf in confs:
        conf.RunAbInitioEspIfNeeded(aiopts)


def run_cosmo_harmonics_conformers(
    confs: Sequence,
    lmax: int,
    qatoms,
    aiopts: AbInitioOptions,
    *,
    onlypos: bool = False,
    total_nproc: Optional[int] = None,
    logger: logging.Logger | None = None,
) -> None:
    """Pool ``MakeCosmoAndSurfaceHarmonics`` across conformers; fit stays serial."""
    log = logger or _LOG
    confs = list(confs)
    if not confs:
        return

    total = int(total_nproc if total_nproc is not None else aiopts.nproc)
    n_workers, nproc_job = split_core_budget(total, len(confs))
    job_opts = copy.copy(aiopts)
    job_opts.nproc = int(nproc_job)
    opts_dict = _aiopts_dict(job_opts)

    log.info(
        "[esp] parallel cosmo/harmonics: %s conformer(s), nproc=%s -> "
        "%s worker(s) x threads=%s",
        len(confs),
        total,
        n_workers,
        nproc_job,
    )

    jobs = [
        {
            "conf": c,
            "aiopts": opts_dict,
            "lmax": int(lmax),
            "qatoms": qatoms,
            "onlypos": bool(onlypos),
        }
        for c in confs
    ]

    if n_workers == 1:
        updated = [_run_cosmo_harmonics_job(job) for job in jobs]
    else:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            updated = pool.map(_run_cosmo_harmonics_job, jobs)

    for conf, new_conf in zip(confs, updated):
        conf.desps = getattr(new_conf, "desps", None)
