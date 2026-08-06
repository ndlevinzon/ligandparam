import pickle

import numpy as np
import pytest

from ffpopt import WaveFront
from ffpopt.WaveFront import Wavefront, wavefront_loader
from wavefront_stub import StubStruct, WritingStubNode, make_seeded_wavefront

# All 12 angles a delta=30 scan should converge to.
ALL_ANGLES = set(range(0, 360, 30))


def _run(tmp_path, nproc, name="wf.pkl"):
    wf = make_seeded_wavefront(tmp_path / name, nproc=nproc, delta=30)
    wf.calculate()
    return wf


def test_serial_full_coverage(tmp_path):
    wf = _run(tmp_path, nproc=1)
    assert set(wf.min_energies) == ALL_ANGLES
    assert all(np.isfinite(e) for e in wf.min_energies.values())


def test_parallel_full_coverage(tmp_path):
    # Exercises the multiprocessing.Pool(apply_async) + spawn path end to end.
    wf = _run(tmp_path, nproc=3)
    assert set(wf.min_energies) == ALL_ANGLES
    assert all(np.isfinite(e) for e in wf.min_energies.values())


def test_serial_and_parallel_agree(tmp_path):
    # The synthetic surface has one energy per angle, so both schedulers must
    # converge to the same per-angle minima despite completion-order drift.
    serial = _run(tmp_path, 1, "serial.pkl").min_energies
    parallel = _run(tmp_path, 3, "parallel.pkl").min_energies
    assert serial == parallel


def test_levels_are_contiguous_and_labelled(tmp_path):
    wf = _run(tmp_path, nproc=1)
    ids = [lvl.level_id for lvl in wf.levels]
    assert ids == list(range(1, len(wf.levels) + 1))
    for lvl in wf.levels:
        for node in lvl.nodes:
            assert node.level == lvl.level_id


def test_level_energies_consistent_with_min_energies(tmp_path):
    wf = _run(tmp_path, nproc=1)
    # One snapshot per level, monotonically growing coverage, and the final
    # snapshot equals the live minima (the WavefrontAnimate contract).
    assert len(wf.level_energies) == len(wf.levels)
    for earlier, later in zip(wf.level_energies, wf.level_energies[1:]):
        assert set(earlier).issubset(set(later))
    assert wf.level_energies[-1] == wf.min_energies


def test_checkpoint_roundtrip_and_idempotent_resume(tmp_path):
    ckpt = tmp_path / "wf.pkl"
    wf = make_seeded_wavefront(ckpt, nproc=1, delta=30)
    wf.calculate()
    coverage = set(wf.min_energies)

    reloaded = wavefront_loader(str(ckpt))
    assert isinstance(reloaded, Wavefront)
    assert set(reloaded.min_energies) == coverage
    # Re-running a finished, reloaded wavefront recomputes nothing and keeps the
    # same coverage (the queue is drained, no active+incomplete nodes remain).
    reloaded.nproc = 1
    reloaded.calculate()
    assert set(reloaded.min_energies) == coverage


def test_resume_from_partial_queue(tmp_path):
    # Simulate a restart: seed nodes left to run sit in _resume_queue.
    ckpt = tmp_path / "wf.pkl"
    wf = make_seeded_wavefront(ckpt, nproc=1, delta=30)
    wf._resume_queue = list(wf.levels[0].nodes)
    wf.calculate()
    assert set(wf.min_energies) == ALL_ANGLES


def test_completed_node_pickles_are_cleaned_up(tmp_path, monkeypatch):
    # Nodes write a *_node.pckl while running; once folded into a saved
    # wavefront checkpoint those are deleted, so none should remain at the end.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(WaveFront, "WavefrontNode", WritingStubNode)
    wf = make_seeded_wavefront(tmp_path / "wf.pkl", nproc=1, delta=30)
    wf.calculate()
    assert set(wf.min_energies) == ALL_ANGLES
    assert list(tmp_path.glob("*_node.pckl")) == []


def test_store_result_writeback_by_node_id(tmp_path):
    wf = make_seeded_wavefront(tmp_path / "wf.pkl", nproc=1, delta=30)
    node = wf.levels[0].nodes[1]
    # A worker returns a distinct copy of the same (level, node_id) slot.
    returned = wf.levels[0].nodes[1].__class__(
        los=StubStruct(), struct=StubStruct(), con=wf.con,
        angle=node.angle, level=node.level, node_id=node.node_id,
    )
    returned.energy = -0.5
    wf._store_result(returned)
    stored = wf.levels[0].nodes[1]
    assert stored is returned
    assert stored.energy == -0.5
    assert stored.los is wf.los  # re-pointed at the shared los, no per-node copy
