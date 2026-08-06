"""Importable stubs for exercising the wavefront calculation queue without QM.

Kept in its own module (not the test file or conftest) so that instances remain
picklable and the synthetic ``calculate`` survives the multiprocessing ``spawn``
boundary: a pool worker re-imports this module to unpickle a node and resolves
``StubNode.calculate`` from here rather than running the real optimizer.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from ffpopt import WaveFront
from ffpopt.WaveFront import Wavefront, WavefrontNode


class StubStruct:
    """Minimal stand-in for ffpopt.Struct.Struct (only ``data['elements']``)."""

    def __init__(self, nelem=2):
        self.data = {"elements": ["C"] * nelem}


class StubLos:
    """Minimal stand-in for ListOfStruct; only needs a nullable ``calc``."""

    def __init__(self):
        self.calc = None


class StubCon:
    """Deep-copyable constraint stand-in."""

    def __init__(self):
        self.value = None


def energy_for_angle(angle):
    """Deterministic synthetic torsion energy with one broad minimum at 180."""
    return -math.cos(math.radians(angle - 180.0))


class StubNode(WavefrontNode):
    """WavefrontNode whose calculate() is a dependency-free synthetic surface."""

    def replace_with_pickle(self):
        return

    def _write_checkpoint(self):
        return

    def cleanup(self):
        return

    def calculate(self):
        if self.complete:
            return
        self.energy = np.round(energy_for_angle(self.angle), 6)
        self.opt_geom = StubStruct(len(self.struct.data["elements"]))
        self.complete = True


class WritingStubNode(WavefrontNode):
    """Like StubNode but uses the real ``_write_checkpoint``/``cleanup`` so the
    per-node ``*_node.pckl`` lifecycle (write while running, delete once folded
    into a wavefront checkpoint) can be asserted. Intended for serial use under
    a temporary cwd."""

    def calculate(self):
        if self.complete:
            return
        self.energy = np.round(energy_for_angle(self.angle), 6)
        self.opt_geom = StubStruct(len(self.struct.data["elements"]))
        self._write_checkpoint()
        self.complete = True


# Make every node the wavefront creates (seeds + spawned neighbors) a StubNode,
# in this process and in any spawn worker that imports this module.
WaveFront.WavefrontNode = StubNode


def make_seeded_wavefront(checkpoint, nproc=1, delta=30, seed_angles=(0, 90, 180, 270)):
    """Build a Wavefront pre-seeded with level-1 stub nodes (skips init_min)."""
    wf = Wavefront(
        StubLos(), StubStruct(), StubCon(),
        delta=delta, max_levels=-1, nproc=nproc, checkpoint=str(checkpoint),
    )
    level1 = wf._get_or_create_level(1)
    for angle in seed_angles:
        level1.add_node(wf.los, StubStruct(), wf.con, angle=angle)
    return wf
