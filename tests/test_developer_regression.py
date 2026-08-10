"""Developer regression suite for ligandparam / ffpopt / scission.

Run after code changes to catch regressions in recipe wiring, logging,
I/O contracts, and core pure-Python helpers (no AmberTools / Gaussian)::

    python -m unittest tests.test_developer_regression -v

Full developer battery (this file + specialized unit tests)::

    python -m unittest discover -s tests -v

Install validation for end users remains::

    python -m unittest tests.test_install_validation -v
"""

from __future__ import annotations

import importlib
import io
import logging
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _has_module(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def _require_rdkit(test: unittest.TestCase) -> None:
    if not _has_module("rdkit"):
        test.skipTest("rdkit required for recipe/stage imports")


# ---------------------------------------------------------------------------
# Recipes — registry + setup() stage graphs (contract / wiring)
# ---------------------------------------------------------------------------


class TestRecipeRegistry(unittest.TestCase):
    def test_available_recipes_matches_registry_keys(self):
        from ligandparam.recipes.registry import _REGISTRY, available_recipes

        self.assertEqual(available_recipes(), sorted(_REGISTRY))
        for expected in (
            "freeligand",
            "lazyligand",
            "lazierligand",
            "dplazyligand",
            "dpfreeligand",
            "sqmligand",
        ):
            self.assertIn(expected, _REGISTRY)

    def test_unknown_recipe_raises(self):
        from ligandparam.recipes.registry import get_recipe

        with self.assertRaises(ValueError) as ctx:
            get_recipe("not-a-recipe")
        self.assertIn("Unknown recipe", str(ctx.exception))

    def test_buildligand_not_registered(self):
        from ligandparam.recipes.registry import available_recipes

        self.assertNotIn("buildligand", available_recipes())


class TestRecipeSetupGraphs(unittest.TestCase):
    """Each registered recipe builds a non-empty, ordered stage list."""

    def setUp(self):
        _require_rdkit(self)

    def _tmp_recipe_args(self, td: str):
        cwd = Path(td)
        return cwd / "ligand.pdb", cwd

    def _assert_tail_parmchk_leap(self, stages):
        from ligandparam.stages import StageLeap, StageParmChk

        types = [type(s) for s in stages]
        self.assertEqual(types[-2], StageParmChk)
        self.assertEqual(types[-1], StageLeap)

    def test_freeligand_setup(self):
        from ligandparam.recipes.freeligand import FreeLigand
        from ligandparam.stages import (
            StageInitialize,
            StageMultiRespFit,
            StageNormalizeCharge,
        )

        with tempfile.TemporaryDirectory() as td:
            inp, cwd = self._tmp_recipe_args(td)
            recipe = FreeLigand(inp, cwd, net_charge=0, nproc=2, mem=4, logger="stream")
            recipe.setup()
            types = [type(s) for s in recipe.stages]
            self.assertGreaterEqual(len(types), 5)
            self.assertEqual(types[0], StageInitialize)
            self.assertIn(StageNormalizeCharge, types)
            self.assertIn(StageMultiRespFit, types)
            self._assert_tail_parmchk_leap(recipe.stages)
            self.assertEqual(recipe.net_charge, 0)

    def test_freeligand_missing_net_charge(self):
        from ligandparam.recipes.freeligand import FreeLigand

        with self.assertRaises(KeyError):
            FreeLigand("ligand.pdb", "out_dir")

    def test_freeligand_bad_orientation_protocol(self):
        from ligandparam.recipes.freeligand import FreeLigand

        with self.assertRaises(ValueError):
            FreeLigand(
                "ligand.pdb",
                "out_dir",
                net_charge=0,
                orientation_protocol="not_a_protocol",
            )

    def test_lazyligand_setup(self):
        from ligandparam.recipes.lazyligand import LazyLigand
        from ligandparam.stages import StageInitialize, StageLazyResp

        with tempfile.TemporaryDirectory() as td:
            inp, cwd = self._tmp_recipe_args(td)
            recipe = LazyLigand(inp, cwd, net_charge=-1, logger="stream")
            recipe.setup()
            types = [type(s) for s in recipe.stages]
            self.assertEqual(types[0], StageInitialize)
            self.assertIn(StageLazyResp, types)
            self._assert_tail_parmchk_leap(recipe.stages)

    def test_lazierligand_setup(self):
        from ligandparam.recipes.lazierligand import LazierLigand
        from ligandparam.stages import StageInitialize, StageNormalizeCharge

        with tempfile.TemporaryDirectory() as td:
            inp, cwd = self._tmp_recipe_args(td)
            recipe = LazierLigand(inp, cwd, net_charge=0, logger="stream")
            recipe.setup()
            types = [type(s) for s in recipe.stages]
            self.assertEqual(types[0], StageInitialize)
            self.assertIn(StageNormalizeCharge, types)
            self._assert_tail_parmchk_leap(recipe.stages)

    def test_dpligand_setup_includes_dpminimize(self):
        from ligandparam.recipes.dplazyligand import DPLigand
        from ligandparam.stages import DPMinimize, StageInitialize

        with tempfile.TemporaryDirectory() as td:
            inp, cwd = self._tmp_recipe_args(td)
            recipe = DPLigand(inp, cwd, net_charge=0, logger="stream")
            recipe.setup()
            types = [type(s) for s in recipe.stages]
            self.assertEqual(types[0], StageInitialize)
            self.assertIn(DPMinimize, types)
            self._assert_tail_parmchk_leap(recipe.stages)

    def test_dpfreeligand_setup(self):
        from ligandparam.recipes.dpfreeligand import DPFreeLigand
        from ligandparam.stages import DPMinimize, StageInitialize, StageMultiRespFit

        with tempfile.TemporaryDirectory() as td:
            inp, cwd = self._tmp_recipe_args(td)
            recipe = DPFreeLigand(inp, cwd, net_charge=0, logger="stream")
            recipe.setup()
            types = [type(s) for s in recipe.stages]
            self.assertEqual(types[0], StageInitialize)
            self.assertIn(DPMinimize, types)
            self.assertIn(StageMultiRespFit, types)
            self._assert_tail_parmchk_leap(recipe.stages)

    def test_sqmligand_setup(self):
        from ligandparam.recipes.optligand import SQMLigand
        from ligandparam.stages import DPMinimize, StageInitialize

        with tempfile.TemporaryDirectory() as td:
            inp, cwd = self._tmp_recipe_args(td)
            recipe = SQMLigand(inp, cwd, net_charge=0, logger="stream")
            recipe.setup()
            types = [type(s) for s in recipe.stages]
            self.assertEqual(types[0], StageInitialize)
            self.assertIn(DPMinimize, types)
            self._assert_tail_parmchk_leap(recipe.stages)

    def test_dihed_correct_appends_twist_stage(self):
        from ligandparam.recipes.freeligand import FreeLigand
        from ligandparam.stages.ffpopt_dihed import StageDihedTwistCorrection

        with tempfile.TemporaryDirectory() as td:
            inp, cwd = self._tmp_recipe_args(td)
            recipe = FreeLigand(
                inp,
                cwd,
                net_charge=0,
                dihed_correct=True,
                dihed_model="xtb",
                dihed_delta=15,
                logger="stream",
            )
            recipe.setup()
            types = [type(s) for s in recipe.stages]
            self.assertEqual(types[-1], StageDihedTwistCorrection)
            twist = recipe.stages[-1]
            self.assertEqual(twist.model, "xtb")
            self.assertEqual(twist.delta, 15)

    def test_dry_run_execute_invokes_each_stage(self):
        """Driver.execute(dry_run=True) must call every stage.execute."""
        from ligandparam.recipes.lazierligand import LazierLigand

        with tempfile.TemporaryDirectory() as td:
            inp, cwd = self._tmp_recipe_args(td)
            recipe = LazierLigand(inp, cwd, net_charge=0, logger="stream")
            recipe.setup()
            calls = []
            for stage in recipe.stages:
                stage.execute = MagicMock(
                    side_effect=lambda *a, _s=stage, **k: calls.append(_s.stage_name)
                )
            recipe.execute(dry_run=True)
            self.assertEqual(len(calls), len(recipe.stages))
            self.assertEqual(calls, [s.stage_name for s in recipe.stages])


# ---------------------------------------------------------------------------
# Stages — charge normalize + abstract contracts
# ---------------------------------------------------------------------------


class TestStageChargeNormalize(unittest.TestCase):
    def _stage(self, net_charge=0, precision=0.001, decimals=3):
        from ligandparam.stages.charge import StageNormalizeCharge

        st = StageNormalizeCharge.__new__(StageNormalizeCharge)
        st.net_charge = net_charge
        st.precision = precision
        st.decimals = decimals
        st.logger = MagicMock()
        return st

    def test_nonzero_net_charge(self):
        st = self._stage(net_charge=1, precision=0.001, decimals=3)
        q = [0.4, 0.3, 0.2]
        rounded, total, diff = st.check_charge(q)
        out = st.normalize(rounded, diff)
        _, new_total, _ = st.check_charge(out)
        self.assertTrue(abs(new_total - 1.0) < 0.002)

    def test_zero_count_safe(self):
        st = self._stage(net_charge=0)
        out = st.normalize([0.0, 0.0], 0.0)
        self.assertEqual(list(out), [0.0, 0.0])

    def test_large_delta_warns(self):
        st = self._stage(net_charge=1)
        out = st.normalize([0.0, 0.0], 0.05)
        self.assertAlmostEqual(float(sum(out)), 0.05, places=6)
        st.logger.warning.assert_called()


class TestCommonRecipeTail(unittest.TestCase):
    def test_charge_update_parmchk_leap_order(self):
        _require_rdkit(self)
        from ligandparam.recipes.common import charge_update_parmchk_leap_stages
        from ligandparam.stages import StageLeap, StageParmChk, StageUpdate

        recipe = SimpleNamespace(cwd=Path("."), net_charge=0, logger=None, kwargs={})
        stages = charge_update_parmchk_leap_stages(
            recipe=recipe,
            initial_mol2="a.mol2",
            final_mol2="b.mol2",
            nonminimized_mol2="c.mol2",
            frcmod="x.frcmod",
            lib="x.lib",
        )
        self.assertEqual(
            [type(s) for s in stages],
            [StageUpdate, StageParmChk, StageLeap],
        )


# ---------------------------------------------------------------------------
# Logging / console contracts
# ---------------------------------------------------------------------------


class TestLoggingContracts(unittest.TestCase):
    def setUp(self):
        from ffpopt.runtime import console as console_mod

        console_mod._BANNER_PRINTED = False
        os.environ.pop("LIGANDPARAM_BANNER_PRINTED", None)

    def tearDown(self):
        from ffpopt.runtime import console as console_mod

        console_mod._BANNER_PRINTED = False
        os.environ.pop("LIGANDPARAM_BANNER_PRINTED", None)

    def test_format_console_line_peels_scopes(self):
        from ffpopt.runtime.console import format_console_line

        line = format_console_line("[frag-twist] hello", tag="ffpopt:fragment_1")
        self.assertRegex(line, r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} ")
        self.assertIn("[ffpopt:fragment_1]", line)
        self.assertIn("[frag-twist]", line)
        self.assertIn("hello", line)
        self.assertTrue(line.endswith("\n"))

    def test_attach_console_handlers_idempotent(self):
        from ffpopt.runtime.console import attach_console_handlers

        logger = logging.getLogger("test.dev.console")
        logger.handlers.clear()
        attach_console_handlers(logger, tag="ligandparam")
        n = len(logger.handlers)
        attach_console_handlers(logger, tag="ligandparam")
        self.assertEqual(len(logger.handlers), n)
        logger.handlers.clear()

    def test_set_stream_logger_tags_ligandparam(self):
        from ligandparam.log import set_stream_logger

        logger = set_stream_logger()
        markers = [
            getattr(h, "_lp_console_marker", None) for h in logger.handlers
        ]
        self.assertTrue(any(m and "ligandparam" in m for m in markers if m))

    def test_banner_not_from_attach(self):
        from ffpopt.runtime import console as console_mod

        logger = logging.getLogger("test.dev.banner")
        logger.handlers.clear()
        console_mod.attach_console_handlers(logger, tag="x")
        self.assertFalse(console_mod._BANNER_PRINTED)
        logger.handlers.clear()


# ---------------------------------------------------------------------------
# ligandparam I/O — amber bundle resolution
# ---------------------------------------------------------------------------


class TestAmberBundleIO(unittest.TestCase):
    def _touch_triplet(self, work_dir: Path, stem: str) -> None:
        (work_dir / f"{stem}.mol2").write_text("@<TRIPOS>MOLECULE\n", encoding="utf-8")
        (work_dir / f"{stem}.lib").write_text("!entry\n", encoding="utf-8")
        (work_dir / f"{stem}.frcmod").write_text("Remark line\n", encoding="utf-8")

    def test_resolve_explicit_paths(self):
        from ligandparam.io.amber_bundle import AmberLigandBundle, resolve_getparam_bundle

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._touch_triplet(root, "chaps")
            bundle = resolve_getparam_bundle(
                mol2=root / "chaps.mol2",
                lib=root / "chaps.lib",
                frcmod=root / "chaps.frcmod",
            )
            self.assertIsInstance(bundle, AmberLigandBundle)
            self.assertEqual(bundle.stem, "chaps")
            self.assertEqual(bundle.work_dir, root.resolve())

    def test_resolve_getparam_layout_with_label(self):
        from ligandparam.io.amber_bundle import resolve_getparam_bundle

        with tempfile.TemporaryDirectory() as td:
            cwd = Path(td)
            work = cwd / "CHA3" / "CHA"
            work.mkdir(parents=True)
            self._touch_triplet(work, "chaps")
            bundle = resolve_getparam_bundle(
                cwd=cwd, data_cwd="CHA3", resname="CHA", label="chaps"
            )
            self.assertEqual(bundle.stem, "chaps")
            self.assertEqual(bundle.work_dir, work.resolve())

    def test_missing_triplet_raises(self):
        from ligandparam.io.amber_bundle import resolve_getparam_bundle

        with tempfile.TemporaryDirectory() as td:
            cwd = Path(td)
            (cwd / "A" / "B").mkdir(parents=True)
            with self.assertRaises(FileNotFoundError):
                resolve_getparam_bundle(
                    cwd=cwd, data_cwd="A", resname="B", label="x"
                )

    def test_to_scission_input(self):
        from ligandparam.io.amber_bundle import resolve_getparam_bundle

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._touch_triplet(root, "LIG")
            bundle = resolve_getparam_bundle(
                mol2=root / "LIG.mol2",
                lib=root / "LIG.lib",
                frcmod=root / "LIG.frcmod",
            )
            inp = bundle.to_scission_input()
            self.assertEqual(inp.mol2_path, bundle.mol2)
            self.assertEqual(inp.lib_path, bundle.lib)
            self.assertEqual(inp.frcmod_path, bundle.frcmod)


# ---------------------------------------------------------------------------
# Dihed option helpers + workflow bond contracts
# ---------------------------------------------------------------------------


class TestDihedOptionsAndBonds(unittest.TestCase):
    def test_pop_and_apply_dihed_options(self):
        from ligandparam.recipes.dihed_options import apply_dihed_options, pop_dihed_options

        kwargs = {
            "dihed_correct": True,
            "dihed_model": "xtb",
            "dihed_delta": 5,
            "keep": 1,
        }
        opts = pop_dihed_options(dict(kwargs))
        self.assertTrue(opts["dihed_correct"])
        self.assertEqual(opts["dihed_delta"], 5)
        obj = SimpleNamespace()
        apply_dihed_options(obj, kwargs)
        self.assertTrue(obj.dihed_correct)
        self.assertEqual(obj.dihed_model, "xtb")
        self.assertEqual(kwargs, {"keep": 1})

    def test_coerce_fragment_config(self):
        from ligandparam.recipes.dihed_options import coerce_fragment_config
        from scission.models import FragmentConfig

        self.assertIsNone(coerce_fragment_config(None))
        cfg = coerce_fragment_config({"angle_step": 15, "cap_strategy": "hydrogen"})
        self.assertIsInstance(cfg, FragmentConfig)
        self.assertEqual(cfg.angle_step, 15)
        with self.assertRaises(TypeError):
            coerce_fragment_config("nope")

    def test_normalize_bond_pairs0(self):
        from ffpopt.Workflows import normalize_bond_pairs0

        self.assertEqual(normalize_bond_pairs0([(1, 2), [3, 4]]), [(1, 2), (3, 4)])
        self.assertEqual(normalize_bond_pairs0(["0,1", "10,11"]), [(0, 1), (10, 11)])
        with self.assertRaises(ValueError):
            normalize_bond_pairs0(["0-1"])
        with self.assertRaises(TypeError):
            normalize_bond_pairs0([42])

    def test_bonds0_from_scission_fit_torsions(self):
        from ffpopt.Workflows import bonds0_from_scission_fit_torsions

        pairs = bonds0_from_scission_fit_torsions(
            [
                {"fragment_rotatable_bond": [1, 2]},
                {"fragment_rotatable_bond": [5, 8]},
            ]
        )
        self.assertEqual(pairs, [(0, 1), (4, 7)])


# ---------------------------------------------------------------------------
# scission — models + merge + torsions on synthetic ligand
# ---------------------------------------------------------------------------


class TestScissionFunctions(unittest.TestCase):
    def _butane_like_ligand(self):
        """Linear C4 chain with hydrogens — rotatable C–C bonds."""
        from scission.models import Atom, Bond, Ligand

        atoms = []
        coords = [
            (0.0, 0.0, 0.0),
            (1.5, 0.0, 0.0),
            (3.0, 0.0, 0.0),
            (4.5, 0.0, 0.0),
        ]
        for i, xyz in enumerate(coords, start=1):
            atoms.append(
                Atom(
                    index=i,
                    name=f"C{i}",
                    element="C",
                    atom_type="c3",
                    charge=-0.1,
                    coords=xyz,
                )
            )
        atoms.append(Atom(5, "H1", "H", "hc", 0.1, (-1.0, 0.0, 0.0)))
        atoms.append(Atom(6, "H4", "H", "hc", 0.1, (5.5, 0.0, 0.0)))
        bonds = [
            Bond(1, 1, 2, "1"),
            Bond(2, 2, 3, "1"),
            Bond(3, 3, 4, "1"),
            Bond(4, 1, 5, "1"),
            Bond(5, 4, 6, "1"),
        ]
        return Ligand(
            name="but",
            atoms=atoms,
            bonds=bonds,
            lib_atom_names=[a.name for a in atoms],
            lib_atom_types={a.name: a.atom_type for a in atoms},
            frcmod_text="",
            mol2_path=Path("but.mol2"),
            lib_path=Path("but.lib"),
            frcmod_path=Path("but.frcmod"),
        )

    def test_find_rotatable_bonds_and_enumerate_torsions(self):
        from scission.torsions import enumerate_torsions, find_rotatable_bonds

        lig = self._butane_like_ligand()
        rots = find_rotatable_bonds(lig)
        self.assertGreaterEqual(len(rots), 1)
        tors = enumerate_torsions(lig)
        self.assertGreaterEqual(len(tors), 1)
        for t in tors:
            self.assertEqual(len(t.atom_indices), 4)
            self.assertEqual(len(t.bond), 2)

    def test_fragment_config_defaults_and_from_dict(self):
        from scission.models import FragmentConfig

        cfg = FragmentConfig()
        self.assertTrue(hasattr(cfg, "angle_step") or hasattr(cfg, "cap_strategy"))
        cfg2 = FragmentConfig.from_dict({"angle_step": 15})
        self.assertEqual(cfg2.angle_step, 15)

    def test_write_fragment_index_and_merge_accumulate(self):
        from scission.merge import _load_fragment_update
        from scission.models import SelectedFragment
        from scission.writers import write_fragment_index

        frag = SelectedFragment(
            fragment_id="frag_0001",
            source_candidate_id="cand_a",
            retained_atoms=[0, 1, 2],
            cut_bonds=[(2, 3)],
            cap_atoms=[],
            torsions=["t1"],
            fit_torsions=[],
            parent_atom_map={0: 0, 1: 1, 2: 2},
            manifest_path=Path("frags/frag_0001/manifest.json"),
        )
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            path = write_fragment_index([frag], out)
            self.assertTrue(path.is_file())

            def _frcmod(lines):
                return (
                    "Remark line goes here\nMASS\n\nBOND\n\nANGLE\n\nDIHE\n"
                    + "".join(f"{ln}\n" for ln in lines)
                    + "\nIMPROPER\n\nNONB\n\n"
                )

            frag_dir = out / "frag_0001"
            frag_dir.mkdir()
            (frag_dir / "it01.frcmod").write_text(
                _frcmod(["c3-c3-c3-c3 1 1.00 0.0 1.", "c3-c3-c3-n  1 2.00 0.0 1."])
            )
            (frag_dir / "it02.frcmod").write_text(
                _frcmod(["c3-c3-c3-n  1 3.50 0.0 1."])
            )
            update = _load_fragment_update(frag_dir)
            self.assertIn(("c3", "c3", "c3", "c3"), update["dihe_groups"])
            self.assertIn(("c3", "c3", "c3", "n"), update["dihe_groups"])


# ---------------------------------------------------------------------------
# ffpopt — constraints, dihedrals, wavefront policy, runtime helpers
# ---------------------------------------------------------------------------


class TestFfpoptCoreFunctions(unittest.TestCase):
    def test_constraints_to_geometric(self):
        from ffpopt.Constraints import Constraint, to_geometric

        lines = to_geometric([Constraint("dihed", [0, 1, 2, 3], value=45.0)])
        self.assertTrue(any("45.0" in ln and ln.startswith("dihedral") for ln in lines))

    def test_is_soft_opt_and_evaluate_policy(self):
        from ffpopt.GeomOpt import is_soft_opt_recovery
        from ffpopt.scan.wavefront_mixins import evaluate_wavefront_minimum

        self.assertTrue(is_soft_opt_recovery("loose"))
        self.assertFalse(is_soft_opt_recovery("primary"))
        d = evaluate_wavefront_minimum(
            energy=1.0,
            soft=True,
            has_incumbent=False,
            incumbent_energy=None,
            incumbent_soft=False,
            threshold_ev=0.1,
        )
        self.assertEqual(d["reason"], "soft_first_seed")

    def test_shape_match_and_joint_ls_symbols(self):
        import numpy as np
        from ffpopt.Dihedrals import shape_match_delta

        hl = np.array([1.0, 2.0, 3.0])
        ll = np.array([0.0, 1.0, 2.0])
        d = shape_match_delta(hl, ll)
        np.testing.assert_allclose(d, shape_match_delta(hl, ll + 9.0))

    def test_align_scan_profiles(self):
        from ffpopt.Dihedrals import align_scan_profiles
        from ffpopt.Struct import ListOfStruct

        def _frame(name, e=0.0):
            return SimpleNamespace(
                data={
                    "name": name,
                    "energy": e,
                    "positions": [[0.0, 0.0, 0.0]],
                    "constraints": [],
                },
                constraints=None,
            )

        hl = ListOfStruct.from_structs_shared(
            [_frame("d000"), _frame("d010"), _frame("d020"), _frame("d030")]
        )
        ll = ListOfStruct.from_structs_shared(
            [_frame("d010"), _frame("d020"), _frame("d030"), _frame("d040")]
        )
        ahl, all_, info = align_scan_profiles(hl, ll)
        self.assertEqual(info["n_common"], 3)
        self.assertEqual(len(ahl.structs), len(all_.structs))

    def test_fast_wavefront_enabled(self):
        from ffpopt.runtime.fast_wavefront import fast_wavefront_enabled

        self.assertTrue(fast_wavefront_enabled(explicit=True))
        self.assertFalse(fast_wavefront_enabled(explicit=False))
        with patch.dict(os.environ, {"FFPOPT_FAST_WAVEFRONT": "1"}):
            self.assertTrue(fast_wavefront_enabled())
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FFPOPT_FAST_WAVEFRONT", None)
            self.assertFalse(fast_wavefront_enabled())

    def test_cpu_budget_fair_share(self):
        from ffpopt.runtime.cpu_budget import fair_share_leases

        leases = fair_share_leases(8, ["a", "b", "c"])
        self.assertEqual(sum(leases.values()), 8)
        self.assertEqual(len(leases), 3)

    def test_split_nproc_for_items(self):
        from ffpopt.runtime.fast_wavefront import split_nproc_for_items

        n_outer, n_inner = split_nproc_for_items(8, 4)
        self.assertEqual(n_outer * n_inner, 8)

    def test_pickle_compat_alias(self):
        import ffpopt.WaveFront as legacy
        from ffpopt.scan.WaveFront import Wavefront

        self.assertIs(legacy.Wavefront, Wavefront)

    def test_prim_dihed_energy_term(self):
        from ffpopt.Dihedrals import PrimDihedFcn

        prim = PrimDihedFcn(2.0, 0.0, 1)
        # CptEne(0°) = 2*(1+cos0) = 4
        self.assertAlmostEqual(float(prim.CptEne(0.0)), 4.0)


# ---------------------------------------------------------------------------
# Parametrization defaults isolation
# ---------------------------------------------------------------------------


class TestRecipeDefaultsIsolation(unittest.TestCase):
    def test_fresh_defaults_not_shared(self):
        from ligandparam.parametrization import fresh_recipe_defaults

        a = fresh_recipe_defaults()
        b = fresh_recipe_defaults()
        a["leaprc"].append("leaprc.protein.ff14SB")
        self.assertNotIn("leaprc.protein.ff14SB", b["leaprc"])
        a["theory"]["low"] = "X"
        self.assertNotEqual(b["theory"]["low"], "X")


if __name__ == "__main__":
    unittest.main()
