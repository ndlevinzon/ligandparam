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
import json
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
        from ligandparam.recipes.Registry import _REGISTRY, available_recipes

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
        from ligandparam.recipes.Registry import get_recipe

        with self.assertRaises(ValueError) as ctx:
            get_recipe("not-a-recipe")
        self.assertIn("Unknown recipe", str(ctx.exception))


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
        from ligandparam.recipes.FreeLigand import FreeLigand
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
        from ligandparam.recipes.FreeLigand import FreeLigand

        with self.assertRaises(KeyError):
            FreeLigand("ligand.pdb", "out_dir")

    def test_freeligand_bad_orientation_protocol(self):
        from ligandparam.recipes.FreeLigand import FreeLigand

        with self.assertRaises(ValueError):
            FreeLigand(
                "ligand.pdb",
                "out_dir",
                net_charge=0,
                orientation_protocol="not_a_protocol",
            )

    def test_lazyligand_setup(self):
        from ligandparam.recipes.LazyLigand import LazyLigand
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
        from ligandparam.recipes.LazierLigand import LazierLigand
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
        from ligandparam.recipes.DpLazyLigand import DPLigand
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
        from ligandparam.recipes.DpFreeLigand import DPFreeLigand
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
        from ligandparam.recipes.OptLigand import SQMLigand
        from ligandparam.stages import StageInitialize, StageLazyResp
        from ligandparam.stages.DeepMd import DPMinimize

        with tempfile.TemporaryDirectory() as td:
            inp, cwd = self._tmp_recipe_args(td)
            recipe = SQMLigand(inp, cwd, net_charge=0, logger="stream")
            recipe.setup()
            types = [type(s) for s in recipe.stages]
            self.assertEqual(types[0], StageInitialize)
            self.assertIn(StageLazyResp, types)
            self.assertNotIn(DPMinimize, types)
            self._assert_tail_parmchk_leap(recipe.stages)

    def test_lazierligand_execute_forwards_overrides(self):
        """LazierLigand must forward dry_run/nproc/mem (not hardcode 1/1)."""
        from ligandparam.recipes.LazierLigand import LazierLigand

        with tempfile.TemporaryDirectory() as td:
            inp, cwd = self._tmp_recipe_args(td)
            recipe = LazierLigand(inp, cwd, net_charge=0, nproc=4, logger="stream")
            recipe.setup()
            seen = []

            def _capture(stage):
                def _exec(*, dry_run=False, nproc=None, mem=None):
                    seen.append((dry_run, nproc, mem))

                return _exec

            for stage in recipe.stages:
                stage.execute = _capture(stage)
            recipe.execute(dry_run=True, nproc=8, mem=16)
            self.assertTrue(seen)
            self.assertTrue(all(t == (True, 8, 16) for t in seen))

    def test_dihed_correct_appends_twist_stage(self):
        from ligandparam.recipes.FreeLigand import FreeLigand
        from ligandparam.stages.FfpoptDihed import StageDihedTwistCorrection

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
        from ligandparam.recipes.LazierLigand import LazierLigand

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

    def test_every_registry_recipe_uses_common_builders(self):
        """Each registry entry builds stages via recipes.common (smoke)."""
        from ligandparam.recipes.Registry import _REGISTRY, get_recipe

        for name, path_cls in _REGISTRY.items():
            mod_path = path_cls.split(":")[0]
            mod = importlib.import_module(mod_path)
            src = Path(mod.__file__).read_text(encoding="utf-8")
            self.assertIn(
                "ligandparam.recipes.Common",
                src,
                f"{name} should import recipes.common builders",
            )
            with tempfile.TemporaryDirectory() as td:
                inp, cwd = self._tmp_recipe_args(td)
                recipe = get_recipe(
                    name,
                    in_filename=str(inp),
                    cwd=str(cwd),
                    net_charge=0,
                    logger="stream",
                )
                recipe.setup()
                self.assertGreater(len(recipe.stages), 0, f"{name} setup() empty")
                self._assert_tail_parmchk_leap(recipe.stages)


# ---------------------------------------------------------------------------
# Stages — charge normalize + abstract contracts
# ---------------------------------------------------------------------------


class TestStageChargeNormalize(unittest.TestCase):
    def _stage(self, net_charge=0, precision=0.001, decimals=3):
        from ligandparam.stages.Charge import StageNormalizeCharge

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
        from ligandparam.recipes.Common import charge_update_parmchk_leap_stages
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

    def test_init_normalize_center_and_gaussian_kwargs(self):
        _require_rdkit(self)
        from ligandparam.recipes.Common import (
            gaussian_runtime_kwargs,
            init_normalize_center_stages,
            rotation_stage_kwargs,
        )
        from ligandparam.stages import StageDisplaceMol, StageInitialize, StageNormalizeCharge

        recipe = SimpleNamespace(
            in_filename=Path("lig.pdb"),
            cwd=Path("."),
            net_charge=0,
            logger=None,
            kwargs={},
            nproc=2,
            mem=4,
            gaussian_root=None,
            gauss_exedir=None,
            gaussian_binary=None,
            gaussian_scratch=None,
            force_gaussian_rerun=False,
            orientation_protocol="so3_n28",
            theory={"low": "HF/6-31G*", "high": "PBE1PBE/6-31G*"},
        )
        stages = init_normalize_center_stages(
            recipe=recipe,
            initial_mol2="i.mol2",
            centered_out="c.mol2",
        )
        self.assertEqual(
            [type(s) for s in stages],
            [StageInitialize, StageNormalizeCharge, StageDisplaceMol],
        )
        gkw = gaussian_runtime_kwargs(recipe)
        self.assertEqual(gkw["nproc"], 2)
        self.assertEqual(gkw["mem"], 4)
        self.assertIn("force_gaussian_rerun", gkw)
        rkw = rotation_stage_kwargs(recipe)
        self.assertEqual(rkw["orientation_protocol"], "so3_n28")


class TestAbstractStageTemplate(unittest.TestCase):
    def test_execute_calls_run_and_tracks_new_files(self):
        from ligandparam.stages.AbstractStage import AbstractStage

        class _Tiny(AbstractStage):
            def _run(self, dry_run=False, nproc=None, mem=None):
                self.seen = (dry_run, nproc, mem)
                (self.cwd / "created.txt").write_text("x", encoding="utf-8")
                return "ok"

        with tempfile.TemporaryDirectory() as td:
            cwd = Path(td)
            stage = _Tiny("Tiny", cwd / "in.pdb", cwd, logger=MagicMock())
            stage.required = []
            out = stage.execute(dry_run=True, nproc=3, mem=7)
            self.assertEqual(out, "ok")
            self.assertEqual(stage.seen, (True, 3, 7))
            self.assertIn("created.txt", stage.new_files)


class TestWavefrontMixinHelpers(unittest.TestCase):
    def test_precheck_geometry_clash_reports_error(self):
        from ffpopt.scan.WavefrontMixins import precheck_geometry_clash

        def boom():
            raise RuntimeError("bad geom")

        err = precheck_geometry_clash(get_atoms=boom, bonds=[], min_dist=0.8)
        self.assertIsNotNone(err)
        self.assertIn("precheck_error", err)

    def test_covalent_geometry_error_flags_flying_hydrogen(self):
        import numpy as np
        from ffpopt.geom.Constraints import covalent_geometry_error

        pos = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        err = covalent_geometry_error(pos, [(0, 1)], numbers=[6, 1])
        self.assertIsNotNone(err)
        self.assertIn("flew off", err)

        ok = covalent_geometry_error(
            np.array([[0.0, 0.0, 0.0], [1.09, 0.0, 0.0]]),
            [(0, 1)],
            numbers=[6, 1],
        )
        self.assertIsNone(ok)

        nan = covalent_geometry_error(
            np.array([[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0]]),
            [(0, 1)],
            numbers=[6, 1],
        )
        self.assertIn("non-finite", nan)

    def test_precheck_skips_broken_covalent_bond(self):
        import numpy as np
        from ffpopt.scan.WavefrontMixins import precheck_geometry_clash

        class _Atoms:
            def get_positions(self):
                return np.array([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]])

            def get_atomic_numbers(self):
                return np.array([6, 1])

        err = precheck_geometry_clash(get_atoms=_Atoms, bonds=[(0, 1)])
        self.assertEqual(err, "broken_geometry")

    def test_replace_node_with_pickle_noop_when_missing(self):
        from ffpopt.scan.WavefrontMixins import replace_node_with_pickle

        node = SimpleNamespace(node_pkl=Path("definitely-missing-node.pkl"), node_id=1, los="keep")
        replace_node_with_pickle(node)
        self.assertEqual(node.los, "keep")


class TestScissionHelpers(unittest.TestCase):
    def test_safe_name_and_param_key(self):
        from scission.Frcmod import _normalize_param_name_to_key
        from scission.Writers import safe_name

        self.assertEqual(safe_name("foo/bar"), "foo_bar")
        self.assertIsNone(_normalize_param_name_to_key("not_a_dihe"))
        key = _normalize_param_name_to_key("LIG_ca-ca-c-o")
        self.assertEqual(len(key), 4)


# ---------------------------------------------------------------------------
# Packaged FFPOPT_* defaults JSON
# ---------------------------------------------------------------------------


class TestEnvDefaults(unittest.TestCase):
    def tearDown(self):
        from ffpopt.runtime.EnvDefaults import clear_defaults_cache

        clear_defaults_cache()

    def test_packaged_json_loads_and_is_the_store(self):
        from ffpopt.runtime.EnvDefaults import (
            defaults_path,
            env_bool,
            env_int,
            env_str,
            env_value,
            packaged_defaults,
        )

        path = defaults_path()
        self.assertTrue(path.is_file(), path)
        data = packaged_defaults()
        self.assertFalse(data["FFPOPT_FAST_WAVEFRONT"])
        self.assertEqual(data["FFPOPT_MIN_WF_NPROC"], 2)
        self.assertIsNone(data["FFPOPT_ASE_FIRST"])
        self.assertEqual(data["FFPOPT_XTB_GUESS"], "eeq")
        self.assertTrue(data["FFPOPT_BOND_BATCH"])
        self.assertEqual(data["FFPOPT_FIT_MODE"], "barrier")
        with patch.dict(os.environ, {}, clear=False):
            for key in (
                "FFPOPT_FAST_WAVEFRONT",
                "FFPOPT_DEFAULTS",
                "FFPOPT_MIN_WF_NPROC",
                "FFPOPT_XTB_GUESS",
            ):
                os.environ.pop(key, None)
            self.assertFalse(env_bool("FFPOPT_FAST_WAVEFRONT"))
            self.assertEqual(env_int("FFPOPT_MIN_WF_NPROC"), 2)
            self.assertIsNone(env_value("FFPOPT_ASE_FIRST"))
            self.assertEqual(env_str("FFPOPT_XTB_GUESS"), "eeq")

    def test_export_overrides_json(self):
        from ffpopt.runtime.EnvDefaults import env_bool, env_int, env_value

        with patch.dict(
            os.environ,
            {
                "FFPOPT_FAST_WAVEFRONT": "1",
                "FFPOPT_MIN_WF_NPROC": "8",
                "FFPOPT_ASE_FIRST": "0",
            },
            clear=False,
        ):
            os.environ.pop("FFPOPT_DEFAULTS", None)
            self.assertTrue(env_bool("FFPOPT_FAST_WAVEFRONT"))
            self.assertEqual(env_int("FFPOPT_MIN_WF_NPROC"), 8)
            self.assertFalse(env_value("FFPOPT_ASE_FIRST"))

    def test_overlay_file_then_export_wins(self):
        from ffpopt.runtime.EnvDefaults import clear_defaults_cache, env_bool, env_int

        with tempfile.TemporaryDirectory() as td:
            overlay = Path(td) / "mine.json"
            overlay.write_text(
                '// overlay\n{"FFPOPT_MIN_WF_NPROC": 4, "FFPOPT_FAST_WAVEFRONT": true}\n',
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"FFPOPT_DEFAULTS": str(overlay), "FFPOPT_FAST_WAVEFRONT": "0"},
                clear=False,
            ):
                os.environ.pop("FFPOPT_MIN_WF_NPROC", None)
                clear_defaults_cache()
                self.assertEqual(env_int("FFPOPT_MIN_WF_NPROC"), 4)
                self.assertFalse(env_bool("FFPOPT_FAST_WAVEFRONT"))

    def test_json_keys_cover_user_ffpopt_env(self):
        import ast
        import re
        from ffpopt.runtime.EnvDefaults import packaged_defaults

        allow = {"FFPOPT_IN_SPAWN_WORKER", "FFPOPT_DEFAULTS"}
        keys = set(packaged_defaults())
        name_re = re.compile(r"^FFPOPT_[A-Z0-9_]+$")
        found: set[str] = set()
        root = Path(__file__).resolve().parents[1] / "src"
        for path in root.rglob("*.py"):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, SyntaxError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                    continue
                if name_re.match(node.value):
                    found.add(node.value)
        missing = found - keys - allow
        self.assertEqual(
            missing,
            set(),
            f"FFPOPT_* used in code but missing from env_defaults.json: {missing}",
        )


# ---------------------------------------------------------------------------
# Logging / console contracts
# ---------------------------------------------------------------------------


class TestLoggingContracts(unittest.TestCase):
    def setUp(self):
        from ffpopt.runtime import Console as console_mod

        console_mod._BANNER_PRINTED = False
        os.environ.pop("LIGANDPARAM_BANNER_PRINTED", None)

    def tearDown(self):
        from ffpopt.runtime import Console as console_mod

        console_mod._BANNER_PRINTED = False
        os.environ.pop("LIGANDPARAM_BANNER_PRINTED", None)

    def test_format_console_line_peels_scopes(self):
        from ffpopt.runtime.Console import format_console_line

        line = format_console_line("[frag-twist] hello", tag="ffpopt:fragment_1")
        self.assertRegex(line, r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} ")
        self.assertIn("[ffpopt:fragment_1]", line)
        self.assertIn("[frag-twist]", line)
        self.assertIn("hello", line)
        self.assertTrue(line.endswith("\n"))

    def test_ascii_for_stdio_maps_common_symbols(self):
        from ffpopt.runtime.Console import ascii_for_stdio, format_console_line

        self.assertEqual(ascii_for_stdio("k=500 +/-0.5 deg"), "k=500 +/-0.5 deg")
        self.assertEqual(ascii_for_stdio("k=500 \u00b10.5\u00b0"), "k=500 +/-0.5 deg")
        self.assertEqual(ascii_for_stdio("shape-match \u03c7\u00b2"), "shape-match chi^2")
        self.assertEqual(ascii_for_stdio("HL \u2192 LL"), "HL -> LL")
        mapped = ascii_for_stdio("ok")
        self.assertTrue(mapped.isascii())

        line = format_console_line("[affdo] band \u00b10.5\u00b0", tag="ffpopt")
        self.assertTrue(line.isascii())
        self.assertIn("+/-0.5 deg", line)

    def test_print_affdo_strips_non_ascii(self):
        from ffpopt.affdo.AffdoLog import print_affdo

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            print_affdo("soft band \u00b10.5\u00b0")
        out = buf.getvalue()
        self.assertTrue(out.isascii())
        self.assertIn("[affdo] soft band +/-0.5 deg", out)

    def test_affdo_scope_peels_on_console(self):
        from ffpopt.runtime.Console import format_console_line

        line = format_console_line("[affdo] extras: whole_ligand=True", tag="ffpopt")
        self.assertIn("[ffpopt]", line)
        self.assertIn("[affdo]", line)
        self.assertIn("extras: whole_ligand=True", line)

    def test_attach_console_handlers_idempotent(self):
        from ffpopt.runtime.Console import attach_console_handlers

        logger = logging.getLogger("test.dev.console")
        logger.handlers.clear()
        attach_console_handlers(logger, tag="ligandparam")
        n = len(logger.handlers)
        attach_console_handlers(logger, tag="ligandparam")
        self.assertEqual(len(logger.handlers), n)
        logger.handlers.clear()

    def test_set_stream_logger_tags_ligandparam(self):
        from ligandparam.Log import set_stream_logger

        logger = set_stream_logger()
        markers = [
            getattr(h, "_lp_console_marker", None) for h in logger.handlers
        ]
        self.assertTrue(any(m and "ligandparam" in m for m in markers if m))

    def test_banner_not_from_attach(self):
        from ffpopt.runtime import Console as console_mod

        logger = logging.getLogger("test.dev.banner")
        logger.handlers.clear()
        console_mod.attach_console_handlers(logger, tag="x")
        self.assertFalse(console_mod._BANNER_PRINTED)
        logger.handlers.clear()


# ---------------------------------------------------------------------------
# AFFDO-style extras — logging helpers + pure scoring
# ---------------------------------------------------------------------------


class TestAffdoLogging(unittest.TestCase):
    def test_describe_affdo_extras_default_and_full(self):
        from ffpopt.affdo.AffdoLog import describe_affdo_extras

        default = describe_affdo_extras()
        self.assertIn("whole_ligand=False", default)
        self.assertIn("multi_centroid=0", default)
        self.assertIn("fit_flags=(barrier / default)", default)
        self.assertNotIn("k=", default)

        full = describe_affdo_extras(
            whole_ligand=True,
            multi_centroid=5,
            boltzmann_charges=True,
            soft_dihed_restraint=True,
            soft_dihed_k=400.0,
            soft_dihed_tol=0.25,
            fit_cli_args=["--fit-full", "--fit-backend", "jax"],
        )
        self.assertIn("whole_ligand=True", full)
        self.assertIn("multi_centroid=5", full)
        self.assertIn("boltzmann_charges=True", full)
        self.assertIn("k=400", full)
        self.assertIn("tol=0.25 deg", full)
        self.assertIn("--fit-full", full)
        self.assertIn("jax", full)

    def test_fit_backend_jax_falls_back_without_jax(self):
        from argparse import Namespace
        from unittest.mock import patch
        from ffpopt.dihed.ExtendedFit import apply_fit_flags_to_args, resolve_fit_backend

        with patch("ffpopt.dihed.ExtendedFit.jax_is_available", return_value=False):
            self.assertEqual(resolve_fit_backend("jax"), "lbfgsb")
            args = Namespace(
                fit_mode="full",
                fit_backend="jax",
                fit_full=True,
                barrier_only=False,
                fit_phases=False,
                fit_periods=False,
                fit_scee_scnb=False,
                opt_phase=False,
                opt_periods=False,
                opt_scee_scnb=False,
                scee=None,
                scnb=None,
            )
            apply_fit_flags_to_args(args)
            self.assertEqual(args.fit_backend, "lbfgsb")
            self.assertEqual(args.fit_mode, "full")

    def test_existing_scan_grid_mismatch(self):
        import json
        import tempfile
        from pathlib import Path
        from ffpopt.workflows.TwistHelpers import existing_scan_grid_mismatch
        from ffpopt.dihed.ExtendedFit import _lbfgsb_options

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "orig.json"
            p.write_text(json.dumps([{}] * 36))
            self.assertTrue(existing_scan_grid_mismatch(p, 15))
            self.assertFalse(existing_scan_grid_mismatch(p, 10))
            p.write_text(json.dumps({"structs": [{}] * 24}))
            self.assertFalse(existing_scan_grid_mismatch(p, 15))
        opts = _lbfgsb_options(type("A", (), {"nltol": 0.02, "nlmaxiter": 10})())
        self.assertNotIn("disp", opts)
        self.assertEqual(opts["ftol"], 0.02)

    def test_format_boltzmann_summary(self):
        from ffpopt.affdo.AffdoLog import format_boltzmann_summary

        lines = format_boltzmann_summary(
            {
                "out_mol2": "lig.boltz.mol2",
                "out_lib": "lig.boltz.lib",
                "weights": [0.7, 0.3],
                "T": 298.15,
                "n_atom": 12,
                "equal_weights": False,
                "rms_vs_first": 0.01234,
                "max_abs_dq": 0.05,
            }
        )
        text = "\n".join(lines)
        self.assertIn("12 atom charges", text)
        self.assertIn("0.7000 0.3000", text)
        self.assertIn("lig.boltz.mol2", text)
        self.assertIn("lig.boltz.lib", text)
        self.assertNotIn("equal weights", text)

        eq = "\n".join(format_boltzmann_summary({"weights": [0.5, 0.5], "equal_weights": True}))
        self.assertIn("equal weights", eq)

    def test_print_affdo_stdout(self):
        from ffpopt.affdo.AffdoLog import print_affdo

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            print_affdo("soft restraint on")
        self.assertEqual(buf.getvalue(), "[affdo] soft restraint on\n")

    def test_pick_smoothest_profile_logs_details(self):
        from ffpopt.affdo.CentroidProfiles import pick_smoothest_profile, score_profile_details

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            smooth = root / "c0.dat"
            jagged = root / "c1.dat"
            missing = root / "c2.dat"
            # Smooth cosine-like scan.
            smooth.write_text(
                "\n".join(f"{a} {0.5 * (1.0 - __import__('math').cos(__import__('math').radians(a)))} 0"
                          for a in range(-180, 181, 30)),
                encoding="utf-8",
            )
            jagged.write_text(
                "\n".join(
                    f"{a} {(0.0 if i % 2 == 0 else 8.0)} 0"
                    for i, a in enumerate(range(-180, 181, 30))
                ),
                encoding="utf-8",
            )
            best, score, rows = pick_smoothest_profile([smooth, jagged, missing])
            self.assertEqual(Path(best), smooth)
            self.assertTrue(score < rows[1]["score"] or Path(rows[0]["path"]) == smooth)
            names = {Path(r["path"]).name: r for r in rows}
            self.assertEqual(names["c2.dat"]["error"], "missing")
            self.assertIsNone(names["c0.dat"]["error"])
            d0 = score_profile_details(smooth)
            self.assertGreater(d0["npts"], 3)
            self.assertTrue(__import__("math").isfinite(d0["fourier"]))

    def test_profile_is_smooth_enough_uses_fourier(self):
        import math
        from ffpopt.affdo.CentroidProfiles import (
            profile_is_smooth_enough,
            score_profile_details,
        )

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            smooth = root / "smooth.dat"
            jagged = root / "jagged.dat"
            barrier = root / "barrier.dat"
            angs = list(range(-180, 181, 30))
            smooth.write_text(
                "\n".join(
                    f"{a} {0.5 * (1.0 - math.cos(math.radians(a)))} 0" for a in angs
                ),
                encoding="utf-8",
            )
            jagged.write_text(
                "\n".join(
                    f"{a} {(0.0 if i % 2 == 0 else 8.0)} 0"
                    for i, a in enumerate(angs)
                ),
                encoding="utf-8",
            )
            barrier.write_text(
                "\n".join(
                    f"{a} {3.0 * (1.0 - math.cos(math.radians(2 * a)))} 0" for a in angs
                ),
                encoding="utf-8",
            )
            self.assertTrue(profile_is_smooth_enough(score_profile_details(smooth), fourier_max=0.5))
            self.assertTrue(profile_is_smooth_enough(score_profile_details(barrier), fourier_max=0.5))
            self.assertFalse(profile_is_smooth_enough(score_profile_details(jagged), fourier_max=0.5))
            self.assertFalse(profile_is_smooth_enough(score_profile_details(smooth), fourier_max=None))

    def test_hl_orig_and_centroid_jobs_share_pools(self):
        from ffpopt.workflows import TwistHelpers as th

        class _Scan:
            def __init__(self, idxs):
                self.idxs = list(idxs)

            def GetIdxStr(self):
                return "-".join(str(i) for i in self.idxs)

        scans = [_Scan([0, 1, 2, 3]), _Scan([4, 5, 6, 7])]
        captured = []

        def _fake_execute(jobs, **kwargs):
            captured.append({"label": kwargs.get("label"), "jobs": list(jobs)})
            return [
                (j["prefix"], tuple(j["dihed_idxs"]), None) for j in jobs
            ]

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            starts = [root / "c0.json", root / "c1.json", root / "c2.json"]
            for p in starts:
                p.write_text("{}", encoding="utf-8")
            angs = list(range(-180, 181, 30))
            import math

            def _write_dat(name, jagged=False):
                path = root / name
                if jagged:
                    text = "\n".join(
                        f"{a} {(0.0 if i % 2 == 0 else 8.0)} 0"
                        for i, a in enumerate(angs)
                    )
                else:
                    text = "\n".join(
                        f"{a} {0.5 * (1.0 - math.cos(math.radians(a)))} 0" for a in angs
                    )
                path.write_text(text, encoding="utf-8")

            _write_dat("xtb.c0_0-1-2-3.dat", jagged=False)
            _write_dat("xtb.c0_4-5-6-7.dat", jagged=True)

            with patch.object(th, "_execute_bond_scan_jobs", side_effect=_fake_execute), patch(
                "ffpopt.affdo.CentroidProfiles.generate_centroid_start_jsons",
                return_value=starts,
            ):
                th._run_hl_and_orig_scans(
                    scans,
                    hl_prefix="xtb",
                    hl_model="xtb",
                    inp=str(root / "start.json"),
                    nproc=8,
                    skip_existing=True,
                    workdir=root,
                    logger=None,
                    wf_kwargs={},
                    multi_centroid=3,
                )
            self.assertTrue((root / "xtb_0-1-2-3.dat").is_file())
            self.assertTrue((root / "xtb_4-5-6-7.dat").is_file())

        self.assertEqual(len(captured), 2)
        first_prefixes = {j["prefix"] for j in captured[0]["jobs"]}
        self.assertEqual(first_prefixes, {"xtb.c0", "orig"})
        self.assertEqual(len(captured[0]["jobs"]), 4)
        extra_prefixes = {j["prefix"] for j in captured[1]["jobs"]}
        self.assertEqual(extra_prefixes, {"xtb.c1", "xtb.c2"})
        extra_idxs = {tuple(j["dihed_idxs"]) for j in captured[1]["jobs"]}
        self.assertEqual(extra_idxs, {(4, 5, 6, 7)})

    def test_interleave_job_groups(self):
        from ffpopt.workflows.TwistHelpers import _interleave_job_groups

        a = [{"id": "a0"}, {"id": "a1"}]
        b = [{"id": "b0"}, {"id": "b1"}, {"id": "b2"}]
        ids = [j["id"] for j in _interleave_job_groups(a, b)]
        self.assertEqual(ids, ["a0", "b0", "a1", "b1", "b2"])
        self.assertEqual(_interleave_job_groups([], a), a)

    def test_boltzmann_average_summary_fields(self):
        from ffpopt.affdo.BoltzmannCharges import boltzmann_average_mol2_charges

        def _mol2(charges):
            lines = [
                "@<TRIPOS>MOLECULE\n",
                "LIG\n",
                "@<TRIPOS>ATOM\n",
            ]
            for i, q in enumerate(charges, start=1):
                name = f"C{i}"
                lines.append(
                    f"{i:7d} {name:<8s} {0.0:10.4f} {0.0:10.4f} {0.0:10.4f} "
                    f"{'C.3':<6s} {'1':>4s} {'LIG':<6s} {q:10.6f}\n"
                )
            lines.append("@<TRIPOS>BOND\n")
            return "".join(lines)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            a = root / "c0.mol2"
            b = root / "c1.mol2"
            a.write_text(_mol2([0.1, -0.1]), encoding="utf-8")
            b.write_text(_mol2([0.3, -0.3]), encoding="utf-8")
            out = root / "avg.mol2"
            info = boltzmann_average_mol2_charges([a, b], [0.0, 0.0], out)
            self.assertTrue(info["equal_weights"])
            self.assertEqual(info["n_atom"], 2)
            self.assertEqual(info["n_conf"], 2)
            self.assertAlmostEqual(info["charges"][0], 0.2, places=6)
            self.assertGreater(info["rms_vs_first"], 0.0)

    def test_format_extended_params(self):
        from ffpopt.dihed.ExtendedFit import format_extended_params

        prim = SimpleNamespace(fc=1.25, phase=180.0, per=2)
        finp = SimpleNamespace(
            ptypedict={"ca-c3-c-o": SimpleNamespace(dfcns=SimpleNamespace(prims=[prim]))},
            opt_phase=True,
            opt_periods=True,
            opt_scee_scnb=True,
            scee=1.2,
            scnb=2.0,
        )
        text = "\n".join(format_extended_params(finp))
        self.assertIn("ca-c3-c-o", text)
        self.assertIn("FC=[1.2500]", text)
        self.assertIn("phase_deg=[180.00]", text)
        self.assertIn("period=[2]", text)
        self.assertIn("scee=1.2000", text)


# ---------------------------------------------------------------------------
# ligandparam I/O — amber bundle resolution
# ---------------------------------------------------------------------------


class TestAmberBundleIO(unittest.TestCase):
    def _touch_triplet(self, work_dir: Path, stem: str) -> None:
        (work_dir / f"{stem}.mol2").write_text("@<TRIPOS>MOLECULE\n", encoding="utf-8")
        (work_dir / f"{stem}.lib").write_text("!entry\n", encoding="utf-8")
        (work_dir / f"{stem}.frcmod").write_text("Remark line\n", encoding="utf-8")

    def test_resolve_explicit_paths(self):
        from ligandparam.io.AmberBundle import AmberLigandBundle, resolve_getparam_bundle

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
        from ligandparam.io.AmberBundle import resolve_getparam_bundle

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
        from ligandparam.io.AmberBundle import resolve_getparam_bundle

        with tempfile.TemporaryDirectory() as td:
            cwd = Path(td)
            (cwd / "A" / "B").mkdir(parents=True)
            with self.assertRaises(FileNotFoundError):
                resolve_getparam_bundle(
                    cwd=cwd, data_cwd="A", resname="B", label="x"
                )

    def test_to_scission_input(self):
        from ligandparam.io.AmberBundle import resolve_getparam_bundle

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
        from ligandparam.recipes.DihedOptions import apply_dihed_options, pop_dihed_options

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
        from ligandparam.recipes.DihedOptions import coerce_fragment_config
        from scission.Models import FragmentConfig

        self.assertIsNone(coerce_fragment_config(None))
        cfg = coerce_fragment_config({"angle_step": 15, "cap_strategy": "hydrogen"})
        self.assertIsInstance(cfg, FragmentConfig)
        self.assertEqual(cfg.angle_step, 15)
        with self.assertRaises(TypeError):
            coerce_fragment_config("nope")

    def test_normalize_bond_pairs0(self):
        from ffpopt.workflows import normalize_bond_pairs0

        self.assertEqual(normalize_bond_pairs0([(1, 2), [3, 4]]), [(1, 2), (3, 4)])
        self.assertEqual(normalize_bond_pairs0(["0,1", "10,11"]), [(0, 1), (10, 11)])
        with self.assertRaises(ValueError):
            normalize_bond_pairs0(["0-1"])
        with self.assertRaises(TypeError):
            normalize_bond_pairs0([42])

    def test_bonds0_from_scission_fit_torsions(self):
        from ffpopt.workflows import bonds0_from_scission_fit_torsions

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
        from scission.Models import Atom, Bond, Ligand

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
        from scission.Torsions import enumerate_torsions, find_rotatable_bonds

        lig = self._butane_like_ligand()
        rots = find_rotatable_bonds(lig)
        self.assertGreaterEqual(len(rots), 1)
        tors = enumerate_torsions(lig)
        self.assertGreaterEqual(len(tors), 1)
        for t in tors:
            self.assertEqual(len(t.atom_indices), 4)
            self.assertEqual(len(t.bond), 2)

    def test_fragment_config_defaults_and_from_dict(self):
        from scission.Models import FragmentConfig

        cfg = FragmentConfig()
        self.assertTrue(hasattr(cfg, "angle_step") or hasattr(cfg, "cap_strategy"))
        cfg2 = FragmentConfig.from_dict({"angle_step": 15})
        self.assertEqual(cfg2.angle_step, 15)

    def test_write_fragment_index_and_merge_accumulate(self):
        from scission.Merge import _load_fragment_update
        from scission.Models import SelectedFragment
        from scission.Writers import write_fragment_index

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
            # Later iteration overwrites shared key; earlier-only key survives.
            n_lines = update["dihe_groups"][("c3", "c3", "c3", "n")]
            self.assertTrue(any("3.50" in ln for ln in n_lines))
            self.assertFalse(any("2.00" in ln for ln in n_lines))
            c3_lines = update["dihe_groups"][("c3", "c3", "c3", "c3")]
            self.assertTrue(any("1.00" in ln for ln in c3_lines))

    def test_merge_dihe_empty_later_iteration_keeps_earlier(self):
        """An empty later itXX.frcmod must not wipe earlier DIHE accumulation."""
        from scission.Merge import _load_fragment_update

        def _frcmod(lines):
            return (
                "Remark line goes here\nMASS\n\nBOND\n\nANGLE\n\nDIHE\n"
                + "".join(f"{ln}\n" for ln in lines)
                + "\nIMPROPER\n\nNONB\n\n"
            )

        with tempfile.TemporaryDirectory() as td:
            frag_dir = Path(td) / "frag_0001"
            frag_dir.mkdir()
            (frag_dir / "it01.frcmod").write_text(
                _frcmod(["c3-c3-c3-c3 1 1.00 0.0 1."])
            )
            (frag_dir / "it02.frcmod").write_text(_frcmod([]))
            update = _load_fragment_update(frag_dir)
            self.assertIn(("c3", "c3", "c3", "c3"), update["dihe_groups"])

    def test_merge_two_fragments_same_scanned_bytype_key(self):
        """bytype collisions: keep first scanned fragment, do not abort."""
        import warnings

        from scission.Merge import MergeWarning, merge_fragment_frcmods

        def _frcmod(lines):
            return (
                "Remark line goes here\nMASS\n\nBOND\n\nANGLE\n\nDIHE\n"
                + "".join(f"{ln}\n" for ln in lines)
                + "\nIMPROPER\n\nNONB\n\n"
            )

        def _fit_json(param: str):
            return json.dumps(
                {
                    "params": {param: {"nprim": 1}},
                    "systems": [
                        {
                            "params": {param: {"nprim": 1}},
                            "profiles": [{"plots": [param]}],
                        }
                    ],
                }
            )

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            parent = root / "parent.frcmod"
            parent.write_text(_frcmod(["c3-c3-n4-c3 1 0.50 0.0 1."]))
            out = root / "merged.frcmod"
            frag6 = root / "fragment_6"
            frag8 = root / "fragment_8"
            for frag, pk in ((frag6, "1.10"), (frag8, "2.20")):
                frag.mkdir()
                (frag / "it01.frcmod").write_text(
                    _frcmod([f"c3-c3-n4-c3 1 {pk} 0.0 1."])
                )
                (frag / "it01.fit.json").write_text(
                    _fit_json("LIG_c3-c3-n4-c3")
                )
                (frag / "fit_torsions.json").write_text("[]")

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                report = merge_fragment_frcmods(
                    parent_frcmod_path=parent,
                    output_frcmod_path=out,
                    fragment_dirs=[frag6, frag8],
                )
            self.assertTrue(out.is_file())
            self.assertTrue(
                any(issubclass(w.category, MergeWarning) for w in caught)
            )
            self.assertEqual(len(report["conflicts"]), 1)
            self.assertEqual(
                report["conflicts"][0]["resolution"], "first_scanned_wins"
            )
            self.assertIn("1.10", out.read_text())
            self.assertNotIn("2.20", out.read_text())


# ---------------------------------------------------------------------------
# ffpopt — constraints, dihedrals, wavefront policy, runtime helpers
# ---------------------------------------------------------------------------


class TestFfpoptCoreFunctions(unittest.TestCase):
    def test_constraints_to_geometric(self):
        from ffpopt.geom.Constraints import Constraint, to_geometric

        lines = to_geometric([Constraint("dihed", [0, 1, 2, 3], value=45.0)])
        self.assertTrue(any("45.0" in ln and ln.startswith("dihedral") for ln in lines))

    def test_cleanup_geometric_scratch_nsf_and_tmp(self):
        import tempfile
        from pathlib import Path
        from ffpopt.geom.Geometric import (
            cleanup_geometric_scratch,
            geometric_prefix_from_node_pkl,
            sweep_geometric_scratch_dir,
        )
        from ffpopt.scan.WavefrontMixins import cleanup_wavefront_geometric_scratch

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            pkl = td / "level_1_angle_0.0_id_0_node.pckl"
            pkl.write_text("node")
            prefix = geometric_prefix_from_node_pkl(pkl)
            self.assertTrue(prefix.endswith("_geom"))
            tmp = Path(prefix + ".tmp")
            tmp.mkdir()
            (tmp / "junk").write_text("x")
            Path(prefix + ".nsf").write_text("geom log")
            Path(prefix + ".log").write_text("log")
            Path(prefix + "_optim.xyz").write_text("xyz")
            Path(prefix + ".r1.tmp").mkdir()
            (td / "log.nsf").write_text("cwd log")
            (td / "orphan_geom.log").write_text("old")

            n = cleanup_geometric_scratch(prefix, keep_optim=True)
            self.assertGreater(n, 0)
            self.assertFalse(tmp.exists())
            self.assertFalse(Path(prefix + ".nsf").exists())
            self.assertTrue(Path(prefix + "_optim.xyz").exists())

            class _Node:
                node_pkl = str(pkl)
                complete = False

            class _Level:
                nodes = [_Node()]

            class _WF:
                levels = [_Level()]
                workdir = str(td)
                checkpoint = str(td / "checkpoint.pkl")

            cleanup_wavefront_geometric_scratch(_WF(), keep_incomplete_optim=True)
            self.assertTrue(Path(prefix + "_optim.xyz").exists())
            self.assertFalse((td / "log.nsf").exists())

            _Node.complete = True
            cleanup_wavefront_geometric_scratch(_WF(), keep_incomplete_optim=False)
            self.assertFalse(Path(prefix + "_optim.xyz").exists())
            self.assertFalse((td / "orphan_geom.log").exists())
            self.assertEqual(sweep_geometric_scratch_dir(td), 0)

    def test_is_soft_opt_and_evaluate_policy(self):
        from ffpopt.geom.GeomOpt import is_soft_opt_recovery
        from ffpopt.scan.WavefrontMixins import evaluate_wavefront_minimum

        self.assertTrue(is_soft_opt_recovery("loose"))
        self.assertFalse(is_soft_opt_recovery("primary"))
        self.assertTrue(is_soft_opt_recovery("linear-torsion"))
        self.assertTrue(is_soft_opt_recovery("linear-torsion-soft"))
        d = evaluate_wavefront_minimum(
            energy=1.0,
            soft=True,
            has_incumbent=False,
            incumbent_energy=None,
            incumbent_soft=False,
            threshold_ev=0.1,
        )
        self.assertEqual(d["reason"], "soft_first_seed")

    def test_linear_torsion_bend_detection_and_unkink(self):
        """Near-180 deg bend in a constrained dihedral is detected and unkinked."""
        try:
            from ase import Atoms
        except ImportError:
            self.skipTest("ase required")
        import numpy as np
        from ffpopt.geom.Constraints import Constraint
        from ffpopt.geom.LinearTorsion import (
            find_near_linear_bends,
            has_near_linear_dihedral_bend,
            is_linear_torsion_error,
            log_linear_torsion,
            unkink_near_linear_bends,
        )

        # A-B-C nearly linear, C-D off axis -> dihedral A-B-C-D ill-defined.
        atoms = Atoms(
            "CCCC",
            positions=[
                [0.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [3.0, 0.05, 0.0],
                [3.5, 1.0, 0.0],
            ],
        )
        # Make A-B-C almost linear explicitly.
        atoms.set_angle(0, 1, 2, 178.8)
        cons = [Constraint("dihed", [0, 1, 2, 3], value=60.0)]
        self.assertTrue(has_near_linear_dihedral_bend(atoms, cons))
        hits = find_near_linear_bends(atoms, cons)
        self.assertTrue(hits)
        self.assertGreaterEqual(hits[0]["angle_deg"], 175.0)
        unkink_near_linear_bends(atoms, hits, target_deg=170.0)
        self.assertAlmostEqual(atoms.get_angle(0, 1, 2), 170.0, places=1)
        self.assertFalse(has_near_linear_dihedral_bend(atoms, cons))

        # Log helper emits strict UTF-8 bytes (ASCII message body).
        msg = "[ffpopt] linear-torsion unkink 1-2-3: 178.80 deg -> 170.00 deg"
        data = (msg + "\n").encode("utf-8")
        self.assertEqual(data.decode("utf-8"), msg + "\n")
        log_linear_torsion(msg)

        class _E(Exception):
            pass

        class LinearTorsionError(_E):
            pass

        self.assertTrue(
            is_linear_torsion_error(
                LinearTorsionError(
                    "A constrained torsion has three consecutive atoms "
                    "forming a nearly linear angle"
                )
            )
        )
        self.assertFalse(is_linear_torsion_error(ValueError("other")))

    def test_wavefront_policy_matrix(self):
        """Lock spawn / update decisions for soft and hard incumbents."""
        from ffpopt.scan.WavefrontMixins import evaluate_wavefront_minimum

        cases = [
            # energy, soft, has, inc_e, inc_soft, thr, reason, update, active
            (1.0, True, False, None, False, 0.1, "soft_first_seed", True, True),
            (0.9, True, True, 1.0, True, 0.1, "soft_improve", True, False),
            (1.1, True, True, 1.0, True, 0.1, "soft_demoted", False, False),
            (1.0, False, False, None, False, 0.1, "hard_first", True, True),
            (0.9, False, True, 1.0, True, 0.1, "hard_replace_soft", True, True),
            (1.1, False, True, 1.0, True, 0.1, "hard_worse_than_soft", False, False),
            (0.8, False, True, 1.0, False, 0.1, "hard_significant_improve", True, True),
            (0.95, False, True, 1.0, False, 0.1, "hard_quiet_improve", True, False),
            (1.0, False, True, 1.0, False, 0.1, "hard_not_lower", False, False),
            (float("nan"), False, False, None, False, 0.1, "nonfinite", False, False),
        ]
        for energy, soft, has, inc_e, inc_soft, thr, reason, upd, act in cases:
            with self.subTest(reason=reason):
                d = evaluate_wavefront_minimum(
                    energy=energy,
                    soft=soft,
                    has_incumbent=has,
                    incumbent_energy=inc_e,
                    incumbent_soft=inc_soft,
                    threshold_ev=thr,
                )
                self.assertEqual(d["reason"], reason)
                self.assertEqual(d["update_min"], upd)
                self.assertEqual(d["active"], act)

    def test_dihed_math_reexported_and_ipc_slim(self):
        from ffpopt.dihed import DihedMath as dihed_math
        from ffpopt.dihed.Dihedrals import shape_match_delta
        from ffpopt.runtime.SlimIpc import slim_scan_result, slim_twist_result

        self.assertIs(shape_match_delta, dihed_math.shape_match_delta)
        self.assertIsNone(slim_scan_result(None))
        self.assertEqual(slim_scan_result({"a": 1, "wf_run": object()})["a"], 1)
        slim = slim_twist_result(
            {"ok": True, "scans": [("p", (0, 1, 2, 3), {"e": 1.0, "wf_run": object()})]}
        )
        self.assertNotIn("wf_run", slim["scans"][0][2])

    def test_shape_match_and_joint_ls_symbols(self):
        import numpy as np
        from ffpopt.dihed.Dihedrals import shape_match_delta

        hl = np.array([1.0, 2.0, 3.0])
        ll = np.array([0.0, 1.0, 2.0])
        d = shape_match_delta(hl, ll)
        np.testing.assert_allclose(d, shape_match_delta(hl, ll + 9.0))

    def test_align_scan_profiles(self):
        from ffpopt.dihed.Dihedrals import align_scan_profiles
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
        self.assertFalse(info.get("interpolated"))

    def test_align_scan_profiles_interpolates_mismatched_full_grids(self):
        from ffpopt.dihed.Dihedrals import align_scan_profiles
        from ffpopt.Struct import ListOfStruct

        def _frame(ang, e):
            return SimpleNamespace(
                data={
                    "name": f"d{int(ang):03d}",
                    "energy": e,
                    "positions": [[0.0, 0.0, 0.0]],
                    "constraints": [],
                },
                constraints=None,
            )

        hl_angs = list(range(0, 360, 15))
        ll_angs = list(range(0, 360, 10))
        hl = ListOfStruct.from_structs_shared(
            [_frame(a, float(a)) for a in hl_angs]
        )
        ll = ListOfStruct.from_structs_shared(
            [_frame(a, 0.0) for a in ll_angs]
        )
        ahl, all_, info = align_scan_profiles(hl, ll)
        self.assertTrue(info.get("interpolated"))
        self.assertEqual(info["n_common"], 36)
        self.assertEqual(len(ahl.structs), 36)
        self.assertEqual(len(all_.structs), 36)
        e10 = next(
            float(s.data["energy"])
            for s in ahl.structs
            if s.data["name"] == "d010"
        )
        self.assertAlmostEqual(e10, 10.0, places=5)

    def test_fast_presets_keep_delta(self):
        from ffpopt.runtime.FastWavefront import (
            LIBRARY_DEFAULTS,
            apply_fast_wavefront_presets,
        )

        knobs = dict(LIBRARY_DEFAULTS)
        applied = apply_fast_wavefront_presets(knobs, enabled=True)
        self.assertNotIn("delta", applied)
        self.assertEqual(knobs["delta"], 10)

    def test_fast_wavefront_enabled(self):
        from ffpopt.runtime.FastWavefront import fast_wavefront_enabled

        self.assertTrue(fast_wavefront_enabled(explicit=True))
        self.assertFalse(fast_wavefront_enabled(explicit=False))
        with patch.dict(os.environ, {"FFPOPT_FAST_WAVEFRONT": "1"}):
            self.assertTrue(fast_wavefront_enabled())
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FFPOPT_FAST_WAVEFRONT", None)
            self.assertFalse(fast_wavefront_enabled())

    def test_cpu_budget_fair_share(self):
        from ffpopt.runtime.CpuBudget import fair_share_leases

        leases = fair_share_leases(8, ["a", "b", "c"])
        self.assertEqual(sum(leases.values()), 8)
        self.assertEqual(len(leases), 3)

    def test_sander_ll_scan_uses_ase_first_and_prefers_depth(self):
        from ffpopt.workflows.TwistHelpers import _is_sander_ll_model, _wf_kwargs_for_scan_model
        from ffpopt.runtime.FastWavefront import (
            prefer_ase_first_model,
            prefer_bond_pool_depth,
            prefer_fragment_pool_depth,
            prefer_wavefront_depth,
            qdpi2_opt_components,
            split_nproc_for_items,
        )

        self.assertTrue(_is_sander_ll_model("sander"))
        self.assertFalse(_is_sander_ll_model("xtb"))
        kw = _wf_kwargs_for_scan_model("sander", {"nproc": 4, "delta": 10})
        self.assertIs(kw["geometric_opt"], False)
        # Explicit override wins.
        kw2 = _wf_kwargs_for_scan_model("sander", {"geometric_opt": True})
        self.assertIs(kw2["geometric_opt"], True)
        # Sander no longer always prefers depth at the model level.
        self.assertFalse(prefer_wavefront_depth(model="sander", fast=False))
        self.assertFalse(prefer_wavefront_depth(model="qdpi2", fast=False))
        # Tiny lease + multi-bond → bond breadth (concurrent bonds).
        self.assertFalse(
            prefer_bond_pool_depth(model="sander", nproc=3, n_bonds=3)
        )
        # Large lease can keep depth when the caller asks for it.
        self.assertTrue(
            prefer_bond_pool_depth(
                model="sander", nproc=12, n_bonds=3, prefer=True
            )
        )
        # Many fragments on a modest node → fragment breadth.
        self.assertFalse(
            prefer_fragment_pool_depth(
                model="xtb", nproc=8, n_fragments=6, fast=True
            )
        )
        # Flatten nested spawn: never both outer and inner > 1.
        n_outer, n_inner = split_nproc_for_items(8, 4, prefer_depth=False)
        self.assertTrue(n_outer == 1 or n_inner == 1)
        n_outer_d, n_inner_d = split_nproc_for_items(8, 4, prefer_depth=True)
        self.assertTrue(n_outer_d == 1 or n_inner_d == 1)
        self.assertTrue(prefer_ase_first_model("xtb", fast=True))
        self.assertFalse(prefer_ase_first_model("xtb", fast=False))
        with patch.dict(os.environ, {"FFPOPT_FAST_WAVEFRONT": "1"}):
            self.assertEqual(qdpi2_opt_components(), "xtb")
        with patch.dict(os.environ, {"FFPOPT_QDPI2_OPT": "both"}):
            self.assertEqual(qdpi2_opt_components(), "both")

    def test_cpu_budget_clear_leases_on_init(self):
        from ffpopt.runtime.CpuBudget import CpuBudget

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / ".cpu_budget.json"
            b = CpuBudget(path, 8)
            b.lease("fragment_1")
            b.lease("fragment_2")
            self.assertGreater(len(b.snapshot()["leases"]), 0)
            CpuBudget(path, 8, clear_leases=True)
            self.assertEqual(CpuBudget(path, 8).snapshot()["leases"], {})

    def test_fragment_twist_done_sentinel(self):
        from ffpopt.workflows import (
            clear_fragment_twist_done,
            is_fragment_twist_done,
            mark_fragment_twist_done,
        )

        with tempfile.TemporaryDirectory() as td:
            frag = Path(td) / "fragment_1"
            frag.mkdir()
            self.assertFalse(is_fragment_twist_done(frag))
            mark_fragment_twist_done(frag)
            self.assertTrue(is_fragment_twist_done(frag))
            clear_fragment_twist_done(frag)
            self.assertFalse(is_fragment_twist_done(frag))

    def test_read_last_optim_xyz_warm_start_helper(self):
        """Interrupted geomopt leaves ``_optim.xyz``; helper reads last frame."""
        import numpy as np
        from ffpopt.geom.Geometric import read_last_optim_xyz, write_plain_xyz

        try:
            import ase
            from ase import Atoms
        except ImportError:
            self.skipTest("ase required")

        with tempfile.TemporaryDirectory() as td:
            prefix = Path(td) / "node_geom"
            atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
            write_plain_xyz(str(prefix) + ".xyz", atoms)
            atoms2 = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.90]])
            # geomeTRIC-style trajectory name
            write_plain_xyz(str(prefix) + "_optim.xyz", atoms2)
            last = read_last_optim_xyz(prefix)
            self.assertIsNotNone(last)
            np.testing.assert_allclose(last[1, 2], 0.90)

    def test_pack_rotatable_bond_batches_conservative(self):
        from ffpopt.workflows.BondBatches import (
            adjacency_from_topology_bonds,
            pack_rotatable_bond_batches,
            should_batch_bonds,
        )

        # Linear chain 0-1-2-3-4-5-6-7: rotatable centrals (0,1),(2,3),(4,5),(6,7)
        topo = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]
        adj = adjacency_from_topology_bonds(topo)
        bonds = [(0, 1), (2, 3), (4, 5), (6, 7)]
        self.assertTrue(should_batch_bonds(len(bonds), max_batch=2))
        batches = pack_rotatable_bond_batches(
            bonds, adj, max_batch=2, couple_radius=2
        )
        # All bonds land in some batch; no empty batches; size capped at 2.
        flat = [b for batch in batches for b in batch]
        self.assertEqual(len(flat), 4)
        self.assertTrue(all(len(batch) <= 2 for batch in batches))
        # Nearby (0,1) and (2,3) should prefer the same or adjacent batches.
        self.assertGreaterEqual(len(batches), 2)

        # Two distant bonds → can be separate components but still ≤ max_batch each.
        far = pack_rotatable_bond_batches(
            [(0, 1), (6, 7)], adj, max_batch=2, couple_radius=1
        )
        self.assertEqual(sum(len(b) for b in far), 2)

    def test_split_nproc_for_items(self):
        from ffpopt.runtime.FastWavefront import split_nproc_for_items
        from ffpopt.workflows.TwistHelpers import _split_fragment_nproc

        self.assertEqual(_split_fragment_nproc(8, 4), split_nproc_for_items(8, 4))
        n_outer, n_inner = split_nproc_for_items(8, 4)
        # Flattened: never nest both axes; product may be < nproc.
        self.assertTrue(n_outer == 1 or n_inner == 1)
        self.assertLessEqual(n_outer * n_inner, 8)
        n_outer2, n_inner2 = split_nproc_for_items(
            8, 4, flatten_nested=False
        )
        self.assertEqual(n_outer2 * n_inner2, 8)

    def test_pickle_compat_alias(self):
        from ffpopt.scan.WavefrontMixins import register_wavefront_pickle_aliases
        from ffpopt.scan.WaveFront import Wavefront

        register_wavefront_pickle_aliases()
        import ffpopt.WaveFront as legacy

        self.assertIs(legacy.Wavefront, Wavefront)

    def test_prim_dihed_energy_term(self):
        from ffpopt.dihed.Dihedrals import PrimDihedFcn

        prim = PrimDihedFcn(2.0, 0.0, 1)
        # CptEne(0°) = 2*(1+cos0) = 4
        self.assertAlmostEqual(float(prim.CptEne(0.0)), 4.0)


# ---------------------------------------------------------------------------
# Parametrization defaults isolation
# ---------------------------------------------------------------------------


class TestGitTrackedModuleCase(unittest.TestCase):
    """Catch Windows case-only renames that Linux checkouts still see as snake_case."""

    _REQUIRED = (
        "src/ligandparam/Log.py",
        "src/ligandparam/Driver.py",
        "src/ligandparam/Interfaces.py",
        "src/ligandparam/Parametrization.py",
        "src/ligandparam/Utils.py",
        "src/ligandparam/cli/LigDihedCorrect.py",
        "src/ffpopt/Scripts.py",
        "src/ffpopt/runtime/Console.py",
        "src/ffpopt/geom/Geometric.py",
        "src/ffpopt/ase/Calculator.py",
        "src/scission/Cli.py",
        "src/scission/Models.py",
    )

    def test_pascalcase_modules_are_tracked_exactly(self):
        import shutil
        import subprocess

        root = Path(__file__).resolve().parents[1]
        if shutil.which("git") is None or not (root / ".git").exists():
            self.skipTest("git checkout required")
        tracked = set(
            subprocess.check_output(
                ["git", "ls-files", "src"], cwd=root, text=True
            ).splitlines()
        )
        missing = [path for path in self._REQUIRED if path not in tracked]
        self.assertEqual(
            missing,
            [],
            "git index still has lowercase names; Linux imports will fail. "
            "Use two-step git mv (file.py -> tmp -> File.py).",
        )


class TestSrcImportGraph(unittest.TestCase):
    """Every ``src/`` import of ffpopt / ligandparam / scission must resolve."""

    _TOP = frozenset({"ffpopt", "ligandparam", "scission"})

    def test_in_tree_imports_resolve(self):
        import ast
        import os

        root = Path(__file__).resolve().parents[1] / "src"
        modules: set[str] = set()
        files: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            rel = Path(dirpath).relative_to(root)
            py_files = [f for f in filenames if f.endswith(".py")]
            if "__init__.py" in filenames:
                modules.add(".".join(rel.parts))
            for name in py_files:
                fp = Path(dirpath) / name
                files.append(fp)
                if name == "__init__.py":
                    modules.add(".".join(rel.parts))
                else:
                    modules.add(".".join(list(rel.parts) + [name[:-3]]))

        broken: list[str] = []
        for fp in files:
            text = fp.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(text, filename=str(fp))
            rel = fp.relative_to(root)
            parts = list(rel.parts)
            if parts[-1] == "__init__.py":
                file_pkg = ".".join(parts[:-1])
            else:
                file_pkg = ".".join(parts[:-1])
            pkg_parts = file_pkg.split(".") if file_pkg else []
            for node in ast.walk(tree):
                tgt = None
                if isinstance(node, ast.ImportFrom):
                    if node.level:
                        if node.level == 1:
                            base = pkg_parts
                        else:
                            trim = node.level - 1
                            if trim > len(pkg_parts):
                                broken.append(f"{rel}:{node.lineno} relative past package root")
                                continue
                            base = pkg_parts[: len(pkg_parts) - trim]
                        if node.module:
                            tgt = ".".join(base + node.module.split(".")) if base else node.module
                        else:
                            tgt = ".".join(base)
                    elif node.module and node.module.split(".")[0] in self._TOP:
                        tgt = node.module
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split(".")[0] in self._TOP and alias.name not in modules:
                            broken.append(f"{rel}:{node.lineno} import {alias.name}")
                    continue
                if tgt and tgt.split(".")[0] in self._TOP and tgt not in modules:
                    broken.append(f"{rel}:{node.lineno} {tgt}")
        self.assertEqual(broken, [], "unresolved in-tree imports:\n" + "\n".join(broken))


class TestRecipeDefaultsIsolation(unittest.TestCase):
    def test_fresh_defaults_not_shared(self):
        from ligandparam.Parametrization import fresh_recipe_defaults

        a = fresh_recipe_defaults()
        b = fresh_recipe_defaults()
        a["leaprc"].append("leaprc.protein.ff14SB")
        self.assertNotIn("leaprc.protein.ff14SB", b["leaprc"])
        a["theory"]["low"] = "X"
        self.assertNotEqual(b["theory"]["low"], "X")


if __name__ == "__main__":
    unittest.main()
