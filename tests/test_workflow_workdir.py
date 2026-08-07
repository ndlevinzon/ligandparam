"""Tests for workdir path helpers (no Amber / wavefront required)."""

import inspect
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ffpopt.Workflows import (
    _in_workdir,
    _prepare_fragment_input,
    _subprocess_cwd,
    run_fragmented_dihed_twist_workflow,
)


class TestWorkdirHelpers(unittest.TestCase):
    def test_in_workdir_relative(self):
        wd = Path("/tmp/frag")
        self.assertEqual(_in_workdir(wd, "start.json"), wd / "start.json")

    def test_in_workdir_absolute_unchanged(self):
        wd = Path("/tmp/frag")
        abs_path = Path("/other/start.json")
        self.assertEqual(_in_workdir(wd, abs_path), abs_path)

    def test_in_workdir_none(self):
        self.assertEqual(_in_workdir(None, "start.json"), Path("start.json"))

    def test_subprocess_cwd(self):
        self.assertIsNone(_subprocess_cwd(None))
        self.assertEqual(_subprocess_cwd(Path("/tmp/frag")), str(Path("/tmp/frag")))


class TestPrepareFragmentNoChdir(unittest.TestCase):
    def test_prepare_uses_cwd_kwarg_not_chdir(self):
        frag_dir = Path("/tmp/fragA").resolve()
        fragment = MagicMock()
        fragment.fragment_id = "f0"
        fragment.manifest_path = frag_dir / "manifest.json"
        fragment.parm7_path = frag_dir / "fragment.parm7"
        fragment.rst7_path = frag_dir / "fragment.rst7"

        with patch("ffpopt.Workflows.subprocess.run") as run, patch.object(
            Path, "exists", return_value=False
        ):
            out = _prepare_fragment_input(
                fragment, skip_existing=True, workdir=frag_dir
            )
        self.assertEqual(out, str(frag_dir / "start.json"))
        run.assert_called_once()
        kwargs = run.call_args.kwargs
        self.assertEqual(kwargs["cwd"], str(frag_dir))
        cmd = run.call_args.args[0]
        self.assertEqual(cmd[0], sys.executable)
        self.assertEqual(cmd[1], "-u")
        self.assertTrue(cmd[2].endswith("ffpopt-PrepareInput.py"))
        self.assertTrue(any(a.startswith("--parm=") for a in cmd))
        self.assertTrue(any(a.startswith("--out=") and "start.json" in a for a in cmd))


class TestRunCurrentPython(unittest.TestCase):
    def test_apply_fit_uses_inprocess_script(self):
        from ffpopt.Workflows import _apply_fit_and_prepare

        wd = Path("/tmp/fragB")
        with patch("ffpopt.Workflows._run_fit_script_inprocess") as apply, patch(
            "ffpopt.Workflows.subprocess.run"
        ) as run, patch.object(Path, "exists", return_value=False), patch.object(
            Path, "is_file", return_value=True
        ), patch(
            "ffpopt.Workflows._ffpopt_bin_script", return_value="/fake/PrepareInput.py"
        ):
            _apply_fit_and_prepare(
                citname="it01",
                origparm="fragment.parm7",
                inp="start.json",
                skip_existing=False,
                workdir=wd,
            )
        apply.assert_called_once()
        self.assertEqual(apply.call_args.args[0].name, "it01.py")
        run.assert_called_once()  # PrepareInput only
        prep_cmd = run.call_args.args[0]
        self.assertEqual(prep_cmd[0], sys.executable)
        self.assertEqual(prep_cmd[1], "-u")
        self.assertTrue("PrepareInput" in prep_cmd[2])

    def test_write_fit_json_uses_absolute_paths(self):
        import tempfile

        from ffpopt.Workflows import _write_fit_json

        class Scan:
            def GetIdxStr(self):
                return "0-1-2-3"

            def GetParamByType(self):
                return "X-X-X-X"

        with tempfile.TemporaryDirectory() as td:
            wd = Path(td)
            (wd / "xtb_0-1-2-3.json").write_text("[]")
            (wd / "orig_0-1-2-3.json").write_text("[]")
            (wd / "frag.parm7").write_text("parm")
            path = _write_fit_json(
                citname="it01",
                scans=[Scan()],
                params={},
                s_template={"params": {}},
                hl_prefix="xtb",
                ll_prefix="orig",
                parm="frag.parm7",
                workdir=wd,
            )
            data = json.loads(Path(path).read_text())
            self.assertTrue(Path(data["output"]).is_absolute())
            self.assertTrue(Path(data["systems"][0]["output"]).is_absolute())
            self.assertTrue(Path(data["systems"][0]["profiles"][0]["hl"]).is_absolute())
            self.assertEqual(Path(data["systems"][0]["output"]).parent, wd.resolve())


class TestFragmentedWorkflowNoChdir(unittest.TestCase):
    def test_source_has_no_os_chdir_call(self):
        src = inspect.getsource(run_fragmented_dihed_twist_workflow)
        self.assertNotIn("os.chdir(", src)
        from ffpopt.Workflows import _run_fragment_twist_job

        job_src = inspect.getsource(_run_fragment_twist_job)
        self.assertIn("workdir=frag_dir", job_src)


class TestSplitFragmentNproc(unittest.TestCase):
    def test_single_fragment_gets_all_cores(self):
        from ffpopt.Workflows import _split_fragment_nproc

        self.assertEqual(_split_fragment_nproc(16, 1), (1, 16))

    def test_many_fragments_prefer_fragment_workers(self):
        from ffpopt.Workflows import _split_fragment_nproc

        self.assertEqual(_split_fragment_nproc(16, 4), (4, 4))
        self.assertEqual(_split_fragment_nproc(16, 20), (16, 1))
        self.assertEqual(_split_fragment_nproc(3, 2), (2, 1))

    def test_pool_used_for_multiple_fragments(self):
        frag_a = MagicMock()
        frag_a.fragment_id = "fragment_1"
        frag_a.fit_torsions = [{"fragment_rotatable_bond": [1, 2]}]
        frag_a.manifest_path = Path("/tmp/f1/manifest.json")
        frag_a.parm7_path = Path("/tmp/f1/fragment.parm7")
        frag_a.rst7_path = Path("/tmp/f1/fragment.rst7")

        frag_b = MagicMock()
        frag_b.fragment_id = "fragment_2"
        frag_b.fit_torsions = [{"fragment_rotatable_bond": [2, 3]}]
        frag_b.manifest_path = Path("/tmp/f2/manifest.json")
        frag_b.parm7_path = Path("/tmp/f2/fragment.parm7")
        frag_b.rst7_path = Path("/tmp/f2/fragment.rst7")

        fake_pool = MagicMock()
        fake_pool.map.return_value = [
            {
                "fragment_id": "fragment_1",
                "dir": "/tmp/f1",
                "bonds": [(0, 1)],
                "twist_result": {},
            },
            {
                "fragment_id": "fragment_2",
                "dir": "/tmp/f2",
                "bonds": [(1, 2)],
                "twist_result": {},
            },
        ]

        # Patch symbols used after the in-function scission import.
        import scission
        import scission.merge as scission_merge

        with patch(
            "ffpopt.Workflows._load_existing_fragments",
            return_value=[frag_a, frag_b],
        ), patch(
            "ffpopt.Workflows._parent_paths_from_args",
            return_value=(Path("/m.mol2"), Path("/m.lib"), Path("/m.frcmod")),
        ), patch(
            "ffpopt.Workflows._make_nondaemon_spawn_pool",
            return_value=fake_pool,
        ) as make_pool, patch.object(
            scission, "FragmentConfig"
        ), patch.object(
            scission, "InputBundle"
        ), patch.object(
            scission, "fragment_ligand"
        ), patch.object(
            scission_merge,
            "merge_fragment_frcmods",
            return_value={"ok": True},
        ):
            result = run_fragmented_dihed_twist_workflow(
                mol2="/m.mol2",
                lib="/m.lib",
                frcmod="/m.frcmod",
                out_dir="/tmp/out",
                merged_frcmod="/tmp/merged.frcmod",
                nproc=8,
                skip_existing=True,
            )

        make_pool.assert_called_once_with(2)
        fake_pool.map.assert_called_once()
        jobs = fake_pool.map.call_args.args[1]
        self.assertEqual(len(jobs), 2)
        self.assertEqual(jobs[0]["wf_nproc"], 4)
        self.assertEqual(jobs[1]["wf_nproc"], 4)
        fake_pool.close.assert_called_once()
        fake_pool.join.assert_called_once()
        self.assertEqual(
            result["merged_frcmod"], str(Path("/tmp/merged.frcmod").resolve())
        )


class TestBondScanPool(unittest.TestCase):
    def test_single_bond_stays_serial(self):
        from ffpopt.Workflows import _run_scans_for_bonds

        scan = MagicMock()
        scan.idxs = [0, 1, 2, 3]
        scan.GetIdxStr.return_value = "0-1-2-3"

        with patch(
            "ffpopt.Workflows._run_bond_scan_job",
            side_effect=lambda job: {
                "prefix": job["prefix"],
                "dihed_idxs": job["dihed_idxs"],
                "result": None,
            },
        ) as worker, patch(
            "ffpopt.Workflows._make_nondaemon_spawn_pool"
        ) as make_pool:
            out = _run_scans_for_bonds(
                [scan],
                prefix="xtb",
                model="xtb",
                inp="/tmp/start.json",
                nproc=8,
                skip_existing=True,
                workdir=None,
                logger=None,
                wf_kwargs={"nproc": 8, "delta": 10},
            )

        make_pool.assert_not_called()
        worker.assert_called_once()
        self.assertEqual(worker.call_args.args[0]["wf_kwargs"]["nproc"], 8)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][0], "xtb")

    def test_multi_bond_splits_nproc_and_pools(self):
        from ffpopt.Workflows import _run_scans_for_bonds

        scans = []
        for idxs, label in (
            ([0, 1, 2, 3], "0-1-2-3"),
            ([1, 2, 3, 4], "1-2-3-4"),
            ([2, 3, 4, 5], "2-3-4-5"),
            ([3, 4, 5, 6], "3-4-5-6"),
        ):
            s = MagicMock()
            s.idxs = idxs
            s.GetIdxStr.return_value = label
            scans.append(s)

        fake_pool = MagicMock()
        fake_pool.map.side_effect = lambda fn, jobs: [
            {
                "prefix": j["prefix"],
                "dihed_idxs": j["dihed_idxs"],
                "result": {"ok": True},
            }
            for j in jobs
        ]

        with patch(
            "ffpopt.Workflows._make_nondaemon_spawn_pool",
            return_value=fake_pool,
        ) as make_pool:
            out = _run_scans_for_bonds(
                scans,
                prefix="xtb",
                model="xtb",
                inp="/tmp/start.json",
                nproc=8,
                skip_existing=True,
                workdir=Path("/tmp/wd"),
                logger=None,
                wf_kwargs={"nproc": 8, "delta": 15},
            )

        make_pool.assert_called_once_with(4)
        fake_pool.map.assert_called_once()
        jobs = fake_pool.map.call_args.args[1]
        self.assertEqual(len(jobs), 4)
        for job in jobs:
            self.assertEqual(job["wf_kwargs"]["nproc"], 2)
            self.assertEqual(job["wf_kwargs"]["delta"], 15)
            self.assertEqual(job["workdir"], str(Path("/tmp/wd")))
        fake_pool.close.assert_called_once()
        fake_pool.join.assert_called_once()
        self.assertEqual(len(out), 4)
        self.assertEqual(out[0][1], (0, 1, 2, 3))


if __name__ == "__main__":
    unittest.main()
