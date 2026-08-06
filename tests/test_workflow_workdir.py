"""Tests for workdir path helpers (no Amber / wavefront required)."""

import inspect
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
        self.assertTrue(any(a.startswith("--parm=") for a in cmd))
        self.assertTrue(any(a.startswith("--out=") and "start.json" in a for a in cmd))


class TestFragmentedWorkflowNoChdir(unittest.TestCase):
    def test_source_has_no_os_chdir_call(self):
        src = inspect.getsource(run_fragmented_dihed_twist_workflow)
        self.assertNotIn("os.chdir(", src)
        self.assertIn("workdir=frag_dir", src)


if __name__ == "__main__":
    unittest.main()
