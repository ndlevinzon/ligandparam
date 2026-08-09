"""Tests for parallel fragment status board helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from ffpopt.fragment_progress import (
    FragmentBoardWatcher,
    FragmentProgressStore,
    format_fragment_board,
    fragment_stdio_to_file,
)


class TestFragmentProgressStore(unittest.TestCase):
    def test_register_update_and_board(self):
        with tempfile.TemporaryDirectory() as td:
            store = FragmentProgressStore(Path(td) / "status.json")
            store.register("frag_a", bonds=2, frag_dir=str(Path(td) / "a"))
            store.register("frag_b", bonds=1, frag_dir=str(Path(td) / "b"))
            store.update(
                "frag_a",
                status="running",
                stage="hl_scan",
                detail="model=xtb | 2 bond(s)",
            )
            snap = store.snapshot()
            self.assertEqual(snap["frag_a"]["status"], "running")
            self.assertEqual(snap["frag_a"]["stage"], "hl_scan")
            self.assertEqual(snap["frag_b"]["status"], "queued")
            board = store.render_board()
            self.assertIn("frag_a", board)
            self.assertIn("hl_scan", board)
            self.assertIn("running", board)
            self.assertIn("queued", board)

    def test_format_empty_board(self):
        text = format_fragment_board({})
        self.assertIn("(none)", text)


class TestFragmentStdioRedirect(unittest.TestCase):
    def test_print_goes_to_file_and_console(self):
        import io
        import sys

        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "frag-twist.log"
            fake_out = io.StringIO()
            old = sys.stdout
            try:
                sys.stdout = fake_out
                with fragment_stdio_to_file(log_path, fragment_id="fragment_1"):
                    print("hello-fragment")
            finally:
                sys.stdout = old
            self.assertIn("hello-fragment", log_path.read_text(encoding="utf-8"))
            console = fake_out.getvalue()
            self.assertIn("hello-fragment", console)
            self.assertIn("[ffpopt:fragment_1]", console)


class TestConsoleFormat(unittest.TestCase):
    def test_format_console_line_tags_message(self):
        from ffpopt.console import format_console_line

        line = format_console_line("wavefront step\n", tag="ffpopt:fragment_2")
        self.assertIn("[ffpopt:fragment_2]", line)
        self.assertIn("wavefront step", line)
        self.assertRegex(line, r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} ")


class TestFragmentBoardWatcher(unittest.TestCase):
    def test_writes_board_file_and_logs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            store = FragmentProgressStore(root / "status.json")
            store.register("f1", bonds=1)
            logger = MagicMock()
            board_path = root / "FRAG_STATUS.txt"
            watcher = FragmentBoardWatcher(
                store,
                board_path=board_path,
                logger=logger,
                interval_sec=60.0,
            )
            watcher.start()
            watcher.stop()
            self.assertTrue(board_path.is_file())
            text = board_path.read_text(encoding="utf-8")
            self.assertIn("f1", text)
            self.assertTrue(logger.info.called)


if __name__ == "__main__":
    unittest.main()
