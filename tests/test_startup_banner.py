"""Startup banner / version smoke tests."""

from __future__ import annotations

import io
import logging
import os
import unittest

from ffpopt.runtime import console as console_mod
from ffpopt.runtime.console import format_startup_banner, print_startup_banner
from ligandparam import __version__


class TestStartupBanner(unittest.TestCase):
    def setUp(self):
        console_mod._BANNER_PRINTED = False
        os.environ.pop("LIGANDPARAM_BANNER_PRINTED", None)

    def tearDown(self):
        console_mod._BANNER_PRINTED = False
        os.environ.pop("LIGANDPARAM_BANNER_PRINTED", None)

    def test_format_includes_logo_version_authors(self):
        text = format_startup_banner(version="9.9.9")
        self.assertIn("ligandparam", text)
        self.assertIn("v9.9.9", text)
        self.assertIn("Zeke Piskulich", text)
        self.assertIn("Nate Levinzon", text)
        self.assertIn(".____", text)

    def test_print_once_module_flag(self):
        buf = io.StringIO()
        self.assertTrue(print_startup_banner(stream=buf))
        self.assertFalse(print_startup_banner(stream=buf))
        self.assertEqual(buf.getvalue().count("Authors:"), 1)

    def test_print_once_via_env_for_spawned_workers(self):
        """Child processes re-import with a fresh module flag; env must block."""
        buf = io.StringIO()
        self.assertTrue(print_startup_banner(stream=buf))
        console_mod._BANNER_PRINTED = False  # simulate fresh import
        self.assertFalse(print_startup_banner(stream=buf))
        self.assertEqual(buf.getvalue().count("Authors:"), 1)

    def test_attach_console_does_not_print_banner(self):
        logger = logging.getLogger("test.banner.attach")
        logger.handlers.clear()
        console_mod.attach_console_handlers(logger, tag="test")
        self.assertFalse(console_mod._BANNER_PRINTED)
        self.assertIsNone(os.environ.get("LIGANDPARAM_BANNER_PRINTED"))
        logger.handlers.clear()

    def test_version_is_150(self):
        self.assertEqual(__version__, "1.5.0")


if __name__ == "__main__":
    unittest.main()
