"""Compatibility alias — canonical home is :mod:`ffpopt.runtime.progress_board`."""

from importlib import import_module
import sys

sys.modules[__name__] = import_module("ffpopt.runtime.progress_board")
