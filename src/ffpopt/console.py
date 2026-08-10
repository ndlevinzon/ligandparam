"""Compatibility alias — canonical home is :mod:`ffpopt.runtime.console`."""

from importlib import import_module
import sys

sys.modules[__name__] = import_module("ffpopt.runtime.console")
