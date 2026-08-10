"""Compatibility alias — canonical home is :mod:`ffpopt.runtime.cpu_budget`."""

from importlib import import_module
import sys

sys.modules[__name__] = import_module("ffpopt.runtime.cpu_budget")
