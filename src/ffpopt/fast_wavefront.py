"""Compatibility alias — canonical home is :mod:`ffpopt.runtime.fast_wavefront`."""

from importlib import import_module
import sys

sys.modules[__name__] = import_module("ffpopt.runtime.fast_wavefront")
