"""Compatibility alias — canonical home is :mod:`ffpopt.scan.wavefront_mixins`."""

from importlib import import_module
import sys

sys.modules[__name__] = import_module("ffpopt.scan.wavefront_mixins")
