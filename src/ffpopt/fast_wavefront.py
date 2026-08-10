"""Compatibility facade — implementation lives in ``ffpopt.runtime.fast_wavefront``."""
from ffpopt.runtime.fast_wavefront import *  # noqa: F403
try:
    from ffpopt.runtime.fast_wavefront import __all__ as __all__  # noqa: F401
except ImportError:
    pass
