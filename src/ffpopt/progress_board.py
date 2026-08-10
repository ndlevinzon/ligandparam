"""Compatibility facade — implementation lives in ``ffpopt.runtime.progress_board``."""
from ffpopt.runtime.progress_board import *  # noqa: F403
try:
    from ffpopt.runtime.progress_board import __all__ as __all__  # noqa: F401
except ImportError:
    pass
