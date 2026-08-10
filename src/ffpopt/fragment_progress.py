"""Compatibility facade — implementation lives in ``ffpopt.runtime.fragment_progress``."""
from ffpopt.runtime.fragment_progress import *  # noqa: F403
try:
    from ffpopt.runtime.fragment_progress import __all__ as __all__  # noqa: F401
except ImportError:
    pass
