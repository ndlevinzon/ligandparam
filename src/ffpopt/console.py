"""Compatibility facade — implementation lives in ``ffpopt.runtime.console``."""
from ffpopt.runtime.console import *  # noqa: F403
try:
    from ffpopt.runtime.console import __all__ as __all__  # noqa: F401
except ImportError:
    pass
