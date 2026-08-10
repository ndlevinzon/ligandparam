"""Compatibility facade — implementation lives in ``ffpopt.runtime.cpu_budget``."""
from ffpopt.runtime.cpu_budget import *  # noqa: F403
try:
    from ffpopt.runtime.cpu_budget import __all__ as __all__  # noqa: F401
except ImportError:
    pass
