"""Compatibility alias — fragment helpers live in ``runtime.progress_board``."""

from importlib import import_module
import sys

# Re-export fragment symbols from progress_board without replacing this module
# (callers import fragment_progress as a distinct name).
_pb = import_module("ffpopt.runtime.progress_board")
KNOWN_STAGES = _pb.KNOWN_STAGES
FragmentBoardWatcher = _pb.FragmentBoardWatcher
FragmentProgressStore = _pb.FragmentProgressStore
format_fragment_board = _pb.format_fragment_board
fragment_stdio_to_file = _pb.fragment_stdio_to_file
make_fragment_file_logger = _pb.make_fragment_file_logger

__all__ = [
    "KNOWN_STAGES",
    "FragmentBoardWatcher",
    "FragmentProgressStore",
    "format_fragment_board",
    "fragment_stdio_to_file",
    "make_fragment_file_logger",
]
