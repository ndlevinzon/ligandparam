"""Parametrization drivers and shared recipe option helpers.

Classes
-------
Parametrization
    Base class that owns input paths, logging, and stage lists.
Recipe
    Thin subclass used by concrete ligand workflows.

Functions
---------
fresh_recipe_defaults
    Build a new defaults mapping with fresh mutable values.
apply_option_defaults
    Assign kwargs or defaults onto a recipe instance safely.
"""

import logging
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Union
from typing_extensions import override

from ligandparam.driver import Driver
from ligandparam.log import get_logger, set_stream_logger, set_file_logger


def fresh_recipe_defaults() -> dict[str, Any]:
    """Return recipe option defaults with newly created mutable values.

    Call this (or rely on :func:`apply_option_defaults`) whenever defaults are
    needed so ``theory`` / ``leaprc`` are never shared across instances.
    """
    return {
        "theory": {"low": "HF/6-31G*", "high": "PBE1PBE/6-31G*"},
        "leaprc": ["leaprc.gaff2"],
        "force_gaussian_rerun": False,
        "nproc": 1,
        "mem": 1,
    }


def _copy_if_mutable(value: Any) -> Any:
    """Shallow-copy dict/list values; leave immutables unchanged."""
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, list):
        return list(value)
    return value


def apply_option_defaults(
    obj: Any,
    kwargs: MutableMapping[str, Any],
    option_names: Iterable[str],
    *,
    defaults: Optional[Mapping[str, Any]] = None,
) -> None:
    """Set ``obj`` attributes from ``kwargs``, else from fresh defaults.

    User-supplied dict/list values are shallow-copied so later mutation on the
    instance (e.g. ``add_leaprc``) does not alter the caller's objects.
    """
    if defaults is None:
        defaults = fresh_recipe_defaults()
    for opt in option_names:
        if opt in kwargs:
            setattr(obj, opt, _copy_if_mutable(kwargs.pop(opt)))
        else:
            setattr(obj, opt, _copy_if_mutable(defaults[opt]))


class Parametrization(Driver):
    """
    A class for parametrizing ligands using various stages.

    Parameters
    ----------
    in_filename : Union[Path, str]
        The input filename of the ligand.
    cwd : Union[Path, str]
        The current working directory.
    *args : tuple
        Additional positional arguments.
    **kwargs : dict
        Additional keyword arguments.

    Keyword Args
    ------------
    label : str, optional
        A label for the ligand, by default the stem of `in_filename`.
    leaprc : list, optional
        A list of leaprc files to use, by default ["leaprc.gaff2"].
    logger : Union[str, logging.Logger], optional
        The logger to use. Can be "file", "stream", or a logging.Logger instance.

    Attributes
    ----------
    in_filename : Path
        The resolved path to the input file.
    label : str
        The label for the ligand.
    cwd : Path
        The current working directory.
    stages : list
        A list of stages to run.
    leaprc : list
        A list of leaprc files to use.
    logger : logging.Logger
        The logger instance.

    Raises
    ------
    ValueError
        If an invalid logger type is provided.
    """

    @override
    def __init__(self, in_filename: Union[Path, str], cwd: Union[Path, str], *args, **kwargs):
        """
        The rough approach to using this class is to generate a new Parametrization class, and then generate self.stages as a list
        of stages that you want to run.

        Parameters
        ----------
        in_filename : Union[Path, str]
            The input filename of the ligand.
        cwd : Union[Path, str]
            The current working directory.
        *args : tuple
            Additional positional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Keyword Args
        ------------
        label : str, optional
            A label for the ligand, by default the stem of `in_filename`.
        leaprc : list, optional
            A list of leaprc files to use, by default ["leaprc.gaff2"].
        logger : Union[str, logging.Logger], optional
            The logger to use. Can be "file", "stream", or a logging.Logger instance.

        Raises
        ------
        ValueError
            If an invalid logger type is provided.
        """
        self.in_filename = Path(in_filename).resolve()
        self.label = kwargs.get("label", self.in_filename.stem)
        self.cwd = Path(cwd)
        self.stages = []
        if "leaprc" in kwargs:
            self.leaprc = list(kwargs["leaprc"])
        else:
            self.leaprc = ["leaprc.gaff2"]
        try:
            logger = kwargs.pop("logger")
            if isinstance(logger, str):
                if logger == "file":
                    self.logger = set_file_logger(self.cwd / f"{self.label}.log")
                elif logger == "stream":
                    self.logger = set_stream_logger()
                else:
                    raise ValueError("Invalid input string for logger. Must be either 'file' or 'stream'.")
            elif isinstance(logger, logging.Logger):
                self.logger = logger
            else:
                raise ValueError("logger must be a string or a logging.Logger instance.")
        except KeyError:
            self.logger = get_logger()

    def add_leaprc(self, leaprc) -> None:
        """
        Add a leaprc file to the list of leaprc files.

        Parameters
        ----------
        leaprc : str
            The name of the leaprc file to add.
        """
        self.leaprc.append(leaprc)


class Recipe(Parametrization):
    """Convenience alias for a ligand-parametrization workflow.

    Subclasses typically implement :meth:`setup` to populate ``self.stages``
    and may override :meth:`execute` for logging around the base pipeline.
    """
    pass
