"""Regression tests for the Constraint constructor signature.

Guards against the call-site bug where a 4-atom dihedral index list was passed
as the first positional argument (the constraint *name*), leaving the required
``idxs`` argument unfilled -- which raised
``TypeError: Constraint.__init__() missing 1 required positional argument: 'idxs'``
at runtime (see Dihedrals.IsolatedLinearSolve).
"""

import pytest

from ffpopt.Constraints import Constraint


def test_name_is_first_positional_argument():
    """A dihedral constraint is built as Constraint("dihed", idxs)."""
    idxs = [0, 1, 2, 3]
    con = Constraint("dihed", idxs)
    assert con.name == "dihed"
    assert con.idxs == idxs
    assert con.value is None


def test_passing_idxs_as_first_arg_is_rejected():
    """The old buggy call -- idxs in the name slot -- must not silently succeed.

    Passing the index list positionally first leaves ``idxs`` unfilled, so the
    constructor raises TypeError. This is exactly the failure the workflow hit.
    """
    with pytest.raises(TypeError):
        Constraint([0, 1, 2, 3], graph=None)


@pytest.mark.parametrize(
    "name,idxs",
    [
        ("bond", [0, 1]),
        ("angle", [0, 1, 2]),
        ("dihed", [0, 1, 2, 3]),
        ("puckerx", [0, 1, 2, 3, 4]),
        ("puckery", [0, 1, 2, 3, 4]),
    ],
)
def test_valid_name_idxs_combinations(name, idxs):
    con = Constraint(name, idxs)
    assert con.name == name
    assert con.idxs == idxs


def test_mismatched_name_and_idxs_length_is_rejected():
    """A 4-atom index list named anything but 'dihed' raises."""
    with pytest.raises(Exception):
        Constraint("bond", [0, 1, 2, 3])
