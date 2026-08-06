"""Tests for typed 0-based bond pairs at the scission→ffpopt boundary."""

import pytest

from ffpopt.Workflows import (
    BondPair0,
    bonds0_from_scission_fit_torsions,
    normalize_bond_pairs0,
)


def test_normalize_bond_pairs0_from_tuples():
    assert normalize_bond_pairs0([(1, 2), [3, 4]]) == [(1, 2), (3, 4)]


def test_normalize_bond_pairs0_from_cli_strings():
    assert normalize_bond_pairs0(["0,1", "10,11"]) == [(0, 1), (10, 11)]


def test_normalize_bond_pairs0_mixed():
    assert normalize_bond_pairs0([(0, 1), "2,3"]) == [(0, 1), (2, 3)]


def test_normalize_bond_pairs0_rejects_bad_string():
    with pytest.raises(ValueError, match="i,j"):
        normalize_bond_pairs0(["0-1"])


def test_normalize_bond_pairs0_rejects_bad_entry():
    with pytest.raises(TypeError):
        normalize_bond_pairs0([42])


def test_bonds0_from_scission_fit_torsions():
    fit = [
        {"fragment_rotatable_bond": [1, 2]},
        {"fragment_rotatable_bond": [5, 8]},
    ]
    pairs: list[BondPair0] = bonds0_from_scission_fit_torsions(fit)
    assert pairs == [(0, 1), (4, 7)]
