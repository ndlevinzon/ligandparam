"""Tests for shared fragment CPU leasing."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ffpopt.runtime.cpu_budget import CpuBudget, fair_share_leases


class TestFairShareLeases(unittest.TestCase):
    def test_even_split_with_leftovers(self):
        shares = fair_share_leases(20, [f"f{i}" for i in range(11)])
        self.assertEqual(sum(shares.values()), 20)
        self.assertEqual(min(shares.values()), 1)
        self.assertEqual(max(shares.values()), 2)
        self.assertEqual(sum(1 for v in shares.values() if v == 2), 9)

    def test_prefer_gets_first_leftover(self):
        shares = fair_share_leases(5, ["a", "b", "c"], prefer="c")
        self.assertEqual(shares["c"], 2)
        self.assertEqual(sum(shares.values()), 5)

    def test_more_owners_than_cores(self):
        shares = fair_share_leases(2, ["a", "b", "c"], prefer="b")
        self.assertEqual(shares, {"b": 1, "a": 1})


class TestCpuBudget(unittest.TestCase):
    def test_lease_release_reclaim(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "budget.json"
            budget = CpuBudget(path, 20)
            budget.lease("fragment_1")
            budget.lease("fragment_2")
            snap = budget.snapshot()
            self.assertEqual(snap["used"], 20)
            self.assertEqual(snap["free"], 0)
            self.assertEqual(sum(snap["leases"].values()), 20)
            budget.release("fragment_1")
            c = budget.lease("fragment_2")
            self.assertEqual(c, 20)
            self.assertEqual(budget.snapshot()["leases"], {"fragment_2": 20})

    def test_eleven_owners_use_all_cores(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "budget.json"
            budget = CpuBudget(path, 20)
            owners = [f"fragment_{i}" for i in range(1, 12)]
            for oid in owners:
                budget.lease(oid, active_owners=owners)
            snap = budget.snapshot()
            self.assertEqual(snap["used"], 20)
            self.assertEqual(len(snap["leases"]), 11)


if __name__ == "__main__":
    unittest.main()
