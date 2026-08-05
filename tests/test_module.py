"""Package smoke tests."""

import unittest


class TestPackageImport(unittest.TestCase):
    def test_import_ligandparam(self):
        import ligandparam

        self.assertTrue(hasattr(ligandparam, "__version__"))


if __name__ == "__main__":
    unittest.main()
