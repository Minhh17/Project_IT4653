import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dlstudy.splits import make_split_indices, split_hash  # noqa: E402


class SplitTests(unittest.TestCase):
    def test_split_is_deterministic_and_disjoint(self):
        first = make_split_indices(100, 80, 20, 4653)
        second = make_split_indices(100, 80, 20, 4653)
        self.assertEqual(first, second)
        self.assertTrue(set(first[0]).isdisjoint(first[1]))

    def test_different_seed_changes_checksum(self):
        first = make_split_indices(100, 80, 20, 4653)
        second = make_split_indices(100, 80, 20, 42)
        self.assertNotEqual(split_hash(*first), split_hash(*second))


if __name__ == "__main__":
    unittest.main()
