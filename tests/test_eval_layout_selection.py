from __future__ import annotations

import unittest

from env.seed_manager.layout_selection import LayoutSelectionError, select_layout_ids


class LayoutSelectionTest(unittest.TestCase):
    def test_unset_preserves_available_order(self):
        self.assertEqual(select_layout_ids(None, [0, 1, 2]), [0, 1, 2])

    def test_exact_subset_preserves_protocol_order(self):
        self.assertEqual(select_layout_ids("25,27,26", range(50)), [25, 27, 26])

    def test_rejects_duplicate_invalid_and_unavailable_ids(self):
        for raw in ("25,25", "-1", "25,", "word", "51"):
            with self.subTest(raw=raw), self.assertRaises(LayoutSelectionError):
                select_layout_ids(raw, range(50))


if __name__ == "__main__":
    unittest.main()
