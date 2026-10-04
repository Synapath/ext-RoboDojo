from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from env.seed_manager.seed_manager import SeedManager


class SeedManagerLayoutContracts(unittest.TestCase):
    def layout_root(self, temporary: str) -> Path:
        root = Path(temporary) / "Eval_Layout" / "RoboDojo" / "arx_x5" / "0"
        root.mkdir(parents=True)
        for layout_id in (2, 7, 11):
            (root / f"stack_bowls_{layout_id}.json").write_text("{}\n")
        return Path(temporary)

    def test_real_sparse_file_suffixes_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary, patch(
            "env.seed_manager.seed_manager.ASSETS_PATH", self.layout_root(temporary)
        ):
            manager = SeedManager(
                {
                    "num_envs": 2,
                    "task_name": "stack_bowls",
                    "config_name": "arx_x5",
                    "seed": 0,
                }
            )
            manager.init_eval()
            self.assertEqual(manager.seed_list, [2, 7, 11])
            self.assertEqual(manager.get_seeds(), [2, 7])
            self.assertEqual(manager.get_seeds(), [11, 11])

    def test_explicit_shard_layouts_never_escape_the_assignment(self):
        with tempfile.TemporaryDirectory() as temporary, patch(
            "env.seed_manager.seed_manager.ASSETS_PATH", self.layout_root(temporary)
        ):
            manager = SeedManager(
                {
                    "num_envs": 2,
                    "task_name": "stack_bowls",
                    "config_name": "arx_x5",
                    "seed": 0,
                    "layout_ids": [2, 11],
                }
            )
            manager.init_eval()
            self.assertEqual(manager.seed_list, [2, 11])
            self.assertEqual(manager.get_seeds(max_count=2), [2, 11])
            self.assertIsNone(manager.get_seeds(max_count=2))

    def test_missing_duplicate_or_unsorted_explicit_layouts_fail(self):
        for layout_ids in ([2, 2], [11, 2], [2, 99], [2, "11"], [2, True]):
            with self.subTest(layout_ids=layout_ids), tempfile.TemporaryDirectory() as temporary, patch(
                "env.seed_manager.seed_manager.ASSETS_PATH", self.layout_root(temporary)
            ):
                manager = SeedManager(
                    {
                        "num_envs": 1,
                        "task_name": "stack_bowls",
                        "config_name": "arx_x5",
                        "seed": 0,
                        "layout_ids": layout_ids,
                    }
                )
                with self.assertRaises(ValueError):
                    manager.init_eval()

    def test_resident_entrypoint_syncs_layouts_into_runtime_config(self):
        source = (
            Path(__file__).parents[1] / "src" / "eval_client" / "main.py"
        ).read_text()
        self.assertIn('"eval_cfg.layout_ids",', source)
        self.assertIn("explicit_layout_ids,", source)


if __name__ == "__main__":
    unittest.main()
