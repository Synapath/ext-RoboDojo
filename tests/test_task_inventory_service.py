from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.internal import task_inventory


class ServiceInventoryContracts(unittest.TestCase):
    def fixture(self, temporary: str):
        root = Path(temporary)
        config = root / "config"
        policy = root / "policy"
        env_config = root / "env_cfg"
        (env_config / "sim").mkdir(parents=True)
        config.mkdir()
        policy.mkdir()
        (config / "_task.yml").write_text(
            "common:\n  eval_nums: 25\ntasks:\n  stack_bowls:\n"
            "    eval_nums: 50\n  stack_bowls_random: {}\n"
        )
        for name, batch in (("Pi_05", True), ("OpenVLA", False)):
            directory = policy / name
            directory.mkdir()
            (directory / "deploy.yml").write_text(
                f"policy_name: {name}\neval_batch: {str(batch).lower()}\n"
            )
            function = "eval_one_episode_batch" if batch else "eval_one_episode"
            (directory / "deploy.py").write_text(f"def {function}():\n    pass\n")
        (env_config / "arx_x5.yml").write_text("config:\n  sim: sim_config\n")
        (env_config / "sim" / "sim_config.yml").write_text(
            "scene:\n  num_envs: 10\n  clutter_env_limit: 3\n"
        )
        tasks = [
            {"name": "stack_bowls", "runnable": True},
            {"name": "stack_bowls_random", "runnable": True},
        ]
        return config, policy, env_config, tasks

    def test_probe_reads_native_counts_adapters_and_environment_limits(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, policy, env_config, tasks = self.fixture(temporary)
            with (
                patch.object(task_inventory, "CONFIG_DIR", config),
                patch.object(task_inventory, "POLICY_DIR", policy),
                patch.object(task_inventory, "ENV_CONFIG_DIR", env_config),
                patch.object(task_inventory, "_task_records", return_value=tasks),
            ):
                result = task_inventory.build_service_inventory(["arx_x5"])
        self.assertEqual(result["schema_version"], "robodojo-runtime-inventory-v1")
        self.assertEqual(result["tasks"]["stack_bowls"]["native_eval_num"], 50)
        self.assertEqual(
            result["tasks"]["stack_bowls_random"]["native_eval_num"], 25
        )
        self.assertTrue(result["policy_adapters"]["Pi_05"]["eval_batch"])
        self.assertFalse(result["policy_adapters"]["OpenVLA"]["eval_batch"])
        self.assertEqual(result["env_configs"]["arx_x5"]["num_envs"], 10)
        self.assertEqual(
            result["env_configs"]["arx_x5"]["random_task_num_envs"], 3
        )

    def test_missing_adapter_entrypoint_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, policy, env_config, tasks = self.fixture(temporary)
            (policy / "Pi_05" / "deploy.py").write_text("def wrong():\n    pass\n")
            with (
                patch.object(task_inventory, "CONFIG_DIR", config),
                patch.object(task_inventory, "POLICY_DIR", policy),
                patch.object(task_inventory, "ENV_CONFIG_DIR", env_config),
                patch.object(task_inventory, "_task_records", return_value=tasks),
                self.assertRaisesRegex(ValueError, "entrypoint is missing"),
            ):
                task_inventory.build_service_inventory(["arx_x5"])


if __name__ == "__main__":
    unittest.main()
