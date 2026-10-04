import unittest
from unittest.mock import patch
import json
import os
from pathlib import Path
import tempfile

from utils.camera_readback import selected_frames_to_numpy
from utils.performance import WallProfile, close_profiled_app
from utils.eval_allocation import effective_num_envs


class RuntimePerformanceTests(unittest.TestCase):
    def test_profile_persisted_before_nonreturning_kit_shutdown(self):
        profile = WallProfile(True)
        with tempfile.TemporaryDirectory() as directory:
            original = Path.cwd()
            try:
                os.chdir(directory)
                class App:
                    def close(self):
                        payload = json.loads(next(Path("eval_result").glob("performance-*.json")).read_text())
                        assert payload["finished"] is True
                        assert payload["metadata"]["shutdown_complete"] is False
                        raise SystemExit(0)
                with patch("utils.performance.PROFILE", profile), patch.dict(os.environ, {"ROBODOJO_RUN_ID": "test-shutdown"}):
                    with self.assertRaises(SystemExit):
                        close_profiled_app(App())
            finally:
                os.chdir(original)

    def test_allocation_is_capped_by_effective_episode_count(self):
        self.assertEqual(effective_num_envs(10, 1), 1)
        self.assertEqual(effective_num_envs(10, 7), 7)
        self.assertEqual(effective_num_envs(10, 50), 10)
        self.assertEqual(effective_num_envs(1, 50), 1)

    def test_allocation_rejects_nonpositive_and_noninteger_values(self):
        for configured, episodes in [(0, 1), (10, 0), (-1, 2), (10, -1), (True, 1), (10, 1.5)]:
            with self.subTest(configured=configured, episodes=episodes):
                with self.assertRaises(ValueError):
                    effective_num_envs(configured, episodes)

    def test_nested_exclusive_time_is_not_double_counted(self):
        values = iter([0, 1, 2, 5, 7, 10, 12])
        profile = WallProfile(True, lambda: next(values))
        with profile.measure("outer"):
            with profile.measure("inner"):
                pass
        profile.set_phase("episode")
        result = profile.snapshot()
        self.assertEqual(result["operations"]["startup/outer"]["inclusive_s"], 6)
        self.assertEqual(result["operations"]["startup/outer"]["exclusive_s"], 3)
        self.assertEqual(result["operations"]["startup/inner"]["exclusive_s"], 3)
        self.assertEqual(result["phases_s"], {"startup": 10, "episode": 2})

    def test_exception_unwinds_timer_stack(self):
        profile = WallProfile(True)
        with self.assertRaises(ValueError):
            with profile.measure("failed"):
                raise ValueError("test")
        self.assertEqual(profile.stack, [])
        self.assertEqual(profile.operations["startup/failed"]["count"], 1)

    def test_disabled_profile_records_nothing(self):
        profile = WallProfile(False)
        with profile.measure("op"):
            pass
        self.assertEqual(profile.operations, {})

    def test_select_happens_before_host_copy_and_preserves_order(self):
        operations = []
        class Tensor:
            def __getitem__(self, indices):
                operations.append(("select", indices))
                return self
            def cpu(self):
                operations.append("cpu")
                return self
            def numpy(self):
                operations.append("numpy")
                return "selected pixels"
        self.assertEqual(selected_frames_to_numpy(Tensor(), [3, 0]), "selected pixels")
        self.assertEqual(operations, [("select", [3, 0]), "cpu", "numpy"])

    def test_real_tensor_pixels_and_dtype(self):
        try:
            import numpy as np
            import torch
            import warp as wp
        except ImportError:
            self.skipTest("Run in the pinned runtime for torch/warp coverage")
        original = np.arange(5 * 2 * 3 * 4, dtype=np.uint8).reshape(5, 2, 3, 4)
        # The GPU path uses the same Warp->torch view and row selection.
        tensor = wp.to_torch(wp.array(original, device="cpu"))
        selected = selected_frames_to_numpy(tensor, [3, 0])
        np.testing.assert_array_equal(selected, original[[3, 0]])
        self.assertEqual(selected.dtype, original.dtype)


if __name__ == "__main__":
    unittest.main()
