"""CPU-only contracts for resident transport and evaluation-state reset."""
import ast
from copy import deepcopy
from datetime import datetime
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
import sys
from unittest.mock import patch

from utils.service_session import validate_request

ROOT = Path(__file__).resolve().parents[1]


def load_method(relative, class_name, method_name, namespace):
    # Execute the actual method without importing Isaac or launching an app.
    module = ast.parse((ROOT / relative).read_text())
    cls = next(node for node in ast.walk(module) if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(relative), "exec"), namespace)
    return namespace[method_name]


class DummyManager:
    def __init__(self, *args, **kwargs):
        self.closed = False
    def initialize(self, env):
        self.env = env
    def init_eval(self, **kwargs):
        self.completed = kwargs
    def close(self):
        self.closed = True
    def call(self, *args, **kwargs):
        pass


class ResidentContracts(unittest.TestCase):
    def request(self):
        return {"schema_version": "sim-service-resident-v1", "session_id": "a" * 32,
                "sequence": 1, "job_id": "b" * 32, "nonce": "c" * 32, "request_hash": "d" * 64,
                "spec": {"task": "stack_bowls", "policy_adapter": "Pi_05", "policy_host": "10.0.0.2",
                         "policy_port": 9999, "checkpoint_ref": "hold", "env_config": "arx_x5",
                         "action_type": "ee", "seed": 0, "eval_num": 1,
                         "execution_horizon": 20, "headless": True}}

    def test_mailbox_rejects_stale_identity_and_extra_paths(self):
        value = self.request()
        self.assertEqual(validate_request(value, "a" * 32, 1), value["spec"])
        for key, bad in (("sequence", 0), ("session_id", "b" * 32), ("job_id", "../escape"),
                         ("nonce", "bad"), ("request_hash", "bad")):
            changed = deepcopy(value)
            changed[key] = bad
            with self.assertRaises(ValueError):
                validate_request(changed, "a" * 32, 1)
        value["output_path"] = "/history"
        with self.assertRaises(ValueError):
            validate_request(value, "a" * 32, 1)

    def test_only_dynamic_fields_can_change_within_session(self):
        value = self.request()
        previous = deepcopy(value["spec"])
        value["spec"].update(seed=1, policy_host="10.0.0.3", policy_port=8000, checkpoint_ref="next")
        validate_request(value, "a" * 32, 1, previous)
        for key, bad in (("task", "insert_gear"), ("eval_num", 2), ("execution_horizon", 50)):
            changed = deepcopy(value)
            changed["spec"][key] = bad
            with self.assertRaises(ValueError):
                validate_request(changed, "a" * 32, 1, previous)

    def test_configure_evaluation_resets_all_job_state_without_replacing_sim(self):
        namespace = {"deepcopy": deepcopy, "os": os, "datetime": datetime, "BENCHMARK": "RoboDojo",
                     "ObsManager": DummyManager, "SeedManager": DummyManager, "WsModelClient": DummyManager,
                     "get_robot_action_dim_info": lambda **kw: {}, "profiled": lambda name: lambda fn: fn,
                     "_patch_websockets_proxy_compat": lambda: None}
        configure = load_method("src/eval_client/eval_env.py", "EvalEnv", "configure_evaluation", namespace)
        env = SimpleNamespace(num_envs=1, dt=.004, sim=object(), video_writers={},
                              scene_manager=SimpleNamespace(layout_manager=SimpleNamespace(replay=False)),
                              _sweep_stream_dir=lambda: None)
        config = SimpleNamespace(sim=SimpleNamespace(scene=SimpleNamespace(num_envs=1), seed=[0]),
                                 eval_cfg={"task_name": "stack_bowls", "policy_name": "Pi_05",
                                           "config_name": "arx_x5", "eval_num": 1, "seed": 0},
                                 deploy_cfg={"port": 9999})
        with patch.dict(os.environ, {"ROBODOJO_RUN_ID": "service-first"}):
            configure(env, config)
        client, seed_manager, obs_manager, simulator = env.model_client, env.seed_manager, env.obs_manager, env.sim
        env.success_nums, env.fail_nums, env.total_score, env.unstable_nums = 3, 7, 9, 5
        env.eval_result["details"][0] = {"layout_id": 8}
        env.abandoned_seeds.add(9)
        env.current_env_seed_map[0] = 8
        config.eval_cfg["seed"] = 1
        with patch.dict(os.environ, {"ROBODOJO_RUN_ID": "service-next", "ROBODOJO_OUTPUT_ROOT": "eval_result/new"}):
            configure(env, config)
        self.assertIs(env.sim, simulator)
        self.assertTrue(client.closed)
        self.assertIsNot(env.model_client, client)
        self.assertIsNot(env.seed_manager, seed_manager)
        self.assertIsNot(env.obs_manager, obs_manager)
        self.assertEqual((env.success_nums, env.fail_nums, env.total_score, env.unstable_nums), (0, 0, 0, 0))
        self.assertEqual(env.eval_result["details"], {})
        self.assertEqual(env.abandoned_seeds, set())
        self.assertEqual(env.current_env_seed_map, {})
        self.assertEqual(env.eval_seed, 1)
        self.assertTrue(env.save_dir.startswith("eval_result/new/"))
        self.assertTrue(env.save_dir.endswith("service-next"))
        self.assertEqual(env.seed_manager.completed, {"completed_layout_ids": [], "abandoned_layout_ids": []})

    def test_live_video_writer_prevents_reconfiguration(self):
        configure = load_method("src/eval_client/eval_env.py", "EvalEnv", "configure_evaluation", {})
        with self.assertRaises(RuntimeError):
            configure(SimpleNamespace(video_writers={0: object()}), None)

    def test_camera_soft_reset_does_not_duplicate_render_products(self):
        reset = load_method("env/camera_manager/capture/tiled_capture_manager.py", "TiledCaptureManager", "reset", {})
        capture = SimpleNamespace(tiled_cameras=[object()], num_cams=1,
                                  cameras=[[SimpleNamespace(prim_path="/camera")]], camera_prim_paths=[["/camera"]],
                                  init_cameras=lambda: self.fail("must reuse existing camera"))
        reset(capture)
        capture.cameras[0][0].prim_path = "/changed"
        with self.assertRaises(RuntimeError):
            reset(capture)

    def test_renderer_history_cleared_after_layout_before_native_warmup(self):
        events = []
        context = SimpleNamespace(reset_renderer_accumulation=lambda: events.append("history-reset"))
        usd = SimpleNamespace(get_context=lambda: context)
        omni = SimpleNamespace(usd=usd)
        setup = load_method("src/eval_client/eval_env.py", "EvalEnv", "setup_scene", {"os": os})
        env = SimpleNamespace(num_envs=1, unstable_nums=0, episode_nums=1, physx_monitor_enabled=False,
            scene_manager=SimpleNamespace(apply_saved_poses=lambda **kw: events.append("layout"),
                layout_manager=SimpleNamespace(check_layout_stability=lambda env: (True, []))),
            _align_layout_success=lambda: None, render=lambda: events.append("render"),
            sim_step=lambda: events.append("physics"), obs_manager=SimpleNamespace(get_obs=lambda: events.append("obs")))
        with patch.dict(os.environ, {"SIM_SERVICE_SESSION_ID": "a" * 32}), patch.dict(sys.modules, {"omni": omni, "omni.usd": usd}):
            setup(env)
        self.assertEqual(events[:3], ["layout", "history-reset", "render"])
        self.assertEqual(events.count("history-reset"), 1)
        self.assertEqual(events.count("render"), 50)
        self.assertEqual(events.count("physics"), 200)
        self.assertEqual(events.count("obs"), 40)
        self.assertTrue(env._renderer_history_reset)


if __name__ == "__main__":
    unittest.main()
