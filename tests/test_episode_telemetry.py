import ast
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("episode_telemetry", ROOT / "utils/episode_telemetry.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
EpisodeTelemetry = module.EpisodeTelemetry


def test_recording_is_copied_and_controls_are_not_observations(tmp_path):
    recorder = EpisodeTelemetry(.01, {"left_arm_joint_state": ["a", "b"]})
    joints = np.array([1., 2.])
    recorder.append(0, {"left_arm_joint_state": joints, "left_ee_joint_state": [1.]},
                    {"left_arm_joint_state": {"position": [2., 3.]}},
                    {"left_ee_pose": [0., 0., 0., 1., 0., 0., 0.]}, {"head": 0})
    joints[:] = 99
    recorder.append(2, {"left_arm_joint_state": [2., float("nan")]}, {}, {}, {"head": 1})
    path = tmp_path / "episode.telemetry.json"
    recorder.save(path, {"episode": 0}, {"head": {"path": "head.mp4", "fps": 25}})
    result = json.loads(path.read_text())
    signals = {s["key"]: s for s in result["series"]}
    assert signals["state/left_arm_joint_state"]["values"] == [[1., 2.], [2., None]]
    assert "state/left_ee_joint_state" not in signals
    assert signals["action/left_ee_pose"]["unit"] is None
    assert signals["state/left_arm_joint_state"]["channels"] == ["a", "b"]
    assert signals["target/left_arm_joint_state"]["values"][1] == [None, None]
    assert result["videos"][0]["frames"] == [0, 1]
    assert signals["state/left_arm_joint_state"]["times_seconds"] == [0., .02]


def test_limits_and_fresh_environment():
    old = EpisodeTelemetry(.01)
    old.MAX_SAMPLES = 1
    old.append(0, {"arm_joint_state": [1.]}, {}, {}, {})
    old.append(1, {"arm_joint_state": [2.]}, {}, {}, {})
    assert old.truncated and len(old.rows) == 1
    fresh = EpisodeTelemetry(.01)
    assert fresh.rows == [] and not fresh.truncated
    assert old.payload({}, {})["status"] == "truncated"
    fresh.MAX_BYTES = 1
    fresh.append(0, {"arm_joint_state": [1.]}, {}, {}, {})
    assert fresh.truncated and fresh.rows == []


def test_eval_hooks_do_not_add_simulation_or_render_calls():
    tree = ast.parse((ROOT / "src/eval_client/eval_env.py").read_text())
    functions = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    observation = functions["get_obs_batch"]
    calls = [n.func.attr for n in ast.walk(observation) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert calls.count("get_obs") == 1 and calls.count("render") == 1
    assert "step" not in calls and "sim_step" not in calls
    close = ast.unparse(functions["close"])
    assert "'telemetry'" in close and "'telemetry_actions'" in close


def test_clock_uses_native_physics_counter_and_starts_at_first_sample():
    tree = ast.parse((ROOT / "src/eval_client/eval_env.py").read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "get_obs_batch")
    source = ast.unparse(method)
    assert "self.sim.physics_dt" in source and "self.sim._sim_step_counter" in source
    recorder = EpisodeTelemetry(.01)
    recorder.append(6, {"arm_joint_state": [0.]}, {}, {}, {})
    recorder.append(9, {"arm_joint_state": [1.]}, {}, {}, {})
    assert recorder.payload({}, {})["series"][0]["times_seconds"] == [0., .03]
