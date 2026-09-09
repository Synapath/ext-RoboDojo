import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "charger_diagnostics", Path(__file__).resolve().parents[1] / "utils/charger_diagnostics.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_socket_frame_rotation_and_common_translation():
    s = np.sqrt(0.5)
    a, b = np.array([0.0, 1.0, 2.0, s, s, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, s, 0.0, 0.0, s])
    expected = m.geometry(a, b)
    np.testing.assert_allclose(expected["charger_root_in_socket_frame_m"], [1.0, 0.0, 2.0], atol=1e-12)
    assert expected["charger_axis_up_angle_deg"] < 1e-6
    a[:3] += [4.0, -3.0, 8.0]
    b[:3] += [4.0, -3.0, 8.0]
    assert m.geometry(a, b) == expected
    np.testing.assert_allclose(m.rotation_wxyz([2, 0, 0, 0]), np.eye(3))


def test_bbox_proxy_is_not_physical_insertion():
    box = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, 1], [-1, 1, 1]])
    result = m.geometry([0, 0, 2, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0], box, box)
    assert result["bbox_z_overlap_proxy_m"] == 0
    assert result["physical_insertion_depth_m"] is None
    assert result["plug_tip_hole_error_m"] is None
    assert m.geometry([0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0])["bbox_z_overlap_proxy_m"] is None
    with pytest.raises(ValueError):
        m.rotation_wxyz([0, 0, 0, 0])
    with pytest.raises(ValueError):
        m.geometry([np.nan, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0])


class Parser:
    def __init__(self):
        self.calls = []

    def is_A_depth_in_B(self, args):
        self.calls.append(("depth", args))
        return 1.0

    def is_A_in_B(self, args):
        self.calls.append(("inside", args))
        return 1.0

    def is_axis_up(self, args):
        self.calls.append(("upright", args))
        return 0.0

    def all_robot_back_to_origin(self, args):
        self.calls.append(("home", args))
        return 1.0


def environment():
    parser = Parser()
    pose = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    lm = SimpleNamespace(
        get_instance_name=lambda **kw: kw["label"],
        get_instance_pose=lambda **kw: (pose[:3], pose[3:]),
        get_instance_bbox_vertices=lambda **kw: None,
    )
    return (
        SimpleNamespace(
            scene_manager=SimpleNamespace(layout_manager=lm), reward_manager=SimpleNamespace(func_parser=parser)
        ),
        pose,
        parser,
    )


def test_direct_predicates_missing_contact_and_no_reward_mutation():
    env, _, parser = environment()
    out = m.snapshot(env, 3)
    assert out["native_instantaneous"] == dict(depth=True, inside=True, upright=False, home=True, all=False)
    assert out["held_by_arm"] is None and out["contact_force_n"] is None
    assert all(args["env_idx"] == 3 for _, args in parser.calls)
    assert parser.calls[0][1]["z_threshold"] == 0.015
    assert parser.calls[-1][1]["pos_threshold"] == 0.15
    # The stub intentionally has no reward step/check/reset methods.


def test_recording_copy_terminal_clock_and_refuse_overwrite(tmp_path):
    env, pose, _ = environment()
    recorder = m.ChargerDiagnostics(0.004)
    recorder.append(env, 0, 100, 0)
    recorder.append(env, 0, 100, 0)
    pose[0] = 4
    recorder.append(env, 0, 110, 1)
    assert len(recorder.rows) == 2 and recorder.rows[-1]["time_seconds"] == 0.04
    assert recorder.rows[0]["poses_env_local_m_wxyz"]["charger"][0] == 0
    path = tmp_path / "case.json"
    recorder.save(path, {"layout_id": 2}, success=False, action_count=1, step_limit=1)
    first = path.read_bytes()
    data = json.loads(first)
    assert data["terminal"]["truncated"] and not data["terminal"]["terminated"]
    with pytest.raises(FileExistsError):
        recorder.save(path, {}, success=True, action_count=1, step_limit=1)
    assert path.read_bytes() == first
    other = m.ChargerDiagnostics(0.004)
    assert not other.rows and other.origin_step is None


def test_limits_and_errors_are_not_false_predicates(tmp_path):
    env, pose, _ = environment()
    recorder = m.ChargerDiagnostics(0.004)
    recorder.MAX_SAMPLES = 1
    recorder.append(env, 0, 100, 0)
    recorder.append(env, 0, 110, 1)
    assert recorder.status == "truncated" and len(recorder.rows) == 1
    bad = m.ChargerDiagnostics(0.004)
    pose[0] = np.nan
    bad.append(env, 0, 100, 0)
    assert bad.status == "error" and not bad.rows and bad.errors
    bad.save(tmp_path / "bad.json", {}, success=False, action_count=1, step_limit=400)
    assert json.loads((tmp_path / "bad.json").read_text())["terminal"]["reason"] == "incomplete"
    env, _, _ = environment()
    clock = m.ChargerDiagnostics(0.004)
    clock.append(env, 0, 100, 3)
    clock.append(env, 0, 90, 0)
    assert clock.status == "error"


def test_annotation_transform_and_unsupported_scale():
    s = np.sqrt(0.5)
    p, r = m.transform_annotation([1, 2, 3, s, 0, 0, s], [2, 2, 2], [1, 0, 0, 1, 0, 0, 0])
    np.testing.assert_allclose(p, [1, 4, 3], atol=1e-12)
    np.testing.assert_allclose(r @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    with pytest.raises(ValueError, match="nonuniform"):
        m.transform_annotation([0, 0, 0, 1, 0, 0, 0], [1, 2, 1], [0, 0, 0, 1, 0, 0, 0])


def test_contact_window_impulses_separation_and_lost_event():
    w = m.ContactWindow()
    pair = ["charger", "finger", "charger/mesh", "finger/mesh"]
    w.add(pair, "found", [([3, 4, 0], -0.001)])
    w.add(pair, "persist", [([0, 0, 2], 0.003)])
    w.add(pair, "lost", [])
    row = w.drain()["pairs"][0]
    assert row["point_count"] == 2
    assert row["impulse_norm_sum_ns"] == 7 and row["impulse_norm_peak_ns"] == 5
    assert row["minimum_separation_m"] == -0.001
    assert row["events"] == dict(found=1, persist=1, lost=1)
    assert w.drain()["pairs"] == []
    w.add(pair, "found", [([0, np.nan, 0], 0)])
    assert w.drain()["status"] == "error"
    assert w.drain()["status"] == "error"  # no silent recovery of invalid measurement


def test_contact_limit_and_isolation():
    w, other = m.ContactWindow(), m.ContactWindow()
    for i in range(65):
        w.add([str(i), "b", "c", "d"], "found", [])
    out = w.drain()
    assert out["status"] == "error" and len(out["pairs"]) == 64
    assert other.drain()["status"] == "reported"


def test_pose_getter_tensor_detaches_before_numpy():
    import torch

    env, _, _ = environment()
    pos = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    quat = torch.tensor([1.0, 0.0, 0.0, 0.0], requires_grad=True)
    env.scene_manager.layout_manager.get_instance_pose = lambda **kw: (pos, quat)
    out = m.snapshot(env, 0)
    assert out["poses_env_local_m_wxyz"]["charger"] == [1, 2, 3, 1, 0, 0, 0]
    assert pos.grad is None and quat.grad is None


def test_same_observation_integrity_detects_mutation_without_advancing_sim():
    env, _, parser = environment()
    env.sim = SimpleNamespace(_sim_step_counter=100)
    env.take_action_cnt = [0]
    observation = {"image": np.zeros((3, 4, 5), dtype=np.uint8), "state": [0.0, 1.0]}
    recorder = m.ChargerDiagnostics(0.004)
    recorder.append(env, 0, 100, 0, observation=observation)
    assert recorder.rows[0]["observer_integrity"]["status"] == "unchanged"
    original = parser.is_A_in_B

    def corrupt(args):
        observation["image"][0, 0, 0] = 1
        return original(args)

    parser.is_A_in_B = corrupt
    recorder = m.ChargerDiagnostics(0.004)
    recorder.append(env, 0, 100, 0, observation=observation)
    assert recorder.status == "error" and "changed observation" in recorder.errors[0]["message"]
    assert env.sim._sim_step_counter == 100


def test_observation_digest_accounts_for_dtype_shape_and_nested_values():
    a = np.array([1, 2], dtype=np.uint8)
    assert m.observation_digest({"a": a, "b": None}) == m.observation_digest({"b": None, "a": a.copy()})
    assert m.observation_digest(a) != m.observation_digest(a.astype(np.int32))
    assert m.observation_digest(a) != m.observation_digest(a.reshape(1, 2))
    assert m.observation_digest({"a": {"b": 1}}) != m.observation_digest({"a": {}, "b": 1})
    assert m.observation_digest(np.int32(1)) != m.observation_digest(np.array(1, dtype=np.int32))
