"""Exercise real termination method with lightweight physics/reward adapters."""

import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_stack_bowls_metric import BOX, frame

from scripts.analyze_stack_bowls import analyze
from utils import stack_bowls_diagnostics as diagnostics
from utils.stack_bowls_metric import StableStack


def termination():
    path = Path(__file__).parents[1] / "src/eval_client/eval_env.py"
    tree = ast.parse(path.read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "is_episode_end")
    namespace = {"deepcopy": deepcopy}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["is_episode_end"]


def environment(mode, tmp_path, monkeypatch):
    env = SimpleNamespace(
        take_action_cnt=[0],
        end_flag=[False],
        success=[True],
        step_lim=800,
        num_envs=1,
        unstable_envs=set(),
        stack_bowls_metric_mode=mode,
        save_dir=str(tmp_path),
        eval_seed=0,
        env_seeds=[3],
        get_running_env_idx_list=lambda: [0],
        get_obs_batch=lambda **kw: None,
    )
    env.reward = 0.0
    env.reward_manager = SimpleNamespace(get_reward=lambda **kw: [env.reward])
    monkeypatch.setattr(diagnostics, "collision_boxes", lambda p: ({"link": BOX}, {"test": "fixture"}))
    monkeypatch.setattr(
        diagnostics,
        "snapshot",
        lambda e, i: json.loads(json.dumps(frame(e.take_action_cnt[i]), default=lambda a: a.tolist())),
    )
    env.stack_bowls_diagnostics = None if mode == "off" else diagnostics.StackBowlsDiagnostics(env, mode, "test")
    return env


@pytest.mark.parametrize("mode", ["off", "shadow"])
def test_native_stopping_unchanged_and_shadow_censored(mode, tmp_path, monkeypatch):
    env = environment(mode, tmp_path, monkeypatch)
    env.take_action_cnt = [1]
    env.reward = 1.0
    assert termination()(env) and env.success == [True]
    if mode == "shadow":
        out = env.stack_bowls_diagnostics.summary(0, 1, 800)
        assert out["stable_success"] is None and out["stable_censored"]
        env.stack_bowls_diagnostics.close()


def test_dual_latches_native_then_waits_for_release_window(tmp_path, monkeypatch):
    env = environment("dual", tmp_path, monkeypatch)
    for step in range(1, 26):
        env.take_action_cnt = [step]
        env.reward = float(step == 1)
        done = termination()(env)
        assert done == (step == 25)
        termination()(env)  # Duplicate termination queries do not advance time.
    assert env.success == [True]
    monitor = env.stack_bowls_diagnostics
    out = monitor.save(0, 0, 25, 800)
    monitor.close()
    assert out["native_first_step"] == 1 and out["stable_first_step"] == 25
    trace = next((tmp_path / "stack-bowls").glob("*.jsonl"))
    analysis = analyze(trace, {"link": BOX})
    assert analysis["stable_first_step"] == out["stable_first_step"]
    assert analysis["native_first_step"] == out["native_first_step"]


def test_stable_without_home_keeps_native_failure(tmp_path, monkeypatch):
    env = environment("dual", tmp_path, monkeypatch)
    for step in range(1, 801):
        env.take_action_cnt = [step]
        assert termination()(env) == (step == 800)
    assert env.success == [False]
    out = env.stack_bowls_diagnostics.summary(0, 800, 800)
    assert out["stable_success"] and out["native_success"] is False
    env.stack_bowls_diagnostics.close()


def test_config_and_resume_isolation():
    cfg = dict(
        task_name="stack_bowls", config_name="arx_x5", observation={"collect_freq": 25}, stack_bowls_metric="dual"
    )
    assert diagnostics.metric_mode(cfg) == "dual"
    for patch in ({"task_name": "stack_blocks"}, {"config_name": "piper"}, {"observation": {"collect_freq": 10}}):
        with pytest.raises(ValueError):
            diagnostics.metric_mode({**cfg, **patch})
    with pytest.raises(ValueError):
        diagnostics.metric_mode(cfg, {"stack_bowls_metric": "shadow"})
    with pytest.raises(ValueError):
        diagnostics.metric_mode(cfg, {})
    assert diagnostics.metric_mode({}, {}) == "off"


@pytest.mark.parametrize("bad", ["arms", "meshes", "nan"])
def test_incomplete_geometry_fails_closed(bad):
    f = frame()
    meshes = {"link": BOX}
    if bad == "arms":
        f["arms"] = {}
    if bad == "meshes":
        meshes = {}
    if bad == "nan":
        f["arms"]["left_arm"]["gripper_open_norm"] = float("nan")
    with pytest.raises(ValueError):
        StableStack(meshes).update(f)


def test_offline_rejects_missing_frames_and_preserves_censoring(tmp_path):
    p = tmp_path / "trace.jsonl"

    def write(rows):
        p.write_text("\n".join(json.dumps(x, default=lambda a: a.tolist()) for x in rows))

    meta = dict(type="metadata", seed=0, layout=0, initial_step=0)
    write([meta, frame(0)])
    assert analyze(p, {"link": BOX})["stable_success"] is None
    write([meta, frame(0), frame(2)])
    with pytest.raises(ValueError):
        analyze(p, {"link": BOX})


def test_snapshot_does_not_mutate_shared_bbox_and_maps_measured_opening():
    vertices = BOX.copy()
    robot = SimpleNamespace(type="target", arm_name="left_arm", gripper_scale=[-0.01, 0.044], gripper_move={"sign": 1})
    right = SimpleNamespace(**{**vars(robot), "arm_name": "right_arm"})
    obj = SimpleNamespace(get_local_scale=lambda: np.array([2.0, 2.0, 2.0]))
    lm = SimpleNamespace(
        get_instance_name=lambda i, label: label,
        get_scene_object=lambda i, name: obj,
        get_instance_pose=lambda i, **kw: ([0, 0, 0.1], [1, 0, 0, 0]),
        get_instance_bbox_vertices=lambda name, i: vertices,
    )
    rm = SimpleNamespace(
        robot_list=[robot, right],
        robot_key=[SimpleNamespace(body_names=["link"])] * 2,
        get_end_effector_real_val=lambda r, ids: {0: [0.044, 0.044]},
        get_link_pose=lambda r, name, ids, relative: {0: [1, 0, 0, 1, 0, 0, 0]},
    )
    env = SimpleNamespace(scene_manager=SimpleNamespace(layout_manager=lm), robot_manager=rm, take_action_cnt=[7])
    got = diagnostics.snapshot(env, 0)
    assert np.array_equal(vertices, BOX)
    assert got["arms"]["left_arm"]["gripper_open_norm"] == 1.0
    assert got["step"] == 7 and len(got["bowls"]) == 3
    json.dumps(got, allow_nan=False)
