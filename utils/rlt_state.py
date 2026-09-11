"""Opt-in charger state restoration for local, hash-pinned diagnostics.

This saves exposed physical/controller state, not private PhysX solver caches.
Oracle command repeatability must be checked before drawing learning conclusions.
"""

import copy
import hashlib
import pickle
import random
from pathlib import Path

import numpy as np
import torch


def _clone(value, device="cpu"):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device).clone()
    if isinstance(value, dict):
        return {k: _clone(v, device) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone(v, device) for v in value)
    if isinstance(value, list):
        return [_clone(v, device) for v in value]
    return copy.deepcopy(value)


def _validate(env):
    from utils.rlt_interaction import enabled

    if not enabled(env) or env.task_name not in {"plug_in_charger", "insert_tubes"} or env.num_envs != 1:
        raise ValueError("single supported RLT diagnostic required")
    if any(not q.is_empty() for q in env.robot_manager.control_manager.control_queue):
        raise ValueError("capture/restore requires an empty control queue")
    for name in ("_articulation_objects", "_garment_objects", "_fluid_objects"):
        if any(getattr(env.scene_manager, name)):
            raise ValueError("unsupported standalone articulated/deformable/fluid state")


def _physical(env):
    scene = env.sim.scene
    return dict(
        scene=_clone(scene.get_state()),
        targets={name: {key: _clone(getattr(robot.data, key)) for key in
                 ("joint_pos_target", "joint_vel_target", "joint_effort_target")}
                 for name, robot in scene.articulations.items()},
        rigid={name: dict(pose=_clone(obj.get_world_pose()),
                         linear=_clone(obj.get_linear_velocity()), angular=_clone(obj.get_angular_velocity()))
               for name, obj in env.scene_manager._rigid_and_dynamic_objects[0].items()},
    )


def capture(env, gate=None):
    _validate(env)
    if env.end_flag[0] or (gate is not None and len(gate.rows) != env.take_action_cnt[0] + 1):
        raise ValueError("live handoff and complete gate history required")
    reward = env.reward_manager
    return dict(
        schema="charger-exposed-state-v1" if env.task_name == "plug_in_charger" else "tubes-exposed-state-v1", task=env.task_name,
        layout=int(env.env_seeds[0]), seed=int(env.eval_seed), step_limit=int(env.step_lim),
        physical=_physical(env),
        env=_clone({k: getattr(env, k) for k in
                   ("success", "end_flag", "take_action_cnt", "rlt_applied_actions", "telemetry_actions")}),
        clock=_clone({k: getattr(env.sim, k) for k in
                     ("_sim_step_counter", "common_step_counter", "episode_length_buf")}),
        reward=_clone({k: v for k, v in vars(reward).items() if k not in {"env", "func_parser"}}),
        parser=_clone({k: getattr(reward.func_parser, k) for k in
                      ("pre_state", "robot_origin_endpose", "joint_ratio_transition_state")}),
        robot=_clone({k: getattr(env.robot_manager, k) for k in ("robot_origin_endpose", "robot_init_joint")}),
        robot_origins=_clone([r.base_link_origin_pose for r in env.robot_manager.robot_list]),
        prev_control=_clone(env.robot_manager.control_manager.prev_control),
        instruction=_clone(env.obs_manager.instruction),
        diagnostics=copy.deepcopy(env.charger_diagnostics if env.task_name == "plug_in_charger" else env.tubes_diagnostics),
        gate=dict(rows=copy.deepcopy(gate.rows), ee=copy.deepcopy(gate.ee)) if gate is not None else None,
        random=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all(),
    )


def save(state, path):
    path = Path(path)
    with path.open("xb") as stream:
        pickle.dump(state, stream, protocol=5)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path, sha256):
    # Pickle is strictly a trusted local experiment artifact, never a network payload.
    data = Path(path).read_bytes()
    if len(sha256) != 64 or hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError("state hash mismatch")
    state = pickle.loads(data)
    if state["schema"] not in {"charger-exposed-state-v1", "tubes-exposed-state-v1"}:
        raise ValueError("state schema")
    return state


def restore(env, state, gate=None):
    _validate(env)
    if (state["task"], state["layout"], state["seed"], state["step_limit"]) != (
            env.task_name, int(env.env_seeds[0]), int(env.eval_seed), env.step_lim):
        raise ValueError("state task/layout/seed/time limit mismatch")
    scene = env.sim.scene
    physical = state["physical"]
    objects = env.scene_manager._rigid_and_dynamic_objects[0]
    if set(physical["rigid"]) != set(objects) or set(physical["targets"]) != set(scene.articulations):
        raise ValueError("scene topology changed")
    env._abort_video_writers()
    env.telemetry.clear()
    scene.reset()
    scene.reset_to(_clone(physical["scene"], env.sim.device))
    for name, values in physical["targets"].items():
        robot = scene.articulations[name]
        robot.set_joint_position_target(_clone(values["joint_pos_target"], env.sim.device))
        robot.set_joint_velocity_target(_clone(values["joint_vel_target"], env.sim.device))
        robot.set_joint_effort_target(_clone(values["joint_effort_target"], env.sim.device))
    for name, values in physical["rigid"].items():
        obj = objects[name]
        obj.set_world_pose(*_clone(values["pose"]))
        obj.set_linear_velocity(_clone(values["linear"]))
        obj.set_angular_velocity(_clone(values["angular"]))
    scene.write_data_to_sim()
    scene.update(0.0)
    for target, source in ((env, "env"), (env.sim, "clock"), (env.reward_manager, "reward"),
                           (env.reward_manager.func_parser, "parser"), (env.robot_manager, "robot")):
        for key, value in state[source].items():
            setattr(target, key, _clone(value, env.sim.device))
    for robot, origin in zip(env.robot_manager.robot_list, state["robot_origins"], strict=True):
        robot.base_link_origin_pose = _clone(origin, env.sim.device)
    env.robot_manager.control_manager.prev_control = _clone(state["prev_control"])
    env.obs_manager.instruction = _clone(state["instruction"])
    if env.task_name == "plug_in_charger":
        env.charger_diagnostics = copy.deepcopy(state["diagnostics"])
    else:
        env.tubes_diagnostics = copy.deepcopy(state["diagnostics"])
    env.rlt_observations = {}
    if getattr(env, "charger_contact_observer", None) is not None:
        from utils.charger_diagnostics import ContactWindow

        env.charger_contact_observer.windows = {0: ContactWindow()}
    # Keep measured prefix history; Driver observes the restored current frame itself.
    if gate is not None:
        gate.rows = copy.deepcopy(state["gate"]["rows"][:-1])
        gate.ee = {k: copy.deepcopy(v[:-1]) for k, v in state["gate"]["ee"].items()}
        gate.last = None
    random.setstate(state["random"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])
    # Render only: no settling steps, no fresh time budget, no recorded image substitution.
    context = env.sim.sim
    context.physics_sim_view.update_articulations_kinematic()
    if env.use_fabric:
        context.forward()
    else:
        # IsaacLab forward() does nothing without Fabric. Tensor writes update
        # PhysX, but RGB still uses the old USD transforms until a physics step.
        # Explicitly publish existing poses; do not simulate a settling step.
        env.physics_interface.update_transformations(False, True, True, False)
    for _ in range(3):
        context.render()
    actual = _physical(env)
    errors = {}

    def compare(a, b, key):
        if isinstance(a, dict):
            for k in a:
                compare(a[k], b[k], key + "/" + k)
        elif isinstance(a, (tuple, list)):
            for i, (x, y) in enumerate(zip(a, b, strict=True)):
                compare(x, y, key + "/" + str(i))
        else:
            errors[key] = float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) if np.size(a) else 0.0

    compare(physical, actual, "physical")
    if max(errors.values(), default=0) > 1e-5:
        raise ValueError(f"restored exposed state mismatch: {errors}")
    return dict(step=int(env.take_action_cnt[0]), max_abs=max(errors.values(), default=0), fields=errors,
                visual_sync="fabric" if env.use_fabric else "physx-to-usd", render_mode=int(context.render_mode))
