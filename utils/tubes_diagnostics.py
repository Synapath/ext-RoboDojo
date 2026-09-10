"""Read-only insertion predicates; never modify native reward queues or observations."""

import json
from pathlib import Path


def snapshot(env, env_idx):
    parser = env.reward_manager.func_parser
    tubes = []
    for i in range(3):
        ab = dict(env_idx=env_idx, label_A=f"tube{i}", label_B="slot")
        values = {
            "depth": parser.is_A_depth_in_B(dict(ab, z_threshold=0.045)),
            "inside": parser.is_A_in_B(dict(ab)),
            "upright": parser.is_axis_up(dict(env_idx=env_idx, label=f"tube{i}", axis=[0, 1, 0], threshold=30)),
        }
        if any(float(v) not in (0.0, 1.0) for v in values.values()):
            raise ValueError("nonbinary tubes predicate")
        tubes.append({k: bool(v) for k, v in values.items()})
    return {
        "control_step": int(env.take_action_cnt[env_idx]),
        "physics_step": int(env.sim._sim_step_counter),
        "tubes": tubes,
        "home": bool(parser.all_robot_back_to_origin(dict(env_idx=env_idx, pos_threshold=0.15, rot_threshold=20))),
        "all_grippers_open": bool(parser.is_all_gripper_open(dict(env_idx=env_idx, open_threshold=0.8))),
    }


class TubesDiagnostics:
    def __init__(self):
        self.frames = []

    def append(self, env, env_idx):
        frame = snapshot(env, env_idx)
        if self.frames and frame["control_step"] == self.frames[-1]["control_step"]:
            self.frames[-1] = frame
        else:
            if self.frames and frame["control_step"] < self.frames[-1]["control_step"]:
                raise ValueError("tubes trace crossed episode boundary")
            self.frames.append(frame)

    def save(self, path, identity, *, success, action_count, step_limit):
        value = dict(
            schema="tubes-diagnostics-v1",
            identity=identity,
            frames=self.frames,
            native_success=bool(success),
            action_count=int(action_count),
            step_limit=int(step_limit),
        )
        with Path(path).open("x") as f:
            json.dump(value, f, allow_nan=False)
