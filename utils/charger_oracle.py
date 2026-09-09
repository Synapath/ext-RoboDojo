"""Opt-in, state-based charger correction through the native joint controller.

This is a candidate controller, not a guaranteed expert. Asset annotations give
targets; success is read independently from the unchanged native environment.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from utils.charger_diagnostics import cpu_array, rotation_wxyz, transform_annotation


def pose_matrix(pose):
    out = np.eye(4)
    out[:3, :3] = rotation_wxyz(np.asarray(pose)[3:])
    out[:3, 3] = np.asarray(pose)[:3]
    return out


def choose_target(targets):
    """Choose a pose-compatible insertion frame, including symmetric annotations."""
    candidates = [t for t in targets if t["rotation_error_deg"] <= 30
                  and t["lateral_m"] <= 0.03 and -0.01 <= t["axial_m"] <= 0.12]
    return min(candidates, key=lambda t: t["lateral_m"] + 0.1 * abs(t["axial_m"])
               + 0.0005 * t["rotation_error_deg"]) if candidates else None


def target_end_link(current_object, current_ee, insert_frame, target_frame,
                    axial_offset, max_translation, max_rotation_deg):
    """Preserve measured object-to-hand transform; bound each proposed change."""
    obj, ee = pose_matrix(current_object), pose_matrix(current_ee)
    insert, target = insert_frame, target_frame.copy()
    target[:3, 3] += target[:3, 2] * axial_offset
    delta = target @ np.linalg.inv(insert)
    vector = Rotation.from_matrix(delta[:3, :3]).as_rotvec()
    vector *= min(1.0, np.deg2rad(max_rotation_deg) / max(np.linalg.norm(vector), 1e-12))
    rotation = Rotation.from_rotvec(vector).as_matrix()
    desired_obj = delta @ obj
    movement = desired_obj[:3, 3] - obj[:3, 3]
    movement *= min(1.0, max_translation / max(np.linalg.norm(movement), 1e-12))
    desired_obj[:3, 3] = obj[:3, 3] + movement
    desired_obj[:3, :3] = rotation @ obj[:3, :3]
    desired_ee = desired_obj @ np.linalg.inv(obj) @ ee
    xyzw = Rotation.from_matrix(desired_ee[:3, :3]).as_quat()
    return np.r_[desired_ee[:3, 3], xyzw[3], xyzw[:3]]


class ChargerOracle:
    def __init__(self, env, config):
        self.env, self.config = env, dict(config)
        if config.get("strategy") not in {"immediate", "stalled", "recovery_pulse"}:
            raise ValueError("unknown Oracle strategy")
        if config.get("strategy") == "recovery_pulse":
            n = config.get("actor_prefix_steps")
            if type(n) is not int or not 1 <= n < 10:
                raise ValueError("actor_prefix_steps must be an integer in [1, 9]")
            period = config.get("oracle_chunks_per_pulse", 1)
            if type(period) is not int or period < 1:
                raise ValueError("oracle_chunks_per_pulse must be a positive integer")
            for key, default in (("recovery_shift_m", 0.01), ("recovery_rotation_deg", 10),
                                 ("recovery_steps", 10)):
                value = config.get(key, default)
                if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                    raise ValueError("invalid recovery limit: " + key)
        self.holder = None
        self.target_key = None
        self.started = None
        self.progress = []
        self.finishing = None
        self.grip = None
        self.settled_chunks = 0
        self.completed = False
        self.pulse_due = True
        self.oracle_chunks_since_pulse = 0
        self.hold_anchor = None
        self.hold_step = None
        self.pulse_stop_reason = None

    def _relative_object_pose(self, frame, holder):
        rm = self.env.robot_manager
        robot = rm.get_robot_by_arm_name(holder + "_arm")
        ee = rm.get_real_endpose(robot, [0], is_relative=True)[0]
        obj = frame["diagnostics"]["poses_env_local_m_wxyz"]["charger"]
        return np.linalg.inv(pose_matrix(ee)) @ pose_matrix(obj)

    def _pulse_gate(self, frame, gate_row):
        """Recover only a recently observed, still contact-supported holder."""
        holder = {"L": "left", "R": "right"}.get(gate_row.get("holding_proxy"))
        if holder:
            if self.holder is not None and self.holder != holder:
                return None, {"reason": "holder_changed"}
            self.holder = holder
            self.hold_anchor = self._relative_object_pose(frame, holder)
            self.hold_step = frame["step"]
            if self.grip is None:
                self.grip = float(frame["proprio"][6 if holder == "left" else 13])
            return gate_row, {"holder_mode": "stable_proxy"}
        if self.hold_anchor is None or frame["step"] - self.hold_step > self.config.get("recovery_steps", 10):
            return None, {"reason": "no_recent_holder"}
        side = "L" if self.holder == "left" else "R"
        contacts = gate_row.get("bilateral_contacts", {})
        if not contacts.get(side) or contacts.get("R" if side == "L" else "L"):
            return None, {"reason": "no_unambiguous_bilateral_contact"}
        relative = self._relative_object_pose(frame, self.holder)
        shift = float(np.linalg.norm(relative[:3, 3] - self.hold_anchor[:3, 3]))
        rotation = float(np.rad2deg(Rotation.from_matrix(
            self.hold_anchor[:3, :3].T @ relative[:3, :3]).magnitude()))
        detail = {"holder_mode": "contact_recovery", "relative_shift_m": shift,
                  "relative_rotation_deg": rotation}
        if shift > self.config.get("recovery_shift_m", 0.01) or rotation > self.config.get("recovery_rotation_deg", 10):
            return None, {**detail, "reason": "relative_grasp_drift"}
        return {**gate_row, "holding_proxy": side}, detail

    def _frames(self, row, selected):
        lm = self.env.scene_manager.layout_manager
        result = []
        for label in ("charger", "socket"):
            name = lm.get_instance_name(label=label, env_idx=0)
            meta = lm.get_instance_metadata(env_idx=0, inst_name=name)
            scale = cpu_array(lm.get_scene_object(env_idx=0, inst_name=name).get_local_scale())
            annotation = (meta["active"]["functional"]["insert"]["frame"][selected["insert_frame"]]
                          if label == "charger" else
                          meta["passive"]["support"][selected["support"]]["center"][selected["support_frame"]])
            p, r = transform_annotation(row["poses_env_local_m_wxyz"][label], scale, annotation)
            frame = np.eye(4)
            frame[:3, :3], frame[:3, 3] = r, p
            result.append(frame)
        return result

    def propose(self, frame, gate_row, actor_proposal):
        if self.config["strategy"] != "recovery_pulse":
            return self._correction(frame, gate_row, actor_proposal)
        meta = {"strategy": "recovery_pulse", "bc_eligible": False}
        if self.completed:
            return None, {**meta, "reason": "correction_complete"}
        if self.started is not None and frame["step"] - self.started >= self.config.get("max_intervention_steps", 180):
            self.pulse_stop_reason = "intervention_limit"
        if self.pulse_stop_reason:
            return None, {**meta, "reason": self.pulse_stop_reason}
        if self.finishing is None:
            recovered, detail = self._pulse_gate(frame, gate_row)
            if recovered is None:
                return None, {**meta, **detail}
            # Only stable states permit another actor pulse. A contact recovery
            # keeps correcting until the ordinary holding proxy is stable again.
            if self.pulse_due and gate_row.get("holding_proxy") and not gate_row.get("native_insertion_three"):
                self.pulse_due = False
                self.oracle_chunks_since_pulse = 0
                return None, {**meta, **detail, "reason": "actor_pulse",
                              "actor_prefix_steps": self.config["actor_prefix_steps"]}
            gate_row = recovered
        else:
            detail = {"holder_mode": "finishing"}
        command, info = self._correction(frame, gate_row, actor_proposal)
        if command is not None:
            self.oracle_chunks_since_pulse += 1
            self.pulse_due = self.oracle_chunks_since_pulse >= self.config.get("oracle_chunks_per_pulse", 1)
        return command, {**info, **detail}

    def _correction(self, frame, gate_row, actor_proposal):
        meta = {"strategy": self.config["strategy"], "reason": "actor", "bc_eligible": False}
        step = frame["step"]
        if self.completed:
            return None, {**meta, "reason": "correction_complete"}
        holder = {"L": "left", "R": "right"}.get(gate_row.get("holding_proxy"))
        if self.finishing is None and holder not in {"left", "right"}:
            self.started = None
            return None, {**meta, "reason": "no_stable_holder"}
        if self.started is not None and self.finishing is None and holder != self.holder:
            self.started = None
            return None, {**meta, "reason": "holder_changed"}
        targets = frame["diagnostics"].get("annotated_alignment", {}).get("targets", [])
        if not targets:
            return None, {**meta, "reason": "missing_target"}
        key = lambda t: (t["insert_frame"], t["support"], t["support_frame"])
        selected = next((t for t in targets if key(t) == self.target_key), None)
        if selected is None:
            selected = choose_target(targets)
        if selected is None:
            return None, {**meta, "reason": "no_pose_compatible_target"}
        progress = selected["lateral_m"] + abs(selected["axial_m"])
        self.progress.append((step, progress))
        window = int(self.config.get("stall_steps", 20))
        self.progress = [(s, p) for s, p in self.progress if step - s <= window]
        if self.started is None and self.finishing is None:
            if self.config["strategy"] == "stalled":
                stalled = (step - self.progress[0][0] >= window and
                           self.progress[0][1] - progress < self.config.get("progress_m", 0.002))
                if not stalled:
                    return None, meta
            self.started, self.holder, self.target_key = step, holder, key(selected)
            if self.config["strategy"] != "recovery_pulse" or self.grip is None:
                self.grip = float(frame["proprio"][6 if holder == "left" else 13])
        if step - self.started >= self.config.get("max_intervention_steps", 180):
            return None, {**meta, "reason": "intervention_limit"}
        # A correction ends on an observable state, rather than requiring the
        # actor to imitate repeated idle commands until an invisible timer ends.
        native = frame["diagnostics"]["native_instantaneous"]
        settled = (selected["lateral_m"] <= self.config.get("settled_lateral_m", 0.0015)
                   and abs(selected["axial_m"] - self.config.get("insert_m", -0.008)) <= 0.001
                   and selected["rotation_error_deg"] <= 2
                   and native["inside"] and native["upright"])
        self.settled_chunks = self.settled_chunks + 1 if settled else 0
        if self.config.get("yield_when_settled", False) and self.settled_chunks >= 2:
            self.completed = True
            return None, {**meta, "reason": "correction_complete"}
        rm = self.env.robot_manager
        robot = rm.get_robot_by_arm_name(self.holder + "_arm")
        offset = 0 if self.holder == "left" else 7
        q = np.asarray(frame["proprio"], dtype=float)
        goal = q.copy()
        goal[offset + 6] = self.grip
        if gate_row.get("native_insertion_three") and self.finishing is None:
            self.finishing = step
        if self.finishing is not None:
            goal[offset + 6] = 1.0
            if step - self.finishing >= 10:
                for side, arm_offset in (("left", 0), ("right", 7)):
                    r = rm.get_robot_by_arm_name(side + "_arm")
                    entity = rm.robot_key[rm.robot_list.index(r)]
                    goal[arm_offset:arm_offset + 6] = cpu_array(entity.data.default_joint_pos[0, r.arm_joint_indices])
                    goal[arm_offset + 6] = 1.0
            reason = "release_home"
        else:
            a, b = self._frames(frame["diagnostics"], selected)
            align = selected["lateral_m"] > self.config.get("alignment_m", 0.002) or selected["rotation_error_deg"] > 3
            axial = self.config.get("approach_m", 0.012) if align else self.config.get("insert_m", -0.008)
            ee = rm.get_real_endpose(robot, [0], is_relative=True)[0]
            pose = target_end_link(frame["diagnostics"]["poses_env_local_m_wxyz"]["charger"], ee, a, b,
                                   axial, self.config.get("translation_m", 0.008),
                                   self.config.get("rotation_deg", 6.0))
            pose[:3] += cpu_array(rm.scene.env_origins[0])
            solution = rm.solve_ik(pose.tolist(), 0, robot, trans="world")
            if solution["status"] != "Success":
                return None, {**meta, "reason": "ik_failed", "target_pose": pose.tolist()}
            goal[offset:offset + 6] = solution["joint_value"]
            reason = "align" if align else "insert"
        delta = goal - q
        arm = np.ones(14, dtype=bool)
        arm[[6, 13]] = False
        scale = min(1.0, 10 * self.config.get("joint_step_rad", 0.04) / max(np.max(np.abs(delta[arm])), 1e-12))
        delta[arm] *= scale
        commands = q[None] + np.arange(1, 11)[:, None] / 10 * delta[None]
        if not np.isfinite(commands).all():
            raise ValueError("nonfinite Oracle command")
        return commands, {**meta, "reason": reason, "bc_eligible": True, "holder": self.holder,
                          "target": list(self.target_key), "lateral_m": selected["lateral_m"],
                          "axial_m": selected["axial_m"], "rotation_error_deg": selected["rotation_error_deg"]}
