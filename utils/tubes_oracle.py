"""Privileged geometry for a continuous, normal-command vertical tube Oracle."""

import numpy as np
from scipy.spatial.transform import Rotation

from utils.charger_diagnostics import cpu_array, rotation_wxyz, transform_annotation
from utils.charger_oracle import pose_matrix


def geometry(env):
    lm, rm = env.scene_manager.layout_manager, env.robot_manager
    objects = {}
    for label in ["tube0", "tube1", "tube2", "slot"]:
        name = lm.get_instance_name(label=label, env_idx=0)
        pos, quat = lm.get_instance_pose(inst_name=name, env_idx=0)
        meta = lm.get_instance_metadata(env_idx=0, inst_name=name)
        scale = cpu_array(lm.get_scene_object(env_idx=0, inst_name=name).get_local_scale())
        pose = np.r_[cpu_array(pos), cpu_array(quat)]
        local_bbox = cpu_array(lm.get_instance_bbox_vertices(inst_name=name, env_idx=0))
        bbox = (local_bbox * scale) @ rotation_wxyz(pose[3:]).T + pose[:3]
        objects[label] = dict(pose=pose.tolist(), bbox=bbox.tolist(), scale=scale.tolist())
        if label == "slot":
            holes = {}
            for key, value in meta["passive"]["support"].items():
                p, r = transform_annotation(pose, scale, value["center"][0])
                holes[key] = dict(position=p.tolist(), axis=r[:, 2].tolist())
            objects[label]["holes"] = holes
    hands = {}
    for side in ["left", "right"]:
        robot = rm.get_robot_by_arm_name(side + "_arm")
        hands[side] = cpu_array(rm.get_real_endpose(robot, [0], is_relative=True)[0]).tolist()
    return dict(objects=objects, hands=hands)


def choose_handoff(g, q, predicates, history):
    """Lift + stable hand/object relative pose; this is a geometric holding proxy."""
    if any(all(p.values()) for p in predicates):
        return None
    candidates = []
    for i in range(3):
        obj = pose_matrix(g["objects"][f"tube{i}"]["pose"])
        center = obj[:3, 3] + obj[:3, 1] * 0.07
        for side, off in [("left", 0), ("right", 7)]:
            ee = pose_matrix(g["hands"][side])
            rel = np.linalg.inv(ee) @ obj
            key = f"{i}-{side}"
            old = history.setdefault(key, [])
            old.append(rel)
            del old[:-5]
            # Fixed object/hand transform across five real control steps plus lift.
            stable = len(old) == 5 and max(np.linalg.norm(x[:3, 3] - rel[:3, 3]) for x in old) < 0.003
            rack_top = np.max(np.asarray(g["objects"]["slot"]["bbox"])[:, 2])
            lifted = min(np.asarray(g["objects"][f"tube{i}"]["bbox"])[:, 2]) > rack_top + 0.005
            hole = min(
                g["objects"]["slot"]["holes"],
                key=lambda h: np.linalg.norm(np.asarray(g["objects"]["slot"]["holes"][h]["position"])[:2] - obj[:2, 3]),
            )
            target = g["objects"]["slot"]["holes"][hole]
            upright = np.dot(obj[:3, 1], target["axis"]) > np.cos(np.deg2rad(15))
            nearby = np.linalg.norm(np.asarray(target["position"])[:2] - obj[:2, 3]) < 0.05
            if (
                stable
                and lifted
                and upright
                and nearby
                and np.linalg.norm(center - ee[:3, 3]) < 0.18
                and q[off + 6] < 0.8
            ):
                candidates.append(
                    dict(tube_id=i, side=side, hole=hole, distance=float(np.linalg.norm(center - ee[:3, 3])))
                )
    return min(candidates, key=lambda c: c["distance"]) if candidates else None


class TubesOracle:
    def __init__(self, env, config, q, g):
        self.env, self.config = env, dict(config)
        self.side = config["side"]
        self.offset = 0 if self.side == "left" else 7
        self.tube = config["tube_id"]
        self.phase = "align-above"
        self.stage_steps = 0
        self.standby = np.asarray(q, dtype=float).copy()
        self.grip = float(q[self.offset + 6])
        obj = pose_matrix(g["objects"][f"tube{self.tube}"]["pose"])
        ee = pose_matrix(g["hands"][self.side])
        self.object_to_ee = np.linalg.inv(obj) @ ee
        hole = g["objects"]["slot"]["holes"][config["hole"]]
        self.hole = np.asarray(hole["position"])
        self.axis = np.asarray(hole["axis"])
        self.axis /= np.linalg.norm(self.axis)
        # Rotate current tube Y onto hole Z with the smallest orientation change.
        v = obj[:3, 1]
        cross = np.cross(v, self.axis)
        norm = np.linalg.norm(cross)
        angle = np.arctan2(norm, np.dot(v, self.axis))
        if norm < 1e-8 and np.dot(v, self.axis) < 0:
            raise ValueError("inverted tube handoff")
        delta = Rotation.from_rotvec(cross / max(norm, 1e-8) * angle).as_matrix()
        self.rotation = delta @ obj[:3, :3]
        self.release_q = None
        self.lift_start = None
        self.joint_target = self.standby[self.offset : self.offset + 6].copy()

    def propose(self, q, g, predicates):
        q = np.asarray(q, dtype=float)
        off = self.offset
        obj = pose_matrix(g["objects"][f"tube{self.tube}"]["pose"])
        ee = pose_matrix(g["hands"][self.side])
        lateral = np.linalg.norm((obj[:3, 3] - self.hole) - self.axis * np.dot(obj[:3, 3] - self.hole, self.axis))
        tilt = np.rad2deg(np.arccos(np.clip(np.dot(obj[:3, 1], self.axis), -1, 1)))
        self.stage_steps += 1
        if (
            self.phase == "align-above"
            and lateral < self.config.get("alignment_m", 0.002)
            and tilt < self.config.get("alignment_deg", 2)
            and np.dot(obj[:3, 3] - self.hole, self.axis) > 0.018
        ):
            self.phase = "insert"
            self.stage_steps = 0
        if self.phase == "insert" and all(predicates[self.tube].values()) and tilt < 3:
            self.phase = "release"
            self.stage_steps = 0
            self.release_q = q.copy()
        if self.phase == "release" and self.stage_steps >= 10:
            self.phase = "lift"
            self.stage_steps = 0
            self.lift_start = ee.copy()
            self.joint_target = q[off : off + 6].copy()
        desired = ee.copy()
        goal = self.standby.copy()
        if self.phase in ["align-above", "insert"]:
            target = obj.copy()
            target[:3, :3] = self.rotation
            target[:3, 3] = self.hole + self.axis * (0.03 if self.phase == "align-above" else -0.05)
            # Refresh the measured grasp transform: a frozen transform leaves a
            # persistent object error after any small slip during alignment.
            desired = target @ np.linalg.inv(obj) @ ee
            goal[off + 6] = self.grip
        elif self.phase == "release":
            goal[off : off + 6] = self.release_q[off : off + 6]
            goal[off + 6] = 1
            return goal, dict(phase=self.phase, lateral_m=float(lateral), tilt_deg=float(tilt))
        else:
            desired = self.lift_start.copy()
            desired[:3, 3] += self.axis * 0.22
            goal[off + 6] = 1
        movement = desired[:3, 3] - ee[:3, 3]
        speed = self.config.get("translation_m", 0.004)
        if self.phase == "lift":
            speed = self.config.get("lift_translation_m", speed)
        desired[:3, 3] = ee[:3, 3] + movement * min(1, speed / max(np.linalg.norm(movement), 1e-9))
        rv = Rotation.from_matrix(desired[:3, :3] @ ee[:3, :3].T).as_rotvec()
        desired[:3, :3] = (
            Rotation.from_rotvec(rv * min(1, np.deg2rad(3) / max(np.linalg.norm(rv), 1e-9))).as_matrix() @ ee[:3, :3]
        )
        quat = Rotation.from_matrix(desired[:3, :3]).as_quat()
        pose = np.r_[desired[:3, 3] + cpu_array(self.env.robot_manager.scene.env_origins[0]), quat[3], quat[:3]]
        robot = self.env.robot_manager.get_robot_by_arm_name(self.side + "_arm")
        if self.config.get("solver") == "jacobian":
            # Local Cartesian servo avoids the global IK solver's 1 mm goal
            # tolerance, which is comparable to this hole's radial clearance.
            rm = self.env.robot_manager
            articulation = rm.robot_key[rm.robot_list.index(robot)]
            body = articulation.body_names.index(robot.ee_link_name)
            jac = cpu_array(articulation.root_physx_view.get_jacobians()[0])
            body_slot = body - 1 if articulation.is_fixed_base else body
            names = rm.ik_solver[robot.robot_name].active_joints_name
            columns = [articulation.joint_names.index(name) for name in names]
            if not articulation.is_fixed_base:
                raise ValueError("tubes Cartesian Oracle requires a fixed base")
            jac = jac[body_slot][:, columns]
            # PhysX Jacobians use the COM; observations use the link origin.
            offset = cpu_array(articulation.data.body_link_pos_w[0, body] - articulation.data.body_com_pos_w[0, body])
            jac[:3] += np.cross(jac[3:].T, offset).T
            error = np.r_[desired[:3, 3] - ee[:3, 3], Rotation.from_matrix(desired[:3, :3] @ ee[:3, :3].T).as_rotvec()]
            delta = jac.T @ np.linalg.solve(jac @ jac.T + np.eye(6) * 0.002**2, error)
            if self.config.get("integral_servo", False):
                self.joint_target += np.clip(delta * 0.5, -0.02, 0.02)
                self.joint_target = np.clip(self.joint_target, q[off : off + 6] - 0.07, q[off : off + 6] + 0.07)
                goal[off : off + 6] = self.joint_target
            else:
                goal[off : off + 6] = q[off : off + 6] + np.clip(delta, -0.035, 0.035)
            return goal, dict(phase=self.phase, lateral_m=float(lateral), tilt_deg=float(tilt), solver="jacobian")
        solution = self.env.robot_manager.solve_ik(pose.tolist(), 0, robot, trans="world")
        if solution["status"] != "Success":
            raise RuntimeError("tubes Oracle IK failed")
        goal[off : off + 6] = q[off : off + 6] + np.clip(
            np.asarray(solution["joint_value"]) - q[off : off + 6], -0.035, 0.035
        )
        return goal, dict(phase=self.phase, lateral_m=float(lateral), tilt_deg=float(tilt))


def released_clear(g, q, config):
    off = 0 if config["side"] == "left" else 7
    axis = np.asarray(g["objects"]["slot"]["holes"][config["hole"]]["axis"])
    hand = np.asarray(g["hands"][config["side"]])[:3]
    vertices = np.asarray(g["objects"][f"tube{config['tube_id']}"]["bbox"])
    clearance = float(hand @ axis - np.max(vertices @ axis))
    # EE-to-fingertip offset must be covered by this frozen conservative clearance.
    return bool(q[off + 6] >= 0.8 and clearance >= config.get("ee_clearance_m", 0.12)), clearance
