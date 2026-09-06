"""Read-only charger measurements, separate from policy observations and reward state."""

import json
import hashlib
import math
import os
import weakref
from pathlib import Path

import numpy as np


def cpu_array(value):
    """Existing Isaac getters can return CPU or CUDA tensors."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def rotation_wxyz(quaternion):
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError("invalid quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def geometry(charger_pose, socket_pose, charger_bbox=None, socket_bbox=None):
    a, b = np.asarray(charger_pose, dtype=float), np.asarray(socket_pose, dtype=float)
    if a.shape != (7,) or b.shape != (7,) or not np.isfinite([a, b]).all():
        raise ValueError("invalid object pose")
    ra, rb = rotation_wxyz(a[3:]), rotation_wxyz(b[3:])
    result = {
        "charger_root_in_socket_frame_m": (rb.T @ (a[:3] - b[:3])).tolist(),
        "charger_axis_up_angle_deg": float(np.degrees(np.arccos(np.clip(ra[2, 1], -1.0, 1.0)))),
        "bbox_z_overlap_proxy_m": None,
        "plug_tip_hole_error_m": None,
        "physical_insertion_depth_m": None,
    }
    if charger_bbox is not None and socket_bbox is not None:
        aa, bb = np.asarray(charger_bbox, dtype=float), np.asarray(socket_bbox, dtype=float)
        for box in (aa, bb):
            if box.ndim != 2 or box.shape[1] != 3 or len(box) < 4 or not np.isfinite(box).all():
                raise ValueError("invalid bounding box")
        az = (aa @ ra.T + a[:3])[:, 2]
        bz = (bb @ rb.T + b[:3])[:, 2]
        result["bbox_z_overlap_proxy_m"] = float(bz.max() - az.min())
    return result


def snapshot(env, env_idx):
    """Call only existing pose getters and four instantaneous, read-only predicates."""
    lm, parser = env.scene_manager.layout_manager, env.reward_manager.func_parser
    poses, boxes = {}, {}
    for label in ("charger", "socket"):
        name = lm.get_instance_name(label=label, env_idx=env_idx)
        if name is None:
            raise ValueError(f"missing {label} instance")
        pos, quat = lm.get_instance_pose(inst_name=name, env_idx=env_idx)
        poses[label] = np.concatenate([cpu_array(pos), cpu_array(quat)]).tolist()
        boxes[label] = lm.get_instance_bbox_vertices(inst_name=name, env_idx=env_idx)
    metrics = geometry(poses["charger"], poses["socket"], boxes["charger"], boxes["socket"])
    ab = {"env_idx": env_idx, "label_A": "charger", "label_B": "socket"}
    native = {
        "depth": parser.is_A_depth_in_B(dict(ab, z_threshold=0.015)),
        "inside": parser.is_A_in_B(dict(ab)),
        "upright": parser.is_axis_up({"env_idx": env_idx, "label": "charger", "axis": [0, 1, 0], "threshold": 10}),
        "home": parser.all_robot_back_to_origin({"env_idx": env_idx, "pos_threshold": 0.15, "rot_threshold": 20}),
    }
    if any(float(value) not in (0.0, 1.0) for value in native.values()):
        raise ValueError("non-binary native predicate")
    native = {key: bool(value) for key, value in native.items()}
    annotations = None
    if hasattr(lm, "get_instance_metadata"):
        annotations = annotated_alignment(lm, env_idx, poses)
    contacts = getattr(env, "charger_contact_observer", None)
    return {
        "poses_env_local_m_wxyz": poses,
        "geometry": metrics,
        "native_instantaneous": dict(native, all=all(native.values())),
        "annotated_alignment": annotations,
        "contact_reports": contacts.drain(env_idx) if contacts is not None else None,
        "contact_force_n": None,
        "held_by_arm": None,
        "missing": {
            "contact_force_n": "no calibrated force sensor; optional reports contain impulses, not forces",
            "held_by_arm": "no calibrated grasp/contact source; gripper commands are insufficient",
            "plug_tip_hole_error_m": "asset functional points not yet calibrated",
            "physical_insertion_depth_m": "bbox overlap is not plug-tip penetration",
        },
    }


def annotated_alignment(lm, env_idx, poses):
    """Asset annotation distances; deliberately not a calibrated pin-tip sensor."""
    metadata, scales, sources = {}, {}, {}
    for label in ("charger", "socket"):
        name = lm.get_instance_name(label=label, env_idx=env_idx)
        item = lm.get_instance_metadata(env_idx=env_idx, inst_name=name)
        if item is None:
            return {"status": "missing", "reason": f"missing {label} metadata"}
        obj = lm.get_scene_object(env_idx=env_idx, inst_name=name)
        scale = cpu_array(obj.get_local_scale())
        if scale.shape != (3,) or not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("invalid annotation scale")
        metadata[label], scales[label] = item, scale
        sources[label] = {
            "metadata_sha256": hashlib.sha256(json.dumps(item, sort_keys=True, allow_nan=False).encode()).hexdigest(),
            "scale": scale.tolist(),
            "prim_path": str(obj.prim_path),
        }
    frames = metadata["charger"].get("active", {}).get("functional", {}).get("insert", {}).get("frame", [])
    supports = metadata["socket"].get("passive", {}).get("support", {})
    rows = []
    for i, frame in enumerate(frames):
        a, ar = transform_annotation(poses["charger"], scales["charger"], frame)
        for label, item in sorted(supports.items()):
            for j, target in enumerate(item.get("center", [])):
                b, br = transform_annotation(poses["socket"], scales["socket"], target)
                error = br.T @ (a - b)
                angle = np.degrees(np.arccos(np.clip((np.trace(br.T @ ar) - 1) / 2, -1, 1)))
                rows.append(
                    {
                        "insert_frame": i,
                        "support": label,
                        "support_frame": j,
                        "error_in_target_frame_m": error.tolist(),
                        "lateral_m": float(np.linalg.norm(error[:2])),
                        "axial_m": float(error[2]),
                        "rotation_error_deg": float(angle),
                    }
                )
    return {"status": "asset_annotation_uncalibrated" if rows else "missing", "sources": sources, "targets": rows}


def transform_annotation(pose, scale, frame):
    pose, scale, frame = (np.asarray(v, dtype=float) for v in (pose, scale, frame))
    if (
        pose.shape != (7,)
        or frame.shape != (7,)
        or scale.shape != (3,)
        or not all(np.isfinite(v).all() for v in (pose, scale, frame))
        or (scale <= 0).any()
    ):
        raise ValueError("invalid annotated frame")
    r = rotation_wxyz(pose[3:])
    # Nonuniform scaling changes axes: orthogonal annotation frames need calibration.
    if not np.allclose(scale, scale[0], atol=1e-8, rtol=0):
        raise ValueError("nonuniform annotation scale requires calibration")
    return pose[:3] + r @ (scale * frame[:3]), r @ rotation_wxyz(frame[3:])


class ContactWindow:
    """Bounded raw contact aggregation; no grasp inference or force conversion."""

    def __init__(self):
        self.pairs = {}
        self.error = None

    def add(self, paths, event, points):
        if self.error is not None:
            return
        try:
            key = tuple(paths)
            if len(points) > 4096 or (key not in self.pairs and len(self.pairs) >= 64):
                raise ValueError("contact window limit exceeded")
            row = self.pairs.setdefault(
                key,
                {
                    "paths": list(paths),
                    "events": {},
                    "point_count": 0,
                    "impulse_norm_sum_ns": 0.0,
                    "impulse_norm_peak_ns": 0.0,
                    "minimum_separation_m": None,
                },
            )
            row["events"][str(event)] = row["events"].get(str(event), 0) + 1
            for impulse, separation in points:
                impulse = np.asarray(impulse, dtype=float)
                if impulse.shape != (3,) or not np.isfinite(impulse).all() or not math.isfinite(separation):
                    raise ValueError("invalid contact data")
                norm = float(np.linalg.norm(impulse))
                row["point_count"] += 1
                row["impulse_norm_sum_ns"] += norm
                row["impulse_norm_peak_ns"] = max(norm, row["impulse_norm_peak_ns"])
                old = row["minimum_separation_m"]
                row["minimum_separation_m"] = float(separation if old is None else min(old, separation))
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def drain(self):
        out = {
            "status": "error" if self.error else "reported",
            "error": self.error,
            "pairs": list(self.pairs.values()),
            "scope": "since_previous_observation",
        }
        self.pairs = {}
        return out


class ContactObserver:
    """Pinned PhysX contact-report subscription, owned and released by one env."""

    def __init__(self, env):
        self.windows = {i: ContactWindow() for i in range(env.num_envs)}
        self.subscription = None
        self.paths = {}
        self.callbacks = 0
        self.headers_seen = 0
        self.headers_matched = 0
        self.settings = None
        self.previous_contact_processing = None
        try:
            from pxr import PhysxSchema
            import carb.settings
            import omni.usd
            from omni.physx import get_physx_simulation_interface

            stage = omni.usd.get_context().get_stage()
            self.settings = carb.settings.get_settings()
            self.previous_contact_processing = self.settings.get_as_bool("/physics/disableContactProcessing")
            self.settings.set_bool("/physics/disableContactProcessing", False)
            lm = env.scene_manager.layout_manager
            for i in range(env.num_envs):
                name = lm.get_instance_name(label="charger", env_idx=i)
                obj = lm.get_scene_object(env_idx=i, inst_name=name)
                path = str(obj.prim_path)
                prim = stage.GetPrimAtPath(path)
                if not prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                    raise ValueError("charger contact reporting was not enabled before physics initialization")
                self.paths[path] = i
            # Weak callback avoids retaining a retired Isaac environment via subscription.
            owner = weakref.ref(self)

            def callback(headers, data):
                current = owner()
                if current is not None:
                    current.receive(headers, data)

            self.subscription = get_physx_simulation_interface().subscribe_contact_report_events(callback)
        except Exception as exc:
            for window in self.windows.values():
                window.error = f"contact initialization: {type(exc).__name__}: {exc}"

    def receive(self, headers, data):
        try:
            self.callbacks += 1
            self.headers_seen += len(headers)
            from pxr import PhysicsSchemaTools

            for header in headers:
                paths = [
                    str(PhysicsSchemaTools.intToSdfPath(getattr(header, field)))
                    for field in ("actor0", "actor1", "collider0", "collider1")
                ]
                ids = {i for root, i in self.paths.items() if any(p == root or p.startswith(root + "/") for p in paths)}
                if not ids:
                    continue
                self.headers_matched += 1
                count, offset = header.num_contact_data, header.contact_data_offset
                if count > 4096 or count < 0 or (count > 0 and (offset < 0 or offset + count > len(data))):
                    raise ValueError("contact callback data limit/range")
                points = [(list(data[k].impulse), float(data[k].separation)) for k in range(offset, offset + count)]
                for i in ids:
                    self.windows[i].add(paths, str(header.type), points)
        except Exception as exc:
            for window in self.windows.values():
                window.error = f"contact callback: {type(exc).__name__}: {exc}"

    def drain(self, env_idx):
        result = self.windows[env_idx].drain()
        result.update(callback_count=self.callbacks, headers_seen=self.headers_seen, headers_matched=self.headers_matched)
        return result

    def close(self):
        self.subscription = None
        if self.settings is not None and self.previous_contact_processing is not None:
            self.settings.set_bool("/physics/disableContactProcessing", self.previous_contact_processing)
        self.settings = None
        self.windows.clear()


class ChargerDiagnostics:
    MAX_SAMPLES = 1001
    MAX_BYTES = 8 * 1024 * 1024

    def __init__(self, physics_dt):
        if not math.isfinite(physics_dt) or physics_dt <= 0:
            raise ValueError("invalid physics dt")
        self.dt = float(physics_dt)
        self.rows = []
        self.errors = []
        self.status = "complete"
        self.bytes_used = 0
        self.origin_step = None

    def append(self, env, env_idx, step, action_count):
        if self.status != "complete":
            return
        try:
            if type(step) is not int or type(action_count) is not int or action_count < 0:
                raise ValueError("invalid clock")
            if self.rows:
                previous = self.rows[-1]
                if step == previous["physics_step"] and action_count == previous["action_count"]:
                    return
                if step <= previous["physics_step"] or action_count < previous["action_count"]:
                    raise ValueError("non-monotonic clock or reset without new recorder")
            if self.origin_step is None:
                self.origin_step = step
            row = {
                "physics_step": step,
                "action_count": action_count,
                "time_seconds": (step - self.origin_step) * self.dt,
                **snapshot(env, env_idx),
            }
            encoded = json.dumps(row, allow_nan=False, separators=(",", ":"))
            if len(self.rows) >= self.MAX_SAMPLES or self.bytes_used + len(encoded.encode()) > self.MAX_BYTES:
                self.status = "truncated"
                return
            self.rows.append(json.loads(encoded))
            self.bytes_used += len(encoded.encode())
            contact = row.get("contact_reports")
            if contact is not None and contact["status"] == "error":
                raise ValueError(contact["error"])
        except Exception as exc:
            # A diagnostic failure must not alter the native policy/scorer path.
            self.status = "error"
            self.errors.append(
                {"step": step, "action_count": action_count, "type": type(exc).__name__, "message": str(exc)[:512]}
            )

    def save(self, path, identity, *, success, action_count, step_limit):
        path = Path(path)
        payload = {
            "schema_version": "charger-diagnostics-v1",
            "identity": identity,
            "status": self.status,
            "errors": self.errors,
            "physics_dt": self.dt,
            "origin_step": self.origin_step,
            "rows": self.rows,
            "terminal": {
                "native_success": bool(success),
                "terminated": bool(success),
                "truncated": not success and action_count >= step_limit,
                "reason": "success" if success else "time_limit" if action_count >= step_limit else "incomplete",
                "action_count": action_count,
                "step_limit": step_limit,
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".partial")
        with temporary.open("x") as stream:
            json.dump(payload, stream, allow_nan=False, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        # Atomic publication without replacing old evidence (including a symlink).
        os.link(temporary, path)
        temporary.unlink()
