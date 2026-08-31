"""CPU-only, bounded sidecar recording; never owns simulator or tensor references."""

import json
import math
import os
from pathlib import Path

import numpy as np


class EpisodeTelemetry:
    MAX_SAMPLES = 100_000
    MAX_BYTES = 64 * 1024 * 1024

    def __init__(self, dt, names=None):
        self.dt = float(dt) if dt and math.isfinite(dt) and dt > 0 else None
        self.names = names or {}
        self.rows = []
        self.bytes_used = 0
        self.truncated = False
        self.origin_step = None

    def append(self, step, state, targets, actions, frames):
        if self.truncated:
            return
        if self.origin_step is None:
            self.origin_step = int(step)
        signals = {}
        for kind, source in (("state", state), ("target", targets), ("action", actions)):
            for key, value in source.items():
                # ObsManager's gripper state is prev_control, NOT a measured position.
                if kind == "state" and "ee_joint_state" in key:
                    continue
                if not (key.endswith("joint_state") or key.endswith("ee_pose")):
                    continue
                if isinstance(value, dict):
                    value = value.get("position")
                arr = np.asarray(value, dtype=float)
                if arr.ndim != 1 or not 1 <= len(arr) <= 64:
                    continue
                signals[f"{kind}/{key}"] = [float(v) if math.isfinite(v) else None for v in arr]
        estimate = 512 + sum(256 + 64 * len(v) for v in signals.values())
        if len(self.rows) >= self.MAX_SAMPLES or self.bytes_used + estimate > self.MAX_BYTES:
            self.truncated = True
            return
        self.bytes_used += estimate
        self.rows.append((int(step) - self.origin_step, signals, dict(frames)))

    def payload(self, identity, videos):
        steps = [row[0] for row in self.rows]
        times = [s * self.dt for s in steps] if self.dt else None
        keys = sorted(
            {key for _, values, _ in self.rows for key in values},
            key=lambda key: ({"state": 0, "target": 1, "action": 2}[key.split("/", 1)[0]],
                             0 if key.endswith("arm_joint_state") else 1, key),
        )
        series = []
        for key in keys:
            kind, native = key.split("/", 1)
            width = len(next(values[key] for _, values, _ in self.rows if key in values))
            channels = self.names.get(native)
            if not channels or len(channels) != width:
                channels = [f"channel_{i}" for i in range(width)]
            unit = "rad" if native.endswith("arm_joint_state") else None
            series.append({
                "key": key, "label": f"{native} · {kind}", "kind": kind,
                "unit": unit, "channels": channels, "steps": steps, "times_seconds": times,
                "values": [values.get(key, [None] * width) for _, values, _ in self.rows],
            })
        mappings = [{"path": info["path"], "fps": info["fps"],
                     "frames": [frames.get(camera) for _, _, frames in self.rows]}
                    for camera, info in videos.items()]
        return {"schema_version": "robodojo-telemetry-v1", "identity": identity,
                "status": "truncated" if self.truncated else "complete", "series": series,
                "videos": mappings, "clock": "physics steps relative to first recorded observation",
                "origin_step": self.origin_step}

    def save(self, path, identity, videos):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".partial")
        # One write after the episode, never synchronous per-step disk I/O.
        with temporary.open("x") as stream:
            json.dump(self.payload(identity, videos), stream, allow_nan=False, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
