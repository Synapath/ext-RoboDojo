"""Explicit engineering-only reset-layout interventions; never changes source assets."""
from copy import deepcopy
import hashlib
import json
import math


def shifted_socket_layout(layout, raw_xy, *, task, checkpoint, diagnostics):
    if task != "plug_in_charger" or not diagnostics or not checkpoint.startswith("engineering-"):
        raise ValueError("socket intervention requires a charger diagnostic engineering replay")
    xy = json.loads(raw_xy)
    if not isinstance(xy, list) or len(xy) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) for v in xy):
        raise ValueError("socket XY must be two finite numbers in meters")
    if math.hypot(*xy) > 0.025:
        raise ValueError("socket XY exceeds 25mm bound")
    sockets = layout.get("Rigid", {}).get("socket", [])
    if len(sockets) != 1:
        raise ValueError("requires exactly one rigid socket")
    pos = sockets[0].get("default_pos")
    if not isinstance(pos, (list, tuple)) or len(pos) != 3 or not all(math.isfinite(v) for v in pos):
        raise ValueError("invalid saved socket position")
    candidate = deepcopy(layout)
    candidate["Rigid"]["socket"][0]["default_pos"] = [pos[0] + xy[0], pos[1] + xy[1], pos[2]]
    digest = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return candidate, {"schema": "charger-socket-xy-v1", "source_layout_sha256": digest(layout),
                       "candidate_layout_sha256": digest(candidate), "world_xy_m": xy,
                       "before_position_m": list(pos), "after_position_m": candidate["Rigid"]["socket"][0]["default_pos"],
                       "applied_at": "before_scene_reset", "engineering_only": True}
