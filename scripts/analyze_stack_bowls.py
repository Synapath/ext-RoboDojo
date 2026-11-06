"""Recompute stable-stack-v2 from geometry JSONL, without Isaac Sim or a policy."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.stack_bowls_metric import StableStack, collision_boxes, geometry


def analyze(path, meshes, *, step_limit=800):
    metric = StableStack(meshes)
    native_first = None
    with Path(path).open() as f:
        metadata = json.loads(next(f))
        if metadata.get("type") != "metadata":
            raise ValueError("geometry metadata required; ordinary robot telemetry is insufficient")
        frames = 0
        for line in f:
            frame = json.loads(line)
            if not frames and frame["step"] != metadata.get("initial_step", 0):
                raise ValueError("missing initial frame")
            metric.update(frame)
            native = frame.get("native_success")
            if native is None:
                native = geometry(frame) and frame["predicates"]["arms_return"]
            if native and native_first is None:
                native_first = frame["step"]
            frames += 1
    if not frames:
        raise ValueError("empty geometry trace")
    limit = int(metadata.get("step_limit", step_limit))
    if frame["step"] > limit or metadata.get("control_hz", 25) != 25:
        raise ValueError("unsupported time contract")
    stable = metric.first_pass
    censored = stable is None and frame["step"] < limit
    return dict(
        metric_version="stable-stack-v2",
        seed=metadata["seed"],
        layout=metadata["layout"],
        frames=frames,
        native_first_step=native_first,
        stable_first_step=stable,
        stable_success=True if stable is not None else (None if censored else False),
        stable_censored=censored,
        final_step=frame["step"],
        step_limit=limit,
        trace=str(path),
        trace_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, nargs="+", help="Geometry JSONL traces, including archived X1 traces")
    parser.add_argument("--urdf", type=Path, required=True, help="Matching X5A.urdf with local collision meshes")
    parser.add_argument(
        "--output", type=Path, required=True, help="New JSON report; existing files are not overwritten"
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    meshes, assets = collision_boxes(args.urdf)
    rows = [analyze(p, meshes) for p in args.traces]
    result = dict(
        schema="stack-bowls-analysis-v1",
        asset_sha256=assets,
        episodes=rows,
        stable_success_count=sum(x["stable_success"] is True for x in rows),
        stable_censored_count=sum(x["stable_censored"] for x in rows),
    )
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
