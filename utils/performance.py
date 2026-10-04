"""Opt-in aggregate wall timers; never synchronize CUDA or retain observations."""
import atexit
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
import json
import os
from pathlib import Path
import time


class WallProfile:
    def __init__(self, enabled=False, clock=time.perf_counter):
        self.enabled = enabled
        self.clock = clock
        self.started = self.clock()
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.phase = "startup"
        self.phase_started = self.started
        self.phases = {}
        self.operations = {}
        self.stack = []
        self.metadata = {}
        self.finished = False

    def set_phase(self, name):
        if not self.enabled:
            return
        if self.stack:
            raise RuntimeError("Cannot change profiling phase within an operation")
        now = self.clock()
        self.phases[self.phase] = self.phases.get(self.phase, 0.0) + now - self.phase_started
        self.phase, self.phase_started = name, now

    @contextmanager
    def measure(self, name):
        if not self.enabled:
            yield
            return
        entry = [self.clock(), 0.0]
        self.stack.append(entry)
        try:
            yield
        finally:
            elapsed = self.clock() - entry[0]
            self.stack.pop()
            if self.stack:
                self.stack[-1][1] += elapsed
            record = self.operations.setdefault(self.phase + "/" + name, {
                "count": 0, "inclusive_s": 0.0, "exclusive_s": 0.0, "max_s": 0.0,
            })
            record["count"] += 1
            record["inclusive_s"] += elapsed
            record["exclusive_s"] += elapsed - entry[1]
            record["max_s"] = max(record["max_s"], elapsed)

    def snapshot(self):
        now = self.clock()
        phases = dict(self.phases)
        phases[self.phase] = phases.get(self.phase, 0.0) + now - self.phase_started
        return {
            "schema_version": "robodojo-wall-profile-v1",
            "clock": "perf_counter; CPU wall intervals; no CUDA synchronization added",
            "started_at": self.started_at, "elapsed_s": now - self.started,
            "finished": self.finished, "metadata": self.metadata,
            "phases_s": phases, "operations": self.operations,
        }

    def write(self):
        if not self.enabled:
            return
        run_id = os.environ.get("ROBODOJO_RUN_ID", "")
        if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in run_id):
            return
        root = Path(os.environ.get("ROBODOJO_OUTPUT_ROOT", "eval_result"))
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"performance-{run_id}-{os.getpid()}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.snapshot(), indent=2, allow_nan=False) + "\n")
        temporary.replace(path)


PROFILE = WallProfile(enabled=os.environ.get("ROBODOJO_PROFILE") == "1")
atexit.register(PROFILE.write)


def close_profiled_app(app):
    """Kit shutdown may terminate Python without returning or running atexit."""
    PROFILE.set_phase("shutdown")
    PROFILE.finished = True  # evaluation finished, not necessarily app shutdown
    PROFILE.metadata["shutdown_complete"] = False
    PROFILE.write()
    app.close()
    PROFILE.metadata["shutdown_complete"] = True
    PROFILE.write()


def profiled(name):
    def decorate(function):
        @wraps(function)
        def measured(*args, **kwargs):
            with PROFILE.measure(name):
                return function(*args, **kwargs)
        return measured
    return decorate
