"""CPU failure paths; real image/state and MP4 acceptance lives in run evidence."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_strict_delivery_never_falls_back_to_render_count(monkeypatch):
    sync = load("sync", "env/camera_manager/capture/render_sync.py")
    monkeypatch.setattr(sync, "zero_delay_active", lambda: True)
    monkeypatch.setattr(sync, "dispatcher_time", lambda: None)
    class Env:
        renders = 0
        def render(self):
            self.renders += 1
    env = Env()
    with pytest.raises(RuntimeError, match="no fresh frame"):
        sync.wait_for_latest_cameras(env, max_passes=3, require_delivery=True)
    assert env.renders == 3


def test_stalled_dispatcher_rejected(monkeypatch):
    sync = load("sync", "env/camera_manager/capture/render_sync.py")
    monkeypatch.setattr(sync, "zero_delay_active", lambda: True)
    monkeypatch.setattr(sync, "dispatcher_time", lambda: 10.0)
    with pytest.raises(RuntimeError, match="no fresh frame"):
        sync.wait_for_latest_cameras(type("Env", (), {"render": lambda self: None})(),
                                     max_passes=3, require_delivery=True)


def test_strict_mode_requires_launch_settings(monkeypatch):
    sync = load("sync", "env/camera_manager/capture/render_sync.py")
    monkeypatch.setattr(sync, "zero_delay_active", lambda: False)
    with pytest.raises(RuntimeError, match="zero-delay"):
        sync.wait_for_latest_cameras(None, require_delivery=True)


def test_dispatcher_can_initialize_on_first_delivered_frame(monkeypatch):
    sync = load("sync", "env/camera_manager/capture/render_sync.py")
    monkeypatch.setattr(sync, "zero_delay_active", lambda: True)
    times = iter([None, None, 0.04])
    monkeypatch.setattr(sync, "dispatcher_time", lambda: next(times))
    env = type("Env", (), {"render": lambda self: None})()
    assert sync.wait_for_latest_cameras(env, max_passes=3, require_delivery=True) == 2


def test_simulation_schedule_and_zero_physics_waits():
    clock = load("clock", "utils/simulation_video_clock.py").SimulationVideoClock(.004, 25, 842)
    samples = []
    for frame in range(842, 1043):
        if clock.due(frame):
            samples.append(frame)
            clock.commit(frame)
        # Arbitrarily many model/snapshot reads at fixed physics time do not add frames.
        assert not clock.due(frame)
    assert samples == list(range(842, 1043, 10))
    assert clock.frames == 21 and clock.payload()["physics_last_frame"] == 1042


def test_schedule_refuses_missing_samples_and_duplicates():
    cls = load("clock", "utils/simulation_video_clock.py").SimulationVideoClock
    clock = cls(.004, 25, 10)
    clock.commit(10)
    with pytest.raises(RuntimeError, match="Duplicate"):
        clock.commit(10)
    with pytest.raises(RuntimeError, match="missed"):
        clock.due(21)


@pytest.mark.parametrize("dt,fps", [(0, 25), (.004, 0), (.004, 30), (float("nan"), 25)])
def test_invalid_sampling_clock(dt, fps):
    cls = load("clock", "utils/simulation_video_clock.py").SimulationVideoClock
    with pytest.raises(ValueError):
        cls(dt, fps, 0)


def test_stale_shared_annotator_does_not_skip_other_releases():
    release = load("cleanup", "utils/camera_cleanup.py").release_render_product
    events = []
    class Annotator:
        def __init__(self, name, error=None): self.name, self.error = name, error
        def detach(self, paths):
            events.append((self.name, paths))
            if self.error: raise self.error
    product = type("Product", (), {"destroy": lambda self: events.append("product")})()
    warnings = release(product, {"rgb": Annotator("rgb", TypeError("Invalid NodeObj object")),
                                 "depth": Annotator("depth")}, "/product")
    assert events == [("rgb", ["/product"]), ("depth", ["/product"]), "product"]
    assert len(warnings) == 1 and warnings[0]["annotator"] == "rgb"


def test_unexpected_release_error_is_not_hidden():
    release = load("cleanup", "utils/camera_cleanup.py").release_render_product
    events = []
    class Annotator:
        def detach(self, paths): raise ValueError("unexpected SDK failure")
    product = type("Product", (), {"destroy": lambda self: events.append("product")})()
    with pytest.raises(RuntimeError, match="all cleanup"):
        release(product, {"rgb": Annotator()}, "/product")
    assert events == ["product"]
