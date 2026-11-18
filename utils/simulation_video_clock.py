"""Integer physics-clock schedule; observation reads never advance video time."""
import math


class SimulationVideoClock:
    def __init__(self, physics_dt, fps, start_frame):
        if not math.isfinite(physics_dt) or not math.isfinite(fps) or physics_dt <= 0 or fps <= 0:
            raise ValueError("Video dt and fps must be positive finite numbers")
        ratio = 1 / (physics_dt * fps)
        self.stride = round(ratio)
        if self.stride < 1 or not math.isclose(ratio, self.stride, abs_tol=1e-8):
            raise ValueError("Video cadence must divide the native physics clock")
        self.start_frame = int(start_frame)
        self.next_frame = int(start_frame)
        self.last_frame = None
        self.frames = 0
        self.physics_dt = float(physics_dt)
        self.fps = float(fps)

    def due(self, frame):
        if frame < self.next_frame:
            return False
        if frame > self.next_frame:
            raise RuntimeError("Recording missed a physics-clock sample; refusing synthetic frames")
        return True

    def commit(self, frame):
        if not self.due(frame):
            raise RuntimeError("Duplicate or out-of-order video sample")
        self.last_frame = int(frame)
        self.frames += 1
        self.next_frame += self.stride

    def payload(self):
        return {"physics_start_frame": self.start_frame, "physics_last_frame": self.last_frame,
                "physics_dt_s": self.physics_dt, "fps": self.fps,
                "physics_stride": self.stride, "frames": self.frames,
                "clock": "simulation", "observation_reads_create_frames": False}
