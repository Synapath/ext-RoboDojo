from env.camera_manager.capture.warmup import EmptyAnnotatorDataError, get_data_with_warmup_retry


class _WarmingCamera:
    calls = 0

    def get_data(self, annotator_name, out):
        self.calls += 1
        if self.calls < 3:
            raise EmptyAnnotatorDataError("warming up")
        return "frames", {"ready": True}


class _Sim:
    renders = 0

    def render(self):
        self.renders += 1


def test_step_renders_and_retries_empty_annotator_data():
    camera = _WarmingCamera()
    sim = _Sim()

    data, info = get_data_with_warmup_retry(camera, "rgb", None, sim.render)

    assert sim.renders == 2
    assert data == "frames"
    assert info == {"ready": True}
