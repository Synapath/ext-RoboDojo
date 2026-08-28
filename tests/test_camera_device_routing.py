from pathlib import Path

import pytest

from env.camera_manager.capture.device import explicit_cuda_device


@pytest.mark.parametrize(("value", "expected"), [("cuda:0", "cuda:0"), ("cuda:7", "cuda:7"), ("cuda:07", "cuda:7")])
def test_explicit_cuda_device(value, expected):
    assert explicit_cuda_device(value) == expected


@pytest.mark.parametrize("value", ["cuda", "cpu", "cuda:x", "cuda:-1"])
def test_explicit_cuda_device_rejects_ambiguous_or_invalid_values(value):
    with pytest.raises(ValueError):
        explicit_cuda_device(value)


def test_capture_sources_do_not_pin_cuda_zero():
    root = Path(__file__).parents[1] / "env/camera_manager/capture"
    for name in ("camera_view.py", "tiled_capture_manager.py"):
        text = (root / name).read_text()
        assert 'device="cuda:0"' not in text
