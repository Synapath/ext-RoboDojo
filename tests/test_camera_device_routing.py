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


@pytest.mark.parametrize(("value", "expected"), [("cpu", "cuda:0"), ("cuda:0", "cuda:0"), ("cuda:7", "cuda:7")])
def test_native_simulation_and_capture_devices_are_independent(value, expected):
    from env.camera_manager.capture.device import capture_cuda_device

    assert capture_cuda_device(value) == expected


@pytest.mark.parametrize("value", ["cuda", "cpu:0", "cuda:x", "cuda:-1"])
def test_capture_device_rejects_invalid_simulation_devices(value):
    from env.camera_manager.capture.device import capture_cuda_device

    with pytest.raises(ValueError):
        capture_cuda_device(value)


def test_tiled_constructor_preserves_cpu_readback_support():
    # Execute the actual constructor without importing Isaac or allocating CUDA.
    import ast
    from types import SimpleNamespace
    from env.camera_manager.capture.device import capture_cuda_device

    path = Path(__file__).parents[1] / "env/camera_manager/capture/tiled_capture_manager.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TiledCaptureManager")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    init.returns = None
    for arg in init.args.args:
        arg.annotation = None
    # Annotated assignments inside functions are not evaluated at runtime.
    module = ast.fix_missing_locations(ast.Module(body=[init], type_ignores=[]))
    namespace = {"capture_cuda_device": capture_cuda_device}
    exec(compile(module, str(path), "exec"), namespace)
    camera_manager = SimpleNamespace(cameras=[], camera_names=[])
    for device, expected in (("cpu", "cuda:0"), ("cuda:7", "cuda:7")):
        obj = SimpleNamespace()
        namespace["__init__"](obj, 1, {}, camera_manager, device)
        assert obj.device == expected
