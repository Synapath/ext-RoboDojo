"""Device normalization shared by native camera capture paths."""


def explicit_cuda_device(device) -> str:
    value = str(device)
    if value == "cuda":
        raise ValueError("Camera capture requires an explicit CUDA index, for example cuda:7")
    if not value.startswith("cuda:"):
        raise ValueError(f"Camera capture requires CUDA, got {value!r}")
    index = value.removeprefix("cuda:")
    if not index.isdigit():
        raise ValueError(f"Invalid CUDA device {value!r}")
    return f"cuda:{int(index)}"


def capture_cuda_device(simulation_device) -> str:
    """Keep native CPU/USD readback independent of CUDA camera capture.

    CPU simulation tensors do not imply CPU rendering. The native CPU-readback
    path uses the default CUDA capture device; explicit CUDA simulation devices
    continue to route all capture allocations to that device.
    """
    if str(simulation_device) == "cpu":
        return "cuda:0"
    return explicit_cuda_device(simulation_device)
