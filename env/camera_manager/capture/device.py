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
