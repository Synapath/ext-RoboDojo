import numpy as np
import torch

from utils.transformer import quat_to_mat


def test_quat_to_mat_accepts_device_tensor():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)

    matrix = quat_to_mat(quaternion)

    np.testing.assert_array_equal(matrix, np.eye(3))
