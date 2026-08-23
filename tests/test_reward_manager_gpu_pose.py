import numpy as np
import torch

from env.reward_manager.func_parser import Func_Parser


class _LayoutManager:
    def get_layout_records(self, env_idx, object_type):
        if object_type == "Rigid":
            return [{"inst_name": "object"}]
        return []

    def get_instance_pose(self, inst_name, env_idx):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return (
            torch.tensor([1.0, 2.0, 3.0], device=device),
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=device),
        )


class _RobotManager:
    robot_list = []


class _Env:
    success = [True]


def test_init_state_accepts_device_tensors():
    parser = Func_Parser(num_envs=1)
    parser.env = _Env()
    parser.layout_manager = _LayoutManager()
    parser.robot_manager = _RobotManager()

    parser.init_state()

    pose = parser.pre_state[0]["object"]["pose"]
    np.testing.assert_array_equal(pose, np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]))
    assert isinstance(pose, np.ndarray)
