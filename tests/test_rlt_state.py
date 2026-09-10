"""State plumbing tests; real PhysX replay remains a separate native verifier."""

from types import SimpleNamespace as NS

import numpy as np
import torch

from utils import rlt_state
from utils.charger_diagnostics import ContactWindow


class Scene:
    def __init__(self):
        self.device = "cpu"
        self.position = torch.tensor([[0.25]])
        data = NS(joint_pos_target=torch.tensor([[0.7]]), joint_vel_target=torch.tensor([[0.1]]),
                  joint_effort_target=torch.tensor([[0.2]]))
        self.articulations = {"arm": NS(data=data)}
        robot = self.articulations["arm"]
        for method, field in (("position", "pos"), ("velocity", "vel"), ("effort", "effort")):
            setattr(robot, "set_joint_" + method + "_target",
                    lambda x, key="joint_" + field + "_target": setattr(data, key, x))

    def get_state(self):
        return {"joint": self.position.clone()}

    def reset(self):
        pass

    def reset_to(self, state):
        self.position = state["joint"].clone()
        # IsaacLab reset_to changes targets to current q; our restore must undo it.
        self.articulations["arm"].data.joint_pos_target = self.position.clone()

    def write_data_to_sim(self):
        pass

    def update(self, dt):
        assert dt == 0


def test_restore_preserves_command_targets_native_home_and_remaining_time(monkeypatch):
    monkeypatch.setenv("ROBODOJO_RLT_INTERACTION", "1")
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda _: None)
    scene = Scene()
    renders = []
    context = NS(render_mode=1, render=lambda: renders.append("camera"),
                 physics_sim_view=NS(update_articulations_kinematic=lambda: renders.append("fk")))
    parser = NS(pre_state=[{"charger": 2}], robot_origin_endpose=[{"home": 9}],
                joint_ratio_transition_state=[{}])
    reward = NS(env=None, func_parser=parser, check_list=[["native-four"]])
    controls = NS(control_queue=[NS(is_empty=lambda: True)], prev_control=[{"position": 0.7}])
    manager = NS(control_manager=controls, robot_origin_endpose=[{"home": 9}],
                 robot_init_joint=[{}], robot_list=[NS(base_link_origin_pose=np.array([1., 2.]))])
    env = NS(task_name="plug_in_charger", num_envs=1, charger_diagnostics_enabled=True,
             use_fabric=False, physics_interface=NS(update_transformations=lambda *args: renders.append(args)),
             env_seeds=[13], eval_seed=0, step_lim=400, success=[True], end_flag=[False],
             take_action_cnt=[160], rlt_applied_actions={0: {}}, telemetry_actions={},
             sim=NS(scene=scene, sim=context, device="cpu", _sim_step_counter=960, common_step_counter=320,
                    episode_length_buf=torch.tensor([320])),
             robot_manager=manager, reward_manager=reward, obs_manager=NS(instruction=["insert"]),
             scene_manager=NS(_rigid_and_dynamic_objects=[{}], _articulation_objects=[{}],
                              _garment_objects=[{}], _fluid_objects=[{}]),
             charger_diagnostics={0: NS(rows=[{"action_count": 160}])},
             charger_contact_observer=NS(windows={0: ContactWindow()}), telemetry={}, rlt_observations={0: "stale image"},
             _abort_video_writers=lambda: None, render=lambda: None)
    gate = NS(rows=list(range(161)), ee={"left": list(range(161)), "right": list(range(161))})
    snapshot = rlt_state.capture(env, gate)
    env.take_action_cnt = [400]
    env.end_flag = [True]
    reward.check_list = [[]]
    parser.robot_origin_endpose = [{"home": -1}]
    scene.position.fill_(10)
    receipt = rlt_state.restore(env, snapshot, gate)
    assert receipt["step"] == 160 and receipt["max_abs"] == 0
    assert env.step_lim - env.take_action_cnt[0] == 240
    assert parser.robot_origin_endpose == [{"home": 9}]
    assert reward.check_list == [["native-four"]]
    assert len(gate.rows) == 160 and len(snapshot["gate"]["rows"]) == 161
    assert env.rlt_observations == {} and env.end_flag == [False]
    assert env.charger_contact_observer.windows[0].drain()["status"] == "reported"
    assert renders == ["fk", (False, True, True, False), "camera", "camera", "camera"]
    np.testing.assert_allclose(scene.articulations["arm"].data.joint_pos_target, [[0.7]])
