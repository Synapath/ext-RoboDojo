import json
from types import SimpleNamespace as NS

import numpy as np
from scipy.spatial.transform import Rotation

from utils.tubes_oracle import TubesOracle, choose_handoff, released_clear


def geometry():
    tube = dict(pose=[0, 0, 1, 0.70710678, 0.70710678, 0, 0], bbox=[[-0.01, -0.01, 1], [0.01, 0.01, 1.14]])
    return dict(
        objects={
            **{f"tube{i}": tube for i in range(3)},
            "slot": dict(
                bbox=[[-0.1, -0.1, 0.8], [0.1, 0.1, 0.86]], holes={"hole": dict(position=[0, 0, 0.86], axis=[0, 0, 1])}
            ),
        },
        hands={"left": [0, 0, 1.08, 1, 0, 0, 0], "right": [1, 0, 1.08, 1, 0, 0, 0]},
    )


def test_holding_requires_temporal_relative_pose_and_no_prior_insertion():
    g = geometry()
    q = np.zeros(14)
    q[13] = 1
    history = {}
    pred = [dict(depth=False, inside=False, upright=True)] * 3
    for _ in range(4):
        assert choose_handoff(g, q, pred, history) is None
    assert choose_handoff(g, q, pred, history)["side"] == "left"
    assert choose_handoff(g, q, [dict(depth=True, inside=True, upright=True)] * 3, history) is None


def test_release_requires_hand_above_tube_and_open_gripper():
    g = geometry()
    q = np.zeros(14)
    cfg = dict(side="left", tube_id=0, hole="hole", ee_clearance_m=0.12)
    q[6] = 1
    assert not released_clear(g, q, cfg)[0]
    g["hands"]["left"][2] = 1.30
    assert released_clear(g, q, cfg)[0]
    q[6] = 0.4
    assert not released_clear(g, q, cfg)[0]


def test_handoff_rejects_horizontal_or_far_away_tube():
    for pose in ([0, 0, 1, 1, 0, 0, 0], [0.2, 0, 1, 0.70710678, 0.70710678, 0, 0]):
        g = geometry()
        for i in range(3):
            g["objects"][f"tube{i}"]["pose"] = pose
        history = {}
        for _ in range(6):
            assert (
                choose_handoff(g, np.zeros(14), [dict(depth=False, inside=False, upright=False)] * 3, history) is None
            )


def fixture():
    robot = NS(ee_link_name="ee", robot_name="arm")
    articulation = NS(
        body_names=["base", "ee"],
        is_fixed_base=True,
        joint_names=list("abcdef"),
        root_physx_view=NS(get_jacobians=lambda: np.eye(6)[None, None]),
        data=NS(body_link_pos_w=np.zeros((1, 2, 3)), body_com_pos_w=np.zeros((1, 2, 3))),
    )
    manager = NS(
        scene=NS(env_origins=np.zeros((1, 3))),
        robot_list=[robot],
        robot_key=[articulation],
        ik_solver={"arm": NS(active_joints_name=list("abcdef"))},
        get_robot_by_arm_name=lambda _: robot,
    )
    q = np.zeros(14)
    q[13] = 0.24
    g = dict(
        objects=dict(
            tube0=dict(pose=[0, 0, 0.03, 0.70710678, 0.70710678, 0, 0]),
            slot=dict(holes={"hole": dict(position=[0, 0, 0], axis=[0, 0, 1])}),
        ),
        hands=dict(right=[0, 0, 0.1, 1, 0, 0, 0]),
    )
    config = dict(
        side="right",
        tube_id=0,
        hole="hole",
        solver="jacobian",
        integral_servo=True,
        stable_contact=True,
        translation_m=0.001,
    )
    return TubesOracle(NS(robot_manager=manager), config, q, g), q, g


def test_single_positive_frame_does_not_open_gripper_and_release_does_not_reverse():
    oracle, q, g = fixture()
    yes = [dict(depth=True, inside=True, upright=True)] * 3
    no = [dict(depth=False, inside=True, upright=True)] * 3
    oracle.propose(q, g, no)
    cmd, _ = oracle.propose(q, g, yes)
    assert cmd[13] == q[13]
    oracle.propose(q, g, no)
    assert oracle.confirm_steps == 0
    for _ in range(5):
        cmd, _ = oracle.propose(q, g, yes)
    assert oracle.phase == "release"
    assert 0 < cmd[13] - q[13] <= 0.05 + 1e-10
    previous = cmd[13]
    for _ in range(10):
        cmd, _ = oracle.propose(q, g, no)
        assert oracle.phase == "release"
        assert previous <= cmd[13] <= previous + 0.05 + 1e-10
        previous = cmd[13]


def test_contact_mode_latches_and_bounds_reversal_of_joint_commands():
    oracle, q, g = fixture()
    oracle.phase = "insert"
    g["objects"]["tube0"]["pose"][2] = -0.02
    no = [dict(depth=False, inside=True, upright=True)] * 3
    commands = [q[7:13].copy()]
    for i in range(40):
        g["objects"]["tube0"]["pose"][0] = 0.03 if i % 2 else -0.03
        if i == 20:
            g["objects"]["tube0"]["pose"][2] = 0.01
        q, detail = oracle.propose(q, g, no)
        json.dumps(detail, allow_nan=False)
        assert detail["contact_mode"]
        commands.append(q[7:13].copy())
    increments = np.diff(commands, axis=0)
    assert abs(increments).max() <= 0.004 + 1e-10
    assert abs(np.diff(increments, axis=0)).max() <= 0.001 + 1e-10


def test_open_gripper_waits_for_stable_insertion_before_lifting():
    oracle, q, g = fixture()
    oracle.phase = "release"
    oracle.release_q = q.copy()
    oracle.release_command = 1.0
    no = [dict(depth=False, inside=True, upright=True)] * 3
    yes = [dict(depth=True, inside=True, upright=True)] * 3
    for _ in range(30):
        cmd, _ = oracle.propose(q, g, no)
    assert oracle.phase == "release"
    assert cmd[13] == 1
    for _ in range(4):
        oracle.propose(q, g, yes)
        assert oracle.phase == "release"
    oracle.propose(q, g, yes)
    assert oracle.phase == "lift"


def test_contact_does_not_chase_tube_axial_spin():
    oracle, q, g = fixture()
    oracle.phase = "insert"
    pose = g["objects"]["tube0"]["pose"]
    pose[2] = -0.02
    original = Rotation.from_quat([pose[4], pose[5], pose[6], pose[3]])
    spun = original * Rotation.from_rotvec([0, 0.4, 0])
    quat = spun.as_quat()
    pose[3:] = [quat[3], *quat[:3]]
    cmd, _ = oracle.propose(q, g, [dict(depth=False, inside=True, upright=True)] * 3)
    # Identity Jacobian fixture separates translation from orientation.
    np.testing.assert_allclose(cmd[10:13], 0, atol=1e-10)


def test_guided_seating_waits_for_depth_before_full_release():
    oracle, q, g = fixture()
    oracle.phase = "insert"
    oracle.config["guided_seat_depth_m"] = 0.035
    g["objects"]["tube0"]["pose"][2] = -0.036
    partial = [dict(depth=False, inside=True, upright=True)] * 3
    for _ in range(5):
        q, _ = oracle.propose(q, g, partial)
    assert oracle.phase == "seat"
    for _ in range(60):
        q, detail = oracle.propose(q, g, partial)
        assert oracle.phase == "seat"
        assert q[13] <= 0.45
        json.dumps(detail, allow_nan=False)
    yes = [dict(depth=True, inside=True, upright=True)] * 3
    for _ in range(5):
        q, _ = oracle.propose(q, g, yes)
    assert oracle.phase == "release"
