import numpy as np

from utils.tubes_oracle import choose_handoff, released_clear


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
