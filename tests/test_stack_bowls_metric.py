import itertools

import numpy as np
import pytest

from utils.stack_bowls_metric import StableStack, boxes_separated, geometry

BOX = np.array(list(itertools.product([-0.02, 0.02], repeat=3)))


def frame(step=0):
    bowls = {}
    for j in range(3):
        p = np.array([0.0, 0.0, 0.02 + j * 0.03])
        bowls[f"bowl{j}"] = dict(pos=p, quat=[1.0, 0.0, 0.0, 0.0], bbox_world=BOX + p, com_world=p)
    arms = {
        a: dict(
            gripper_open_norm=1.0,
            ee=[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            links={"link": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]},
        )
        for a in ("left_arm", "right_arm")
    }
    return dict(step=step, bowls=bowls, arms=arms, predicates={"arms_return": False})


def test_released_stable_accepts_without_home_and_needs_one_second():
    metric = StableStack({"link": BOX})
    for t in range(25):
        assert metric.update(frame(t))["first_pass_step"] is None
    assert metric.update(frame(25))["first_pass_step"] == 25


@pytest.mark.parametrize("kind", ["closed", "support", "fall", "drift", "gap"])
def test_rejects_false_success(kind):
    metric = StableStack({"link": BOX})
    for t in range(26):
        f = frame(t)
        if kind == "closed":
            f["arms"]["left_arm"]["gripper_open_norm"] = 0.7
        if kind == "support":
            f["arms"]["left_arm"]["links"]["link"][0] = 0
        if kind == "fall" and t == 25:
            f["bowls"]["bowl2"]["pos"][0] = 0.05
        if kind == "drift":
            for b in f["bowls"].values():
                b["pos"][0] += t * 0.0003
        if kind == "gap":
            f["bowls"]["bowl2"]["pos"][2] = 0.052
        assert metric.update(f)["first_pass_step"] is None


def test_missing_frame_and_bad_quaternion_rejected():
    metric = StableStack({"link": BOX})
    metric.update(frame(0))
    with pytest.raises(ValueError):
        metric.update(frame(2))
    f = frame()
    f["bowls"]["bowl1"]["quat"] = [0.0, 0.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        geometry(f)


def test_quaternion_sign_is_same_orientation():
    metric = StableStack({"link": BOX})
    for t in range(26):
        f = frame(t)
        f["bowls"]["bowl1"]["quat"][0] = -1 if t % 2 else 1
        out = metric.update(f)
    assert out["first_pass_step"] == 25


def test_rotated_boxes_certify_clearance_without_world_axis_gap():
    # Parallel long, thin boxes: their world AABBs overlap, their surfaces do not.
    box = np.array(list(itertools.product([-0.1, 0.1], [-0.01, 0.01], [-0.01, 0.01])))
    angle = np.pi / 4
    R = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    a = box @ R.T
    b = a + np.array([-np.sin(angle), np.cos(angle), 0]) * 0.04
    assert not np.any(np.maximum(a.min(0) - b.max(0), b.min(0) - a.max(0)) >= 0.01)
    assert boxes_separated(a, b)
    assert not boxes_separated(a, a + 0.001)
    rounded = b.copy()
    rounded[:, 0] += rounded[:, 1] * 0.0006
    assert boxes_separated(a, rounded)
