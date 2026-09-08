import numpy as np
from utils.charger_oracle import ChargerOracle, target_end_link, pose_matrix, choose_target


def test_nearest_position_must_not_select_opposite_insertion_frame():
    reverse = dict(insert_frame=0, lateral_m=0.0009, axial_m=0.0145, rotation_error_deg=178.5)
    aligned = dict(insert_frame=1, lateral_m=0.0011, axial_m=0.0145, rotation_error_deg=1.5)
    assert choose_target([reverse, aligned]) is aligned
    assert choose_target([reverse]) is None


def test_measured_grasp_offset_is_preserved_by_target_transform():
    obj = np.array([0.2, 0.1, 0.8, 1, 0, 0, 0])
    ee = np.array([0.25, 0.1, 0.85, 1, 0, 0, 0])
    insert = pose_matrix(obj)
    target = insert.copy()
    target[0, 3] += 0.02
    actual = target_end_link(obj, ee, insert, target, 0, 0.01, 6)
    np.testing.assert_allclose(actual[:3], ee[:3] + [0.01, 0, 0], atol=1e-10)
    np.testing.assert_allclose(actual[3:], ee[3:], atol=1e-10)


def test_axial_offset_follows_target_frame_not_world_z():
    obj = ee = np.array([0, 0, 0, 1, 0, 0, 0])
    insert = pose_matrix(obj)
    target = insert.copy()
    target[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    actual = target_end_link(obj, ee, insert, target, 0.005, 0.01, 6)
    np.testing.assert_allclose(actual[:3], [0.005, 0, 0], atol=1e-10)


def test_settled_correction_returns_control_and_does_not_restart():
    oracle = ChargerOracle(None, dict(strategy="immediate", yield_when_settled=True))
    oracle.started, oracle.holder, oracle.settled_chunks = 100, "left", 1
    target = dict(insert_frame=1, support="slot", support_frame=0,
                  lateral_m=0.0008, axial_m=-0.0082, rotation_error_deg=0.5)
    frame = dict(step=140, diagnostics={"annotated_alignment": {"targets": [target]},
                                       "native_instantaneous": {"inside": True, "upright": True}})
    action, metadata = oracle.propose(frame, {"holding_proxy": "L"}, None)
    assert action is None and metadata["reason"] == "correction_complete"
    assert not metadata["bc_eligible"]
    # Once handed back, later loss of the holding proxy does not restart Oracle.
    action, metadata = oracle.propose({"step": 150}, {}, None)
    assert action is None and metadata["reason"] == "correction_complete"
