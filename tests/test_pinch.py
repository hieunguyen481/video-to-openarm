from __future__ import annotations

import numpy as np

from openarm_retarget.pinch import detect_pinch
from openarm_retarget.synthetic import generate_hand_pose


def test_pinch_hysteresis_and_missing_frames():
    pose = generate_hand_pose(frames=100, noise_std=0)
    result = detect_pinch(
        pose,
        close_threshold=0.03,
        open_threshold=0.06,
        initial_state=0,
        invalid_policy="hold",
    )

    assert result["gripper_cmd"][10] == 0
    assert result["gripper_cmd"][30] == 1
    assert result["gripper_cmd"][50] == result["gripper_cmd"][49]
    assert result["pinch_finger"][30] == 0
    assert np.isnan(result["pinch_distance"][50])


def test_threshold_order_is_validated():
    pose = generate_hand_pose(frames=20)
    try:
        detect_pinch(pose, close_threshold=0.1, open_threshold=0.05)
    except ValueError as exc:
        assert "smaller" in str(exc)
    else:
        raise AssertionError("Expected invalid thresholds to fail")


def test_confirmation_frames_reject_single_frame_false_pinch():
    pose = generate_hand_pose(frames=30, noise_std=0)
    pose["index_tip"][:] = pose["wrist"] + np.array([0.1, 0, 0])
    pose["index_tip"][5] = pose["thumb_tip"][5]
    pose["index_tip"][10:13] = pose["thumb_tip"][10:13]

    result = detect_pinch(
        pose,
        close_threshold=0.03,
        open_threshold=0.06,
        close_confirm_frames=3,
        open_confirm_frames=2,
    )

    assert result["gripper_cmd"][5] == 0
    assert result["gripper_cmd"][11] == 0
    assert result["gripper_cmd"][12] == 1
    assert result["gripper_cmd"][13] == 1
    assert result["gripper_cmd"][14] == 0
