from __future__ import annotations

import numpy as np

from openarm_retarget.bimanual import side_pose
from openarm_retarget.pinch import detect_pinch
from openarm_retarget.synthetic import generate_bimanual_hand_pose


def test_bimanual_synthetic_has_independent_hands_and_pinch():
    pose = generate_bimanual_hand_pose(frames=100, noise_std=0)
    left = side_pose(pose, "left")
    right = side_pose(pose, "right")
    left_pinch = detect_pinch(
        left, close_threshold=0.03, open_threshold=0.06
    )
    right_pinch = detect_pinch(
        right, close_threshold=0.03, open_threshold=0.06
    )

    assert left["wrist"].shape == (100, 3)
    assert right["wrist"].shape == (100, 3)
    assert left["palm_scale"].shape == (100,)
    assert np.nanmax(left["palm_scale"]) - np.nanmin(left["palm_scale"]) > 0
    assert not np.array_equal(
        left_pinch["gripper_cmd"], right_pinch["gripper_cmd"]
    )
    assert np.count_nonzero(np.diff(left_pinch["gripper_cmd"])) == 4
    assert np.count_nonzero(np.diff(right_pinch["gripper_cmd"])) >= 3
