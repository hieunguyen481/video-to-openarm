from __future__ import annotations

from pathlib import Path

import numpy as np

from openarm_retarget.bimanual import side_pose
from openarm_retarget.config import load_config
from openarm_retarget.hand_tracking import _map_hands_to_robot_sides
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


def test_egocentric_camera_config_cross_maps_robot_sides():
    root = Path(__file__).resolve().parents[1]
    tracking = load_config(root / "configs" / "hand_tracking.yaml")
    retarget = load_config(root / "configs" / "bimanual_retarget.yaml")

    assert tracking["mirror_input"] is False
    assert tracking["swap_left_right"] is False
    assert retarget["left"]["openarm_origin"][1] > 0
    assert retarget["right"]["openarm_origin"][1] < 0
    assert retarget["left"]["axis_mapping"]["human_x"] == "y"
    assert retarget["right"]["axis_mapping"]["human_x"] == "y"


def test_opposing_camera_cross_maps_hands_to_robot_sides():
    physical_left = object()
    physical_right = object()

    mapped = _map_hands_to_robot_sides(
        {"left": physical_left, "right": physical_right},
        swap_left_right=True,
    )

    assert mapped["left"] is physical_right
    assert mapped["right"] is physical_left
