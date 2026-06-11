from __future__ import annotations

import numpy as np

from openarm_retarget.dataset import build_bimanual_dataset


def test_bimanual_dataset_contains_both_arms_and_grippers():
    timestamps = np.array([0.0, 0.1, 0.2])
    smooth = {
        "timestamps": timestamps,
        "left_wrist_smooth": np.zeros((3, 3)),
        "right_wrist_smooth": np.ones((3, 3)),
        "left_gripper_cmd": np.array([0, 1, 1]),
        "right_gripper_cmd": np.array([1, 0, 0]),
    }
    target = {
        "left_target_pos": np.zeros((3, 3)),
        "right_target_pos": np.ones((3, 3)),
        "left_gripper_cmd": np.array([0, 1, 1]),
        "right_gripper_cmd": np.array([1, 0, 0]),
    }
    trajectory = {
        "qpos": np.zeros((3, 19)),
        "left_arm_qpos": np.zeros((3, 7)),
        "right_arm_qpos": np.ones((3, 7)),
        "left_ee_pos": np.zeros((3, 3)),
        "right_ee_pos": np.ones((3, 3)),
        "left_gripper_cmd": np.array([0, 1, 1]),
        "right_gripper_cmd": np.array([1, 0, 0]),
    }

    dataset = build_bimanual_dataset(smooth, target, trajectory)

    assert dataset["action_left_arm_joint_target"].shape == (3, 7)
    assert dataset["action_right_arm_joint_target"].shape == (3, 7)
    assert dataset["action_left_gripper_cmd"].shape == (3, 1)
    assert dataset["action_right_gripper_cmd"].shape == (3, 1)
