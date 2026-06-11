from __future__ import annotations

import numpy as np

from openarm_retarget.dataset import build_dataset


def test_dataset_builds_qvel_and_actions():
    timestamps = np.array([0.0, 0.1, 0.2])
    smooth = {
        "timestamps": timestamps,
        "wrist_smooth": np.zeros((3, 3)),
        "gripper_cmd": np.array([0, 1, 1]),
    }
    target = {
        "target_pos": np.ones((3, 3)),
        "gripper_cmd": np.array([0, 1, 1]),
    }
    trajectory = {
        "qpos": np.arange(12, dtype=float).reshape(3, 4),
        "arm_qpos": np.arange(6, dtype=float).reshape(3, 2),
        "ee_pos": np.ones((3, 3)),
        "gripper_cmd": np.array([0, 1, 1]),
    }

    dataset = build_dataset(smooth, target, trajectory)

    assert dataset["obs_qvel"].shape == (3, 4)
    assert dataset["action_gripper_cmd"].shape == (3, 1)
    assert np.all(np.isfinite(dataset["obs_qvel"]))

