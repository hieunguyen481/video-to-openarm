from __future__ import annotations

import numpy as np

from openarm_retarget.baseline import prepare_supervised_data


def test_prepare_supervised_data_predicts_next_action():
    dataset = {
        "obs_qpos": np.zeros((5, 4)),
        "obs_target_pos": np.ones((5, 3)),
        "obs_gripper_state": np.zeros((5, 1)),
        "action_arm_joint_target": np.arange(10).reshape(5, 2),
        "action_gripper_cmd": np.array([[0], [0], [1], [1], [0]]),
    }
    features, arm, gripper = prepare_supervised_data(dataset)

    assert features.shape == (4, 8)
    np.testing.assert_array_equal(arm[0], [2, 3])
    np.testing.assert_array_equal(gripper, [0, 1, 1, 0])

