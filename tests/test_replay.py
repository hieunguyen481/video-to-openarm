from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("openarm_mujoco")

import mujoco

from openarm_retarget.mujoco_replay import (
    _gripper_target,
    replay_bimanual_trajectory,
    replay_trajectory,
)
from openarm_retarget.openarm_model import (
    load_bimanual_openarm,
    load_openarm,
    reset_home,
)


def test_kinematic_replay_without_video_dependency():
    model, info = load_openarm(
        {
            "model_asset": "cell.xml",
            "side": "left",
            "home_keyframe": "home",
            "ee_site": "auto",
            "arm_joint_names": "auto",
            "gripper_actuator": "auto",
        }
    )
    data = mujoco.MjData(model)
    reset_home(model, data, info.home_keyframe)
    trajectory = np.repeat(data.qpos[None, :], 3, axis=0)

    achieved = replay_trajectory(
        model,
        info,
        trajectory,
        np.array([0.0, 1.0, 0.0]),
        mode="kinematic",
    )

    assert achieved.shape == trajectory.shape
    assert np.all(np.isfinite(achieved))


def test_bimanual_replay_controls_both_grippers():
    model, info = load_bimanual_openarm(
        {
            "model_asset": "cell.xml",
            "sides": ["left", "right"],
            "home_keyframe": "home",
        }
    )
    data = mujoco.MjData(model)
    reset_home(model, data, info.home_keyframe)
    trajectory = np.repeat(data.qpos[None, :], 2, axis=0)

    achieved = replay_bimanual_trajectory(
        model,
        info,
        trajectory,
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0]),
        mode="kinematic",
    )

    left_index = model.jnt_qposadr[
        model.actuator_trnid[
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                info.sides["left"].gripper_actuator,
            ),
            0,
        ]
    ]
    right_index = model.jnt_qposadr[
        model.actuator_trnid[
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                info.sides["right"].gripper_actuator,
            ),
            0,
        ]
    ]
    assert achieved[0, left_index] == 0
    assert achieved[0, right_index] == 0
    assert achieved[1, left_index] > 0
    assert achieved[1, right_index] < 0


def test_gripper_target_uses_zero_as_open_for_signed_ranges():
    assert _gripper_target(0, np.array([0.0, 0.8])) == 0
    assert _gripper_target(1, np.array([0.0, 0.8])) == 0.8
    assert _gripper_target(0, np.array([-0.8, 0.0])) == 0
    assert _gripper_target(1, np.array([-0.8, 0.0])) == -0.8
