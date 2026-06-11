from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("openarm_mujoco")

import mujoco

from openarm_retarget.mujoco_replay import replay_trajectory
from openarm_retarget.openarm_model import load_openarm, reset_home


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

