from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("openarm_mujoco")

import mujoco

from openarm_retarget.mujoco_replay import (
    _gripper_target,
    _render_camera,
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

    for side, sign in (("left", 1), ("right", -1)):
        indices = [
            model.jnt_qposadr[
                mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    f"openarm_{side}_finger_joint{number}",
                )
            ]
            for number in (1, 2)
        ]
        assert np.all(sign * achieved[0, indices] > 0)
        assert np.all(achieved[1, indices] == 0)

        distances = []
        fingertip_geom_ids = [
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                f"finger_{position}_{side}_02",
            )
            for position in ("inner", "outer")
        ]
        for frame in achieved:
            data.qpos[:] = frame
            mujoco.mj_forward(model, data)
            distances.append(
                np.linalg.norm(
                    data.geom_xpos[fingertip_geom_ids[0]]
                    - data.geom_xpos[fingertip_geom_ids[1]]
                )
            )
        assert distances[0] > distances[1]


def test_gripper_target_uses_zero_as_closed_for_signed_ranges():
    assert _gripper_target(0, np.array([0.0, 0.8])) == 0.8
    assert _gripper_target(1, np.array([0.0, 0.8])) == 0
    assert _gripper_target(0, np.array([-0.8, 0.0])) == -0.8
    assert _gripper_target(1, np.array([-0.8, 0.0])) == 0


def test_free_render_camera_uses_front_view_configuration():
    model, _ = load_bimanual_openarm(
        {
            "model_asset": "cell.xml",
            "sides": ["left", "right"],
            "home_keyframe": "home",
        }
    )
    camera = _render_camera(
        model,
        {
            "mode": "free",
            "lookat": [0.4, 0.0, 1.05],
            "distance": 1.05,
            "azimuth": 0,
            "elevation": -8,
        },
    )

    np.testing.assert_allclose(camera.lookat, [0.4, 0.0, 1.05])
    assert camera.distance == pytest.approx(1.05)
    assert camera.azimuth == pytest.approx(0)
    assert camera.elevation == pytest.approx(-8)
