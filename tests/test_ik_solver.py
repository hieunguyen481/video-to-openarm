from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("openarm_mujoco")

import mujoco

from openarm_retarget.ik_solver import JacobianIKSolver
from openarm_retarget.openarm_model import load_openarm, reset_home


def test_jacobian_ik_tracks_small_openarm_motion():
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
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, info.ee_site)
    origin = data.site_xpos[site_id].copy()
    targets = np.stack(
        [origin, origin + [0.01, 0.0, 0.0], origin + [0.01, 0.01, 0.0]]
    )
    solver = JacobianIKSolver(
        model,
        info,
        {
            "tolerance": 0.005,
            "max_iterations": 100,
            "damping": 0.01,
            "step_size": 0.5,
            "max_delta_q": 0.05,
            "max_frame_delta_q": 0.2,
        },
    )

    result = solver.solve(targets)

    assert np.max(result.ik_error) < 0.01
    assert np.all(np.isfinite(result.qpos))

