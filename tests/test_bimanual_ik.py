from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("openarm_mujoco")

import mujoco

from openarm_retarget.ik_solver import BimanualJacobianIKSolver
from openarm_retarget.openarm_model import load_bimanual_openarm, reset_home


def test_bimanual_ik_tracks_both_end_effectors():
    model, info = load_bimanual_openarm(
        {
            "model_asset": "cell.xml",
            "sides": ["left", "right"],
            "home_keyframe": "home",
        }
    )
    data = mujoco.MjData(model)
    reset_home(model, data, info.home_keyframe)
    origins = {}
    for side in ("left", "right"):
        site_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            info.sides[side].ee_site,
        )
        origins[side] = data.site_xpos[site_id].copy()
    left_targets = np.stack(
        (origins["left"], origins["left"] + [0.01, 0.01, 0.0])
    )
    right_targets = np.stack(
        (origins["right"], origins["right"] + [0.01, -0.01, 0.0])
    )
    solver = BimanualJacobianIKSolver(
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

    result = solver.solve(left_targets, right_targets)

    assert result.left_arm_qpos.shape == (2, 7)
    assert result.right_arm_qpos.shape == (2, 7)
    assert np.max(result.left_ik_error) < 0.01
    assert np.max(result.right_ik_error) < 0.01
    assert np.all(result.converged)
