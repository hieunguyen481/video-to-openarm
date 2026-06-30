from __future__ import annotations

import numpy as np

from openarm_retarget.retargeting import retarget_wrist


def test_axis_mapping_scale_and_workspace_clamp():
    wrist = np.array([[0, 0, 0], [1, 2, 3]], dtype=float)
    config = {
        "openarm_origin": [0.3, 0.0, 0.4],
        "scale": {"x": 0.5, "y": 0.25, "z": 0.1},
        "axis_mapping": {
            "human_x": "x",
            "human_y": "z_negative",
            "human_z": "y",
        },
        "workspace_limit": {
            "x": [0.0, 0.6],
            "y": [-0.2, 0.2],
            "z": [0.1, 0.7],
        },
    }
    target = retarget_wrist(wrist, config)

    np.testing.assert_allclose(target[0], [0.3, 0.0, 0.4])
    np.testing.assert_allclose(target[1], [0.6, 0.2, 0.1])


def test_opposing_camera_mapping_mirrors_horizontal_axis():
    wrist = np.array([[0, 0, 0], [1, 2, 3]], dtype=float)
    config = {
        "openarm_origin": [0.4, 0.1, 1.1],
        "scale": {"x": 0.1, "y": 0.2, "z": 0.05},
        "axis_mapping": {
            "human_x": "y_negative",
            "human_y": "z_negative",
            "human_z": "x_negative",
        },
    }

    target = retarget_wrist(wrist, config)

    np.testing.assert_allclose(target[1], [0.25, 0.0, 0.7])


def test_ego_centric_mapping_no_mirror():
    """Ego-centric camera: human_x -> y (no negate), human_z -> x (no negate)."""
    wrist = np.array([[0, 0, 0], [1, 2, 3]], dtype=float)
    config = {
        "openarm_origin": [0.4, 0.1, 1.1],
        "scale": {"x": 0.1, "y": 0.2, "z": 0.05},
        "axis_mapping": {
            "human_x": "y",
            "human_y": "z_negative",
            "human_z": "x",
        },
    }

    target = retarget_wrist(wrist, config)

    # human_x=1 * scale_x=0.1 -> robot_y += 0.1 -> 0.1+0.1 = 0.2
    # human_y=2 * scale_y=0.2 -> robot_z -= 0.4 -> 1.1-0.4 = 0.7
    # human_z=3 * scale_z=0.05 -> robot_x += 0.15 -> 0.4+0.15 = 0.55
    np.testing.assert_allclose(target[1], [0.55, 0.2, 0.7])


def test_explicit_human_reference_preserves_absolute_layout():
    wrist = np.array([[0.25, 0.4, 0.5], [0.75, 0.6, 0.5]], dtype=float)
    config = {
        "openarm_origin": [0.4, 0.0, 1.1],
        "human_reference": [0.5, 0.5, 0.5],
        "scale": {"x": 0.4, "y": 0.4, "z": 0.2},
        "axis_mapping": {
            "human_x": "y_negative",
            "human_y": "z_negative",
            "human_z": "x",
        },
    }

    target = retarget_wrist(wrist, config)

    np.testing.assert_allclose(target[0], [0.4, 0.1, 1.14])
    np.testing.assert_allclose(target[1], [0.4, -0.1, 1.06])
