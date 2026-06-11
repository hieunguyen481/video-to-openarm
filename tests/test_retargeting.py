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

