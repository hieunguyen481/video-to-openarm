from __future__ import annotations

import numpy as np

from openarm_retarget.hawor_adapter import (
    HAWOR_HAND_INDEX,
    _normalize_points,
    _proxy_joints_from_wrist,
)


def test_normalize_points_uses_valid_frames_only():
    points = np.zeros((2, 3, 21, 3), dtype=np.float32)
    points[0, 0, :, :] = 1.0
    points[1, 2, :, :] = 3.0
    valid = np.zeros((2, 3), dtype=bool)
    valid[0, 0] = True
    valid[1, 2] = True

    normalized = _normalize_points(points, valid, workspace_scale=2.0)

    assert normalized.shape == points.shape
    np.testing.assert_allclose(normalized[0, 0, 0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(normalized[1, 2, 0], [1.0, 1.0, 1.0])


def test_proxy_joints_have_required_landmarks():
    wrist = np.zeros((5, 3), dtype=np.float32)

    joints = _proxy_joints_from_wrist(wrist)

    assert joints.shape == (5, 21, 3)
    assert np.any(joints[:, 4, :] != joints[:, 0, :])
    assert HAWOR_HAND_INDEX == {"left": 0, "right": 1}
