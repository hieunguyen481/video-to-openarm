from __future__ import annotations

import numpy as np

from openarm_retarget.smoothing import interpolate_missing, smooth_wrist


def test_interpolation_fills_internal_and_edge_gaps():
    values = np.array(
        [[np.nan], [1.0], [np.nan], [3.0], [np.nan]], dtype=float
    )
    valid = np.array([False, True, False, True, False])
    result = interpolate_missing(values, valid)
    np.testing.assert_allclose(result[:, 0], [1, 1, 2, 3, 3])


def test_smoothing_returns_finite_bounded_trajectory():
    wrist = np.zeros((10, 3), dtype=float)
    wrist[5:, 0] = 10
    wrist[3] = np.nan
    valid = np.ones(10, dtype=bool)
    valid[3] = False
    timestamps = np.arange(10) / 10

    result = smooth_wrist(
        wrist, valid, window=3, max_speed=1.0, timestamps=timestamps
    )

    assert np.all(np.isfinite(result))
    speed = np.linalg.norm(np.diff(result, axis=0), axis=1) / np.diff(timestamps)
    assert np.max(speed) <= 1.00001

