from __future__ import annotations

import numpy as np

from openarm_retarget.live_control import (
    LiveHandSample,
    LiveMetrics,
    LivePinchDetector,
    LiveRetargeter,
)
from openarm_retarget.live_teleop import LiveHandTracker


def _sample(
    wrist: tuple[float, float, float] = (0.4, 0.5, 0.0),
    *,
    palm_scale: float = 0.1,
    pinch_distance: float = 0.1,
) -> LiveHandSample:
    thumb = np.array([0.0, 0.0, 0.0], dtype=float)
    fingers = np.array(
        [
            [pinch_distance, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.3, 0.0, 0.0],
            [0.4, 0.0, 0.0],
        ],
        dtype=float,
    )
    return LiveHandSample(
        wrist=np.asarray(wrist, dtype=float),
        palm_scale=palm_scale,
        thumb_tip=thumb,
        finger_tips=fingers,
    )


def _retarget_config() -> dict[str, dict[str, object]]:
    side = {
        "openarm_origin": [0.4, 0.1, 1.1],
        "scale": {"x": 0.5, "y": 0.5, "z": 0.5},
        "axis_mapping": {
            "human_x": "y_negative",
            "human_y": "z_negative",
            "human_z": "x_negative",
        },
        "workspace_limit": {
            "x": [0.1, 0.7],
            "y": [-0.5, 0.5],
            "z": [0.6, 1.4],
        },
    }
    return {"left": side, "right": {**side, "openarm_origin": [0.4, -0.1, 1.1]}}


def test_live_retargeter_calibrates_holds_and_reanchors_after_loss():
    retargeter = LiveRetargeter(
        _retarget_config(),
        smoothing_tau_s=0.01,
        max_target_speed_m_s=10.0,
        lost_hand_timeout_s=0.5,
    )
    origin = retargeter.update("left", _sample(), 1.0)
    moved = retargeter.update(
        "left", _sample((0.5, 0.5, 0.0)), 1.1
    )

    np.testing.assert_allclose(origin, [0.4, 0.1, 1.1])
    assert moved[1] < origin[1]
    np.testing.assert_allclose(
        retargeter.update("left", None, 1.2), moved
    )
    retargeter.update("left", None, 1.7)
    reanchored = retargeter.update(
        "left", _sample((0.2, 0.3, 0.0)), 1.8
    )
    np.testing.assert_allclose(reanchored, moved)


def test_live_pinch_confirms_and_opens_after_tracking_loss():
    detector = LivePinchDetector(
        close_threshold=0.04,
        open_threshold=0.06,
        close_confirm_frames=2,
        open_confirm_frames=2,
        lost_hand_timeout_s=0.5,
    )

    assert detector.update(_sample(pinch_distance=0.02), 1.0) == 0
    assert detector.update(_sample(pinch_distance=0.02), 1.1) == 1
    assert detector.update(None, 1.2) == 1
    assert detector.update(None, 1.7) == 0


def test_live_metrics_reports_latency_components_and_skips():
    metrics = LiveMetrics()
    metrics.record(
        sequence=1,
        latency_ms=20,
        inference_ms=10,
        ik_ms=2,
        render_ms=3,
    )
    metrics.record(
        sequence=3,
        latency_ms=40,
        inference_ms=20,
        ik_ms=4,
        render_ms=5,
    )

    summary = metrics.summary()

    assert summary["processed_frames"] == 2
    assert summary["dropped_or_skipped_frames"] == 1
    assert summary["latency_ms"]["mean"] == 30
    assert summary["render_ms"]["max"] == 5


def test_live_tracker_can_toggle_left_right_without_detector():
    tracker = object.__new__(LiveHandTracker)
    tracker.swap_left_right = True

    assert tracker.toggle_left_right() is False
    assert tracker.toggle_left_right() is True
