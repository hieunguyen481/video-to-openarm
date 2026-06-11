from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .validation import require_shape, require_time_length

FINGER_KEYS = ("index_tip", "middle_tip", "ring_tip", "pinky_tip")


def detect_pinch(
    hand_pose: Mapping[str, Any],
    *,
    close_threshold: float,
    open_threshold: float,
    initial_state: int = 0,
    invalid_policy: str = "hold",
) -> dict[str, np.ndarray]:
    if close_threshold >= open_threshold:
        raise ValueError("close_threshold must be smaller than open_threshold")
    if invalid_policy not in {"hold", "open"}:
        raise ValueError("invalid_policy must be 'hold' or 'open'")

    keys = ("thumb_tip", *FINGER_KEYS, "valid")
    length = require_time_length(hand_pose, keys)
    thumb = require_shape("thumb_tip", hand_pose["thumb_tip"], (3,)).astype(float)
    fingers = np.stack(
        [require_shape(key, hand_pose[key], (3,)) for key in FINGER_KEYS],
        axis=1,
    ).astype(float)
    valid = np.asarray(hand_pose["valid"], dtype=bool)

    distances = np.linalg.norm(fingers - thumb[:, None, :], axis=2)
    finite = np.all(np.isfinite(distances), axis=1)
    usable = valid & finite
    pinch_distance = np.full(length, np.nan, dtype=np.float32)
    pinch_finger = np.full(length, -1, dtype=np.int8)
    pinch_distance[usable] = np.min(distances[usable], axis=1)
    pinch_finger[usable] = np.argmin(distances[usable], axis=1).astype(np.int8)

    state = float(bool(initial_state))
    command = np.empty(length, dtype=np.float32)
    for index in range(length):
        if not usable[index]:
            if invalid_policy == "open":
                state = 0.0
        elif pinch_distance[index] < close_threshold:
            state = 1.0
        elif pinch_distance[index] > open_threshold:
            state = 0.0
        command[index] = state

    return {
        "pinch_distance": pinch_distance,
        "pinch_finger": pinch_finger,
        "gripper_cmd": command,
    }

