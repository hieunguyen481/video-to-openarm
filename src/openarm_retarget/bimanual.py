from __future__ import annotations

from typing import Any, Mapping

import numpy as np

LANDMARK_KEYS = (
    "wrist",
    "thumb_tip",
    "index_tip",
    "middle_tip",
    "ring_tip",
    "pinky_tip",
)


def side_pose(data: Mapping[str, Any], side: str) -> dict[str, np.ndarray]:
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")
    result = {
        "timestamps": np.asarray(data["timestamps"]),
        "valid": np.asarray(data[f"{side}_valid"]),
        "handedness": np.asarray(side.title()),
        **{
            key: np.asarray(data[f"{side}_{key}"])
            for key in LANDMARK_KEYS
        },
    }
    palm_key = f"{side}_palm_scale"
    if palm_key in data:
        result["palm_scale"] = np.asarray(data[palm_key])
    return result


def prefix_fields(
    side: str,
    data: Mapping[str, Any],
    *,
    exclude: tuple[str, ...] = ("timestamps",),
) -> dict[str, np.ndarray]:
    return {
        f"{side}_{key}": np.asarray(value)
        for key, value in data.items()
        if key not in exclude
    }
