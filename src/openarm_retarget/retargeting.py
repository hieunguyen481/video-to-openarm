from __future__ import annotations

from typing import Any, Mapping

import numpy as np

ROBOT_AXES = {"x": 0, "y": 1, "z": 2}


def _parse_mapping(value: str) -> tuple[int, float]:
    normalized = value.lower().strip()
    sign = -1.0 if normalized.endswith("_negative") else 1.0
    axis_name = normalized.removesuffix("_negative").removesuffix("_positive")
    if axis_name not in ROBOT_AXES:
        raise ValueError(
            f"Invalid axis mapping {value!r}; expected x/y/z with optional _negative"
        )
    return ROBOT_AXES[axis_name], sign


def retarget_wrist(
    wrist_smooth: np.ndarray,
    config: Mapping[str, Any],
) -> np.ndarray:
    wrist = np.asarray(wrist_smooth, dtype=float)
    if wrist.ndim != 2 or wrist.shape[1] != 3:
        raise ValueError(f"wrist_smooth must have shape [T, 3], got {wrist.shape}")
    if not np.all(np.isfinite(wrist)):
        raise ValueError("wrist_smooth contains NaN or infinite values")

    origin = np.asarray(config["openarm_origin"], dtype=float)
    if origin.shape != (3,):
        raise ValueError("openarm_origin must contain x, y, z")
    scale_config = config.get("scale", {})
    scale = np.asarray(
        [scale_config.get(axis, 1.0) for axis in ("x", "y", "z")], dtype=float
    )
    mapping = config.get("axis_mapping", {})
    transformed = np.zeros_like(wrist)
    delta = wrist - wrist[0]
    for human_index, human_axis in enumerate(("human_x", "human_y", "human_z")):
        robot_index, sign = _parse_mapping(mapping.get(human_axis, "xyz"[human_index]))
        transformed[:, robot_index] += sign * delta[:, human_index] * scale[human_index]

    target = transformed + origin
    limits = config.get("workspace_limit", {})
    for axis_name, axis_index in ROBOT_AXES.items():
        bounds = np.asarray(limits.get(axis_name, [-np.inf, np.inf]), dtype=float)
        if bounds.shape != (2,) or bounds[0] > bounds[1]:
            raise ValueError(f"Invalid workspace bounds for axis {axis_name}: {bounds}")
        target[:, axis_index] = np.clip(
            target[:, axis_index], bounds[0], bounds[1]
        )
    return target.astype(np.float32)

