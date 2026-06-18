from __future__ import annotations

from typing import Any, Mapping

import numpy as np

ROBOT_AXES = {"x": 0, "y": 1, "z": 2}
HUMAN_AXES = ("human_x", "human_y", "human_z")


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


def auto_calibrate_scale(
    wrist_smooth: np.ndarray,
    config: Mapping[str, Any],
    *,
    workspace_utilization: float = 0.85,
    max_scale: float = 1.0,
    min_scale: float = 0.05,
    percentile_clip: float = 5.0,
) -> dict[str, float]:
    """Compute optimal per-axis scale from wrist motion amplitude.

    Analyzes the range of human wrist motion in each axis and computes
    a scale factor that maps the motion to a fraction of the available
    robot workspace, preventing overflow that causes IK failure.

    Parameters
    ----------
    wrist_smooth : np.ndarray
        Smoothed wrist positions, shape [T, 3] in normalized image coords.
    config : Mapping
        Retarget config containing axis_mapping, workspace_limit, and
        openarm_origin.
    workspace_utilization : float
        Fraction of robot workspace to use (0.0-1.0). Lower values leave
        more margin for IK convergence. Default 0.85 (85%).
    max_scale : float
        Maximum allowed scale per axis. Prevents tiny motions from being
        amplified excessively. Default 1.0.
    min_scale : float
        Minimum allowed scale per axis. Prevents near-zero scale when
        motion is negligible. Default 0.05.
    percentile_clip : float
        Percentile to clip outliers before computing amplitude. E.g., 5.0
        means use 5th-95th percentile range. Default 5.0.

    Returns
    -------
    dict[str, float]
        Computed scale factors for x, y, z axes.
    """
    wrist = np.asarray(wrist_smooth, dtype=float)
    if wrist.ndim != 2 or wrist.shape[1] != 3:
        raise ValueError(f"wrist_smooth must have shape [T, 3], got {wrist.shape}")

    mapping = config.get("axis_mapping", {})
    limits = config.get("workspace_limit", {})
    origin = np.asarray(config["openarm_origin"], dtype=float)

    # Compute human motion amplitude per axis (in image space)
    low_pct = percentile_clip
    high_pct = 100.0 - percentile_clip
    human_amplitudes = np.zeros(3)
    for human_index in range(3):
        axis_data = wrist[:, human_index]
        finite_mask = np.isfinite(axis_data)
        if not np.any(finite_mask):
            human_amplitudes[human_index] = 0.0
            continue
        finite_data = axis_data[finite_mask]
        # Use percentile range to ignore outliers
        p_low = np.percentile(finite_data, low_pct)
        p_high = np.percentile(finite_data, high_pct)
        human_amplitudes[human_index] = p_high - p_low

    # Compute available robot workspace per axis
    robot_room = np.zeros(3)
    for axis_name, axis_index in ROBOT_AXES.items():
        bounds = np.asarray(
            limits.get(axis_name, [-np.inf, np.inf]), dtype=float
        )
        if np.all(np.isfinite(bounds)):
            robot_room[axis_index] = bounds[1] - bounds[0]
        else:
            # If no workspace limit, use a default room of 0.5m
            robot_room[axis_index] = 0.5

    # Compute scale per human axis → mapped robot axis
    computed_scale = {}
    for human_index, human_axis in enumerate(HUMAN_AXES):
        robot_index, _ = _parse_mapping(
            mapping.get(human_axis, "xyz"[human_index])
        )
        amplitude = human_amplitudes[human_index]
        room = robot_room[robot_index]

        if amplitude < 1e-6:
            # Negligible motion: use minimum scale
            scale_value = min_scale
        else:
            # Scale so that human amplitude maps to workspace_utilization * room
            scale_value = (workspace_utilization * room) / amplitude
            scale_value = np.clip(scale_value, min_scale, max_scale)

        computed_scale["xyz"[human_index]] = float(scale_value)

    return computed_scale


def auto_calibrate_origin(
    wrist_smooth: np.ndarray,
    config: Mapping[str, Any],
    *,
    percentile_clip: float = 5.0,
) -> np.ndarray:
    """Compute optimal origin from median wrist position.

    Instead of using a fixed origin, computes the origin so that the
    median human wrist position maps to the center of the robot workspace.

    Parameters
    ----------
    wrist_smooth : np.ndarray
        Smoothed wrist positions, shape [T, 3] in normalized image coords.
    config : Mapping
        Retarget config containing axis_mapping, workspace_limit.
    percentile_clip : float
        Percentile to clip outliers. Default 5.0.

    Returns
    -------
    np.ndarray
        Computed origin [x, y, z] in robot coordinates.
    """
    wrist = np.asarray(wrist_smooth, dtype=float)
    if wrist.ndim != 2 or wrist.shape[1] != 3:
        raise ValueError(f"wrist_smooth must have shape [T, 3], got {wrist.shape}")

    mapping = config.get("axis_mapping", {})
    limits = config.get("workspace_limit", {})

    # Compute median human wrist position
    low_pct = percentile_clip
    high_pct = 100.0 - percentile_clip
    human_median = np.zeros(3)
    for human_index in range(3):
        axis_data = wrist[:, human_index]
        finite_mask = np.isfinite(axis_data)
        if np.any(finite_mask):
            finite_data = axis_data[finite_mask]
            human_median[human_index] = np.median(finite_data)
        else:
            human_median[human_index] = 0.5  # fallback to center

    # Compute center of robot workspace
    robot_center = np.zeros(3)
    for axis_name, axis_index in ROBOT_AXES.items():
        bounds = np.asarray(
            limits.get(axis_name, [-np.inf, np.inf]), dtype=float
        )
        if np.all(np.isfinite(bounds)):
            robot_center[axis_index] = (bounds[0] + bounds[1]) / 2.0
        else:
            robot_center[axis_index] = 0.0

    # Compute origin: we want median_human * scale → robot_center
    # Since retarget_wrist does: target = origin + (wrist - wrist[0]) * scale
    # At median: target = origin + (median - wrist[0]) * scale = robot_center
    # So: origin = robot_center - (median - wrist[0]) * scale
    # But we need scale first, so we use the configured/auto-computed scale
    scale_config = config.get("scale", {})
    scale = np.asarray(
        [scale_config.get(axis, 1.0) for axis in ("x", "y", "z")], dtype=float
    )

    delta_median = human_median - wrist[0]
    origin = np.zeros(3)
    for human_index, human_axis in enumerate(HUMAN_AXES):
        robot_index, sign = _parse_mapping(
            mapping.get(human_axis, "xyz"[human_index])
        )
        origin[robot_index] = (
            robot_center[robot_index]
            - sign * delta_median[human_index] * scale[human_index]
        )

    return origin.astype(np.float64)


def retarget_wrist_auto(
    wrist_smooth: np.ndarray,
    config: Mapping[str, Any],
) -> np.ndarray:
    """Retarget with automatic scale and origin calibration.

    If ``auto_scale`` is True in config, computes per-axis scale from
    motion amplitude. If ``auto_origin`` is True, computes origin from
    median wrist position. Then delegates to :func:`retarget_wrist`.

    Parameters
    ----------
    wrist_smooth : np.ndarray
        Smoothed wrist positions, shape [T, 3].
    config : Mapping
        Retarget config. Recognized extra keys:
        - ``auto_scale`` (bool): Enable auto scale calibration.
        - ``auto_origin`` (bool): Enable auto origin calibration.
        - ``workspace_utilization`` (float): Fraction of workspace to use.
        - ``max_scale`` (float): Maximum scale per axis.
        - ``min_scale`` (float): Minimum scale per axis.

    Returns
    -------
    np.ndarray
        Robot target positions, shape [T, 3].
    """
    config = dict(config)  # make mutable copy

    if config.get("auto_scale", False):
        computed_scale = auto_calibrate_scale(
            wrist_smooth,
            config,
            workspace_utilization=float(
                config.get("workspace_utilization", 0.85)
            ),
            max_scale=float(config.get("max_scale", 1.0)),
            min_scale=float(config.get("min_scale", 0.05)),
        )
        config["scale"] = computed_scale
        # When scale changes, origin must also be adjusted so that
        # the median wrist position maps to the center of the workspace.
        # Otherwise targets overflow on one side.
        config["auto_origin"] = True

    if config.get("auto_origin", False):
        computed_origin = auto_calibrate_origin(wrist_smooth, config)
        config["openarm_origin"] = computed_origin.tolist()

    return retarget_wrist(wrist_smooth, config)

