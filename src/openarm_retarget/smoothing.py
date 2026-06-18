from __future__ import annotations

import numpy as np


def reject_outliers(
    values: np.ndarray,
    valid: np.ndarray,
    *,
    max_jump: float = 0.15,
) -> np.ndarray:
    """Mark frames with sudden landmark jumps as invalid.

    Compares consecutive valid frames and marks the later frame as invalid
    if the displacement exceeds ``max_jump``. This prevents tracking glitches
    from propagating through smoothing.

    Parameters
    ----------
    values : np.ndarray
        Landmark positions, shape [T, D].
    valid : np.ndarray
        Boolean validity mask, shape [T].
    max_jump : float
        Maximum allowed displacement between consecutive valid frames
        in normalized image coordinates. Default 0.15.

    Returns
    -------
    np.ndarray
        Updated validity mask with outlier frames set to False.
    """
    points = np.asarray(values, dtype=float)
    mask = np.asarray(valid, dtype=bool).copy()
    if points.ndim != 2:
        raise ValueError(f"values must be [T, D], got {points.shape}")
    if mask.shape != (len(points),):
        raise ValueError(f"valid must have shape {(len(points),)}, got {mask.shape}")
    if max_jump <= 0:
        raise ValueError("max_jump must be positive")

    last_valid_idx = None
    for i in range(len(points)):
        if not mask[i]:
            continue
        if not np.all(np.isfinite(points[i])):
            mask[i] = False
            continue
        if last_valid_idx is not None:
            delta = np.linalg.norm(points[i] - points[last_valid_idx])
            if delta > max_jump:
                mask[i] = False
                continue
        last_valid_idx = i

    return mask


def interpolate_missing(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    points = np.asarray(values, dtype=float)
    mask = np.asarray(valid, dtype=bool)
    if points.ndim != 2:
        raise ValueError(f"values must be [T, D], got {points.shape}")
    if mask.shape != (len(points),):
        raise ValueError(f"valid must have shape {(len(points),)}, got {mask.shape}")
    mask &= np.all(np.isfinite(points), axis=1)
    if not np.any(mask):
        raise ValueError("Cannot interpolate a trajectory without any valid samples")

    frame = np.arange(len(points))
    result = points.copy()
    for axis in range(points.shape[1]):
        result[:, axis] = np.interp(frame, frame[mask], points[mask, axis])
    return result


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    points = np.asarray(values, dtype=float)
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    if window == 1:
        return points.copy()
    radius = window // 2
    padded = np.pad(points, ((radius, radius), (0, 0)), mode="edge")
    kernel = np.full(window, 1.0 / window)
    return np.stack(
        [np.convolve(padded[:, axis], kernel, mode="valid") for axis in range(points.shape[1])],
        axis=1,
    )


def clamp_velocity(
    values: np.ndarray,
    max_speed: float,
    timestamps: np.ndarray | None = None,
) -> np.ndarray:
    if max_speed <= 0:
        raise ValueError("max_speed must be positive")
    points = np.asarray(values, dtype=float)
    if timestamps is None:
        delta_t = np.ones(len(points) - 1)
    else:
        times = np.asarray(timestamps, dtype=float)
        if times.shape != (len(points),):
            raise ValueError("timestamps must have shape [T]")
        delta_t = np.diff(times)
        if np.any(delta_t <= 0):
            raise ValueError("timestamps must be strictly increasing")

    result = points.copy()
    for index in range(1, len(result)):
        delta = result[index] - result[index - 1]
        distance = float(np.linalg.norm(delta))
        allowed = max_speed * delta_t[index - 1]
        if distance > allowed:
            result[index] = result[index - 1] + delta * (allowed / distance)
    return result


def smooth_wrist(
    wrist: np.ndarray,
    valid: np.ndarray,
    *,
    window: int = 7,
    max_speed: float = 2.0,
    timestamps: np.ndarray | None = None,
    max_jump: float = 0.15,
) -> np.ndarray:
    cleaned_valid = reject_outliers(wrist, valid, max_jump=max_jump)
    interpolated = interpolate_missing(wrist, cleaned_valid)
    averaged = moving_average(interpolated, window)
    return clamp_velocity(averaged, max_speed, timestamps).astype(np.float32)

