from __future__ import annotations

import numpy as np


def generate_hand_pose(
    *,
    frames: int = 180,
    fps: float = 30.0,
    noise_std: float = 0.002,
    seed: int = 7,
) -> dict[str, np.ndarray]:
    if frames < 10 or fps <= 0:
        raise ValueError("frames must be >= 10 and fps must be positive")
    rng = np.random.default_rng(seed)
    phase = np.linspace(0.0, 2.0 * np.pi, frames, endpoint=False)
    wrist = np.column_stack(
        (
            0.50 + 0.16 * np.sin(phase),
            0.52 + 0.10 * np.sin(phase * 2.0),
            -0.02 + 0.04 * np.cos(phase),
        )
    )
    wrist += rng.normal(0.0, noise_std, wrist.shape)

    thumb = wrist + np.array([0.035, -0.025, -0.01])
    offsets = np.array(
        [
            [0.095, -0.055, -0.005],
            [0.070, -0.090, 0.000],
            [0.035, -0.095, 0.005],
            [0.005, -0.080, 0.010],
        ]
    )
    fingers = wrist[:, None, :] + offsets[None, :, :]
    pinch_mask = ((np.arange(frames) >= frames * 0.22) & (np.arange(frames) < frames * 0.42)) | (
        (np.arange(frames) >= frames * 0.62) & (np.arange(frames) < frames * 0.80)
    )
    fingers[pinch_mask, 0, :] = thumb[pinch_mask] + np.array([0.012, 0.0, 0.0])
    fingers += rng.normal(0.0, noise_std, fingers.shape)

    valid = np.ones(frames, dtype=bool)
    gap_start = frames // 2
    valid[gap_start : min(frames, gap_start + 5)] = False
    arrays = {
        "wrist": wrist,
        "thumb_tip": thumb,
        "index_tip": fingers[:, 0],
        "middle_tip": fingers[:, 1],
        "ring_tip": fingers[:, 2],
        "pinky_tip": fingers[:, 3],
    }
    for value in arrays.values():
        value[~valid] = np.nan

    return {
        "timestamps": np.arange(frames, dtype=np.float64) / fps,
        "valid": valid,
        "handedness": np.asarray("Right"),
        **{key: value.astype(np.float32) for key, value in arrays.items()},
    }

