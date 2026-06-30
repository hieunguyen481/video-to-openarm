"""Temporally filter normalized WiLoR 3D landmarks before retargeting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.signal import medfilt, savgol_filter


SIDES = ("left", "right")
LANDMARKS = (
    "wrist",
    "thumb_tip",
    "index_tip",
    "middle_tip",
    "ring_tip",
    "pinky_tip",
)


def _interpolate(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=np.float32).copy()
    frames = np.arange(len(output))
    for axis in range(output.shape[1]):
        usable = valid & np.isfinite(output[:, axis])
        if np.any(usable):
            output[:, axis] = np.interp(
                frames, frames[usable], output[usable, axis]
            )
        else:
            output[:, axis] = 0.0
    return output


def _reject_jumps(
    wrist: np.ndarray,
    valid: np.ndarray,
    *,
    max_step: float,
) -> np.ndarray:
    accepted = np.asarray(valid, dtype=bool).copy()
    previous: np.ndarray | None = None
    rejected_run = 0
    for index, point in enumerate(wrist):
        if not accepted[index] or not np.isfinite(point).all():
            accepted[index] = False
            continue
        if previous is not None and np.linalg.norm(point - previous) > max_step:
            rejected_run += 1
            if rejected_run < 5:
                accepted[index] = False
                continue
        rejected_run = 0
        previous = point
    return accepted


def _smooth(values: np.ndarray, *, median_window: int, smooth_window: int) -> np.ndarray:
    output = np.asarray(values, dtype=np.float32).copy()
    for axis in range(output.shape[1]):
        output[:, axis] = medfilt(output[:, axis], kernel_size=median_window)
        output[:, axis] = savgol_filter(
            output[:, axis],
            window_length=smooth_window,
            polyorder=2,
            mode="interp",
        )
    return output.astype(np.float32)


def _smooth_rotations(
    matrices: np.ndarray,
    valid: np.ndarray,
    *,
    smooth_window: int,
) -> np.ndarray:
    values = np.asarray(matrices, dtype=np.float64).copy()
    frames = np.arange(len(values))
    usable = valid & np.isfinite(values).all(axis=(1, 2))
    if not np.any(usable):
        return values.astype(np.float32)
    for row in range(3):
        for column in range(3):
            values[:, row, column] = np.interp(
                frames,
                frames[usable],
                values[usable, row, column],
            )
    # Project interpolated matrices back onto SO(3).
    u, _, vh = np.linalg.svd(values)
    rotations = u @ vh
    negative = np.linalg.det(rotations) < 0
    u[negative, :, -1] *= -1
    rotations = u @ vh
    quaternions = Rotation.from_matrix(rotations).as_quat()
    for index in range(1, len(quaternions)):
        if np.dot(quaternions[index - 1], quaternions[index]) < 0:
            quaternions[index] *= -1
    for axis in range(4):
        quaternions[:, axis] = savgol_filter(
            quaternions[:, axis],
            window_length=smooth_window,
            polyorder=2,
            mode="interp",
        )
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    return Rotation.from_quat(quaternions).as_matrix().astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-step", type=float, default=0.20)
    parser.add_argument("--median-window", type=int, default=7)
    parser.add_argument("--smooth-window", type=int, default=15)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for value, name in (
        (args.median_window, "median-window"),
        (args.smooth_window, "smooth-window"),
    ):
        if value < 3 or value % 2 == 0:
            raise ValueError(f"{name} must be an odd integer >= 3")

    with np.load(args.input, allow_pickle=False) as archive:
        source = {key: archive[key] for key in archive.files}
    output = {
        key: np.asarray(value).copy()
        for key, value in source.items()
        if not key.startswith("_")
    }
    diagnostics: dict[str, object] = {}
    for side in SIDES:
        original_valid = np.asarray(source[f"{side}_valid"], dtype=bool)
        accepted = _reject_jumps(
            np.asarray(source[f"{side}_wrist"]),
            original_valid,
            max_step=args.max_step,
        )
        output[f"{side}_valid"] = accepted
        diagnostics[f"{side}_original_valid_ratio"] = float(
            np.mean(original_valid)
        )
        diagnostics[f"{side}_filtered_valid_ratio"] = float(np.mean(accepted))
        diagnostics[f"{side}_rejected_jumps"] = int(
            np.count_nonzero(original_valid & ~accepted)
        )
        for landmark in LANDMARKS:
            key = f"{side}_{landmark}"
            interpolated = _interpolate(source[key], accepted)
            output[key] = _smooth(
                interpolated,
                median_window=args.median_window,
                smooth_window=args.smooth_window,
            )
        output[f"{side}_palm_scale"] = output[f"{side}_wrist"][:, 2].copy()
        orientation_key = f"{side}_global_orient"
        if orientation_key in source:
            output[orientation_key] = _smooth_rotations(
                source[orientation_key],
                original_valid,
                smooth_window=args.smooth_window,
            )

    output["_schema_version"] = np.asarray("1.0")
    output["_stage"] = np.asarray("wilor_full_3d_filtered_pose")
    output["_metadata_json"] = np.asarray(
        json.dumps(
            {
                "source": str(args.input),
                "max_step": args.max_step,
                "median_window": args.median_window,
                "smooth_window": args.smooth_window,
                **diagnostics,
            },
            sort_keys=True,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)
    args.output.chmod(0o644)
    print(json.dumps({"output": str(args.output), **diagnostics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
