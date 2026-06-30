"""Remove shared HaWoR/SLAM translation drift from a bimanual pose."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from openarm_retarget.io import load_npz, save_npz


SIDES = ("left", "right")
POINT_NAMES = (
    "wrist",
    "thumb_tip",
    "index_tip",
    "middle_tip",
    "ring_tip",
    "pinky_tip",
)


def stabilize_pose(
    pose: dict[str, np.ndarray],
    *,
    strength: float = 1.0,
) -> dict[str, np.ndarray]:
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1")

    output = {key: np.asarray(value).copy() for key, value in pose.items()}
    wrists = np.stack(
        [np.asarray(output[f"{side}_wrist"], dtype=np.float32) for side in SIDES],
        axis=0,
    )
    valid = np.stack(
        [np.asarray(output[f"{side}_valid"], dtype=bool) for side in SIDES],
        axis=0,
    )
    usable = valid[:, :, None] & np.isfinite(wrists)
    count = usable.sum(axis=0)
    shared = np.divide(
        np.where(usable, wrists, 0.0).sum(axis=0),
        count,
        out=np.full_like(wrists[0], np.nan),
        where=count > 0,
    )

    # Fill frames where neither hand is valid, then anchor shared translation
    # to the first stable frame. Relative motion between the hands is retained.
    for axis in range(3):
        values = shared[:, axis]
        finite = np.flatnonzero(np.isfinite(values))
        if finite.size == 0:
            values[:] = 0.0
            continue
        values[: finite[0]] = values[finite[0]]
        for index in range(finite[0] + 1, len(values)):
            if not np.isfinite(values[index]):
                values[index] = values[index - 1]
    shared_delta = (shared - shared[0]) * np.float32(strength)

    for side in SIDES:
        for name in POINT_NAMES:
            key = f"{side}_{name}"
            if key in output:
                output[key] = (
                    np.asarray(output[key], dtype=np.float32) - shared_delta
                )
        palm_key = f"{side}_palm_scale"
        if palm_key in output:
            output[palm_key] = output[f"{side}_wrist"][:, 2].astype(np.float32)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    args = parser.parse_args()

    stabilized = stabilize_pose(load_npz(args.input), strength=args.strength)
    save_npz(
        args.output,
        stabilized,
        stage="hawor_world_stabilized_bimanual_hand_pose",
        metadata={
            "source": str(args.input),
            "shared_translation_removal_strength": args.strength,
        },
    )
    print(f"Saved stabilized pose: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
