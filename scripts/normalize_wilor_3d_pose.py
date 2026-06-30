"""Convert raw WiLoR camera coordinates to robust image/depth coordinates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


SIDES = ("left", "right")
LANDMARKS = (
    "wrist",
    "thumb_tip",
    "index_tip",
    "middle_tip",
    "ring_tip",
    "pinky_tip",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-focal-length", type=float, default=5000.0)
    parser.add_argument("--model-image-size", type=float, default=256.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with np.load(args.input, allow_pickle=False) as archive:
        source = {key: archive[key] for key in archive.files}

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    focal = args.model_focal_length / args.model_image_size * max(width, height)

    valid_depths: list[np.ndarray] = []
    for side in SIDES:
        wrist = np.asarray(source[f"{side}_wrist"], dtype=np.float32)
        valid = np.asarray(source[f"{side}_valid"], dtype=bool)
        usable = valid & np.isfinite(wrist).all(axis=1) & (wrist[:, 2] > 0)
        valid_depths.append(np.log(wrist[usable, 2]))
    depth_values = np.concatenate(valid_depths)
    depth_near, depth_far = np.quantile(depth_values, [0.05, 0.95])
    depth_span = max(float(depth_far - depth_near), 1e-6)

    output = {
        key: np.asarray(value).copy()
        for key, value in source.items()
        if not key.startswith("_")
    }
    for side in SIDES:
        valid = np.asarray(output[f"{side}_valid"], dtype=bool)
        for landmark in LANDMARKS:
            key = f"{side}_{landmark}"
            camera = np.asarray(source[key], dtype=np.float32)
            z = camera[:, 2]
            safe_z = np.where(z > 1e-6, z, np.nan)
            normalized = np.empty_like(camera)
            normalized[:, 0] = (
                focal * camera[:, 0] / safe_z + width / 2.0
            ) / max(width - 1, 1)
            normalized[:, 1] = (
                focal * camera[:, 1] / safe_z + height / 2.0
            ) / max(height - 1, 1)
            normalized[:, 2] = np.clip(
                (np.log(safe_z) - depth_near) / depth_span,
                0.0,
                1.0,
            )
            normalized[~valid] = np.nan
            output[key] = normalized.astype(np.float32)
            output[f"{side}_camera_{landmark}"] = camera
        output[f"{side}_palm_scale"] = output[f"{side}_wrist"][:, 2].copy()

    output["_schema_version"] = np.asarray("1.0")
    output["_stage"] = np.asarray("wilor_full_3d_normalized_pose")
    output["_metadata_json"] = np.asarray(
        json.dumps(
            {
                "source": str(args.input),
                "video": str(args.video),
                "image_size": [width, height],
                "scaled_focal_length": focal,
                "log_depth_q05": float(depth_near),
                "log_depth_q95": float(depth_far),
            },
            sort_keys=True,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)
    args.output.chmod(0o644)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "frames": int(len(output["timestamps"])),
                "depth_m_q05": float(np.exp(depth_near)),
                "depth_m_q95": float(np.exp(depth_far)),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
