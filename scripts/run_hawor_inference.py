"""Run HaWoR inference without the Qt/aitviewer visualization step."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HaWoR detect/reconstruct/SLAM/infiller and stop at world_space_res.pth."
    )
    parser.add_argument(
        "--hawor-root",
        type=Path,
        default=Path("external_repos/HaWoR"),
        help="Path to the external HaWoR checkout.",
    )
    parser.add_argument("--video", required=True, help="Video path for HaWoR.")
    parser.add_argument("--img-focal", type=float)
    parser.add_argument(
        "--checkpoint",
        default="./weights/hawor/checkpoints/hawor.ckpt",
        help="HaWoR checkpoint path relative to HaWoR root.",
    )
    parser.add_argument(
        "--infiller-weight",
        default="./weights/hawor/checkpoints/infiller.pt",
        help="HaWoR infiller path relative to HaWoR root.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.hawor_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"HaWoR root not found: {root}")

    video = Path(args.video)
    if not video.is_absolute():
        video = Path.cwd() / video
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")

    sys.path.insert(0, str(root))
    os.chdir(root)

    from lib.eval_utils.custom_utils import load_slam_cam
    from scripts.scripts_test_video.detect_track_video import detect_track_video
    from scripts.scripts_test_video.hawor_slam import hawor_slam
    from scripts.scripts_test_video.hawor_video import (
        hawor_infiller,
        hawor_motion_estimation,
    )

    class HaWoRArgs:
        video_path = str(video)
        input_type = "file"
        img_focal = args.img_focal
        checkpoint = args.checkpoint
        infiller_weight = args.infiller_weight

    hawor_args = HaWoRArgs()
    start_idx, end_idx, seq_folder, _ = detect_track_video(hawor_args)
    frame_chunks_all, _ = hawor_motion_estimation(
        hawor_args, start_idx, end_idx, seq_folder
    )
    slam_path = os.path.join(
        seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz"
    )
    if not os.path.exists(slam_path):
        hawor_slam(hawor_args, start_idx, end_idx)
    load_slam_cam(slam_path)
    hawor_infiller(hawor_args, start_idx, end_idx, frame_chunks_all)
    output = Path(seq_folder) / "world_space_res.pth"
    print(f"HaWoR output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
