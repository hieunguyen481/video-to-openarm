"""Run OpenArm retargeting from a converted HaWoR world-space pose NPZ."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from openarm_retarget.io import load_npz
from openarm_retarget.pipeline import run_bimanual_from_pose


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a converted HaWoR world-space pose with OpenArm."
    )
    parser.add_argument(
        "--pose",
        type=Path,
        default=Path(
            "outputs/external_trials/hawor_world/video_0_hawor_world_hand_pose.npz"
        ),
        help="Bimanual hand pose NPZ exported from HaWoR.",
    )
    parser.add_argument(
        "--name",
        default="video_0_hawor_world",
        help="Artifact name prefix.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs/presets/hawor_world_tuned"),
        help="OpenArm pipeline config directory.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Project root for generated artifacts.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render a MuJoCo replay video.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optionally process only the first N frames.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pose = load_npz(args.pose)
    if args.max_frames is not None:
        if args.max_frames <= 0:
            raise ValueError("max-frames must be positive")
        pose = {
            key: value[: args.max_frames]
            if getattr(value, "ndim", 0) > 0 and len(value) >= args.max_frames
            else value
            for key, value in pose.items()
        }
    artifacts, quality = run_bimanual_from_pose(
        pose,
        name=args.name,
        root=args.root,
        config_dir=args.config_dir,
        source="hawor_world",
        render=args.render,
    )
    print(
        json.dumps(
            {
                "name": args.name,
                "quality_report": str(artifacts.quality_report),
                "replay_video": (
                    str(artifacts.replay_video)
                    if artifacts.replay_video is not None
                    else None
                ),
                "mean_ik_error_m": quality["mean_ik_error_m"],
                "max_ik_error_m": quality["max_ik_error_m"],
                "ik_converged_ratio": quality["ik_converged_ratio"],
                "left_mean_ik_error_m": quality["left"]["mean_ik_error_m"],
                "right_mean_ik_error_m": quality["right"]["mean_ik_error_m"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
