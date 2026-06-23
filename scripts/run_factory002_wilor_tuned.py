"""Run the selected WiLoR YOLO tuned retargeting preset for factory002_middle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from openarm_retarget.comparison_video import create_comparison_video
from openarm_retarget.io import load_npz
from openarm_retarget.pipeline import run_bimanual_from_pose


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay factory002_middle with the tuned WiLoR YOLO preset."
    )
    parser.add_argument(
        "--pose",
        type=Path,
        default=Path(
            "outputs/external_trials/wilor_yolo/"
            "factory002_middle_wilor_yolo_hand_pose.npz"
        ),
        help="Bimanual hand pose NPZ exported from WiLoR YOLO.",
    )
    parser.add_argument(
        "--debug-video",
        type=Path,
        default=Path(
            "outputs/external_trials/wilor_yolo/"
            "factory002_middle_wilor_yolo_hand_debug.mp4"
        ),
        help="Human/debug video to place beside the robot replay.",
    )
    parser.add_argument(
        "--name",
        default="factory002_middle_wilor_yolo_tuned",
        help="Artifact name prefix.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs/presets/wilor_yolo_tuned"),
        help="Preset config directory.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Project root for generated artifacts.",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Skip MuJoCo video rendering.",
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=Path(
            "outputs/external_trials/wilor_yolo_tuned/"
            "factory002_middle_wilor_yolo_tuned_human_vs_robot.mp4"
        ),
        help="Output path for the side-by-side human/robot video.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pose = load_npz(args.pose)
    artifacts, quality = run_bimanual_from_pose(
        pose,
        name=args.name,
        root=args.root,
        config_dir=args.config_dir,
        source="wilor_yolo_tuned",
        render=not args.no_render,
    )

    comparison_output = None
    if artifacts.replay_video is not None and args.debug_video.is_file():
        comparison = create_comparison_video(
            args.debug_video,
            artifacts.replay_video,
            args.comparison_output,
            panel_width=960,
            panel_height=720,
            overwrite=True,
        )
        comparison_output = str(comparison.output)

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
                "comparison_video": comparison_output,
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
