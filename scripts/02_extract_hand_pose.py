from __future__ import annotations

import argparse
from pathlib import Path

from openarm_retarget.config import load_config
from openarm_retarget.hand_tracking import extract_video_bimanual_hand_pose
from openarm_retarget.io import save_npz


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract hand landmarks from a video")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/hand_tracking.yaml"))
    parser.add_argument("--debug-video", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    result = extract_video_bimanual_hand_pose(
        args.video, config, debug_video=args.debug_video
    )
    save_npz(
        args.output,
        result.data,
        stage="bimanual_hand_pose",
        metadata={
            "source_video": str(args.video),
            "fps": result.fps,
            "frame_size": result.frame_size,
        },
    )
    left_valid = float(result.data["left_valid"].mean())
    right_valid = float(result.data["right_valid"].mean())
    print(f"Saved {len(result.data['timestamps'])} frames to {args.output}")
    print(f"Valid hand tracking: left={left_valid:.1%}, right={right_valid:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
