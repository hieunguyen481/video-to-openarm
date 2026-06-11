from __future__ import annotations

import argparse
from pathlib import Path

from openarm_retarget.io import save_npz
from openarm_retarget.synthetic import (
    generate_bimanual_hand_pose,
    generate_hand_pose,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic hand pose data")
    parser.add_argument("--output", type=Path, default=Path("data/hand_pose/synthetic_hand_pose.npz"))
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--single-hand",
        action="store_true",
        help="Generate the legacy single-hand schema",
    )
    args = parser.parse_args()
    data = (
        generate_hand_pose(frames=args.frames, fps=args.fps)
        if args.single_hand
        else generate_bimanual_hand_pose(frames=args.frames, fps=args.fps)
    )
    save_npz(
        args.output,
        data,
        stage="hand_pose" if args.single_hand else "bimanual_hand_pose",
        metadata={
            "source": "synthetic",
            "fps": args.fps,
            "mode": "single" if args.single_hand else "bimanual",
        },
    )
    print(f"Saved synthetic hand pose to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
