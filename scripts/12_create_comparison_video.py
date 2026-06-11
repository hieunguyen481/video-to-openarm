from __future__ import annotations

import argparse
from pathlib import Path

from openarm_retarget.comparison_video import create_comparison_video


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a side-by-side comparison")
    parser.add_argument("--human", required=True, type=Path)
    parser.add_argument("--robot", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--panel-width", type=int, default=960)
    parser.add_argument("--panel-height", type=int, default=720)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = create_comparison_video(
        args.human,
        args.robot,
        args.output,
        panel_width=args.panel_width,
        panel_height=args.panel_height,
        overwrite=args.overwrite,
    )
    print(f"Saved {result.frames} synchronized frames to {result.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

