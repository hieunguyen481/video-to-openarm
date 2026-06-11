from __future__ import annotations

import argparse
from pathlib import Path

from openarm_retarget.camera_recorder import record_camera


def main() -> int:
    parser = argparse.ArgumentParser(description="Record a two-hand webcam video")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw_videos/demo_001.mp4"),
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--auto-start", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-mirror-preview", action="store_true")
    args = parser.parse_args()

    result = record_camera(
        args.output,
        camera=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        backend=args.backend,
        duration=args.duration,
        auto_start=args.auto_start,
        overwrite=args.overwrite,
        mirror_preview=not args.no_mirror_preview,
    )
    print(f"Saved {result.duration_seconds:.1f}s to {result.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
