#!/usr/bin/env python3
"""Download EgoWorld episodes from HuggingFace and save as pipeline-ready NPZ.

Usage
-----
# List available episodes
python scripts/download_egoworld.py --list

# Download first 3 episodes
python scripts/download_egoworld.py --episodes 0,1,2

# Download and save to custom directory
python scripts/download_egoworld.py --episodes 0 --output data/egoworld/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download EgoWorld episodes from HuggingFace",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available episodes and exit",
    )
    parser.add_argument(
        "--episodes",
        type=str,
        default="0",
        help="Comma-separated episode indices to download (default: 0)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/egoworld"),
        help="Output directory for converted NPZ files",
    )
    parser.add_argument(
        "--repo-id",
        default="haoyang-li/EgoWorld",
        help="HuggingFace dataset repository ID",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="HuggingFace cache directory",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Assumed FPS for timestamps",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Keep world-frame coordinates instead of normalizing to [0,1]",
    )
    args = parser.parse_args()

    from openarm_retarget.lerobot_loader import (
        download_and_convert_episode,
        list_available_episodes,
    )

    if args.list:
        print(f"Querying episodes from {args.repo_id}...")
        episodes = list_available_episodes(
            args.repo_id, cache_dir=args.cache_dir,
        )
        print(f"Found {len(episodes)} episode(s):")
        for ep in episodes[:50]:
            print(f"  {json.dumps(ep, ensure_ascii=False)}")
        if len(episodes) > 50:
            print(f"  ... and {len(episodes) - 50} more")
        return 0

    indices = [int(x.strip()) for x in args.episodes.split(",")]
    print(f"Downloading {len(indices)} episode(s) from {args.repo_id}...")
    for idx in indices:
        print(f"\n--- Episode {idx} ---")
        try:
            save_path, pose = download_and_convert_episode(
                idx,
                repo_id=args.repo_id,
                output_dir=args.output,
                cache_dir=args.cache_dir,
                fps=args.fps,
                normalize=not args.no_normalize,
            )
            num_frames = len(pose["timestamps"])
            left_valid = pose["left_valid"].sum()
            right_valid = pose["right_valid"].sum()
            print(f"  Saved: {save_path}")
            print(f"  Frames: {num_frames}")
            print(
                f"  Valid: left={left_valid}/{num_frames} "
                f"({100*left_valid/num_frames:.1f}%), "
                f"right={right_valid}/{num_frames} "
                f"({100*right_valid/num_frames:.1f}%)"
            )
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            continue

    print(f"\nDone. Output directory: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
