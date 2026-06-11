from __future__ import annotations

import argparse
from pathlib import Path

from openarm_retarget.dataset import build_dataset
from openarm_retarget.io import load_npz, save_npz


def main() -> int:
    parser = argparse.ArgumentParser(description="Build imitation-learning NPZ dataset")
    parser.add_argument("--smooth", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--traj", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    dataset = build_dataset(
        load_npz(args.smooth),
        load_npz(args.target),
        load_npz(args.traj),
    )
    save_npz(
        args.output,
        dataset,
        stage="dataset",
        metadata={
            "smooth_source": str(args.smooth),
            "target_source": str(args.target),
            "trajectory_source": str(args.traj),
        },
    )
    print(f"Saved {len(dataset['timestamps'])} samples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

