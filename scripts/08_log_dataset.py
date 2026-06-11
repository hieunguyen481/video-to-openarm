from __future__ import annotations

import argparse
from pathlib import Path

from openarm_retarget.dataset import build_bimanual_dataset, build_dataset
from openarm_retarget.io import load_npz, save_npz


def main() -> int:
    parser = argparse.ArgumentParser(description="Build imitation-learning NPZ dataset")
    parser.add_argument("--smooth", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--traj", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    smooth = load_npz(args.smooth)
    target = load_npz(args.target)
    trajectory = load_npz(args.traj)
    is_bimanual = "left_wrist_smooth" in smooth
    dataset = (
        build_bimanual_dataset(smooth, target, trajectory)
        if is_bimanual
        else build_dataset(smooth, target, trajectory)
    )
    save_npz(
        args.output,
        dataset,
        stage="bimanual_dataset" if is_bimanual else "dataset",
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
