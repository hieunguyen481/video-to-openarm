from __future__ import annotations

import argparse
import json
from pathlib import Path

from openarm_retarget.baseline import train_baseline
from openarm_retarget.io import load_npz


def main() -> int:
    parser = argparse.ArgumentParser(description="Train baseline arm/gripper policy")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("models/openarm_baseline.joblib")
    )
    parser.add_argument("--test-fraction", type=float, default=0.2)
    args = parser.parse_args()
    metrics = train_baseline(
        load_npz(args.dataset),
        output=args.output,
        test_fraction=args.test_fraction,
    )
    print(json.dumps(metrics, indent=2))
    print(f"Saved baseline model to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

