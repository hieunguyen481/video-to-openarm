from __future__ import annotations

import argparse
from pathlib import Path

from openarm_retarget.config import load_config
from openarm_retarget.openarm_model import load_openarm, model_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect OpenArm MuJoCo model")
    parser.add_argument("--config", type=Path, default=Path("configs/openarm.yaml"))
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/openarm_model_report.txt")
    )
    args = parser.parse_args()

    model, info = load_openarm(load_config(args.config))
    report = model_report(model, info)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

