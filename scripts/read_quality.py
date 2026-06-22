"""Read and display bimanual quality reports."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    files = [Path(item) for item in sys.argv[1:]]
    if not files:
        files = sorted(Path("outputs").glob("*_bimanual_quality_report.json"))

    for path in files:
        try:
            with path.open(encoding="utf-8") as handle:
                quality = json.load(handle)
            name = quality.get("name", "?")
            mean_err = quality.get("mean_ik_error_m", -1)
            max_err = quality.get("max_ik_error_m", -1)
            conv = quality.get("ik_converged_ratio", -1)
            left_err = quality.get("left", {}).get("mean_ik_error_m", -1)
            right_err = quality.get("right", {}).get("mean_ik_error_m", -1)
            print(
                f"{name}: mean={mean_err:.4f}m max={max_err:.4f}m "
                f"conv={conv:.1%} L={left_err:.4f}m R={right_err:.4f}m"
            )
        except Exception as exc:
            print(f"ERROR reading {path}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
