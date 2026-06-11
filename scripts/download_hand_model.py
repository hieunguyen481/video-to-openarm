from __future__ import annotations

import argparse
import shutil
import tempfile
import urllib.request
from pathlib import Path

DEFAULT_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the official MediaPipe hand model")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/hand_landmarker.task"),
    )
    parser.add_argument("--url", default=DEFAULT_URL)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=args.output.parent,
        prefix=f".{args.output.stem}.",
        delete=False,
        suffix=".task",
    ) as handle:
        temporary = Path(handle.name)
    try:
        print(f"Downloading {args.url}")
        with urllib.request.urlopen(args.url, timeout=120) as response:
            with temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
        if temporary.stat().st_size < 1_000_000:
            raise RuntimeError("Downloaded file is unexpectedly small")
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Saved model to {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
