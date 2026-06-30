"""Check WiLoR full-3D assets, imports, CUDA, and model loading."""
from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wilor-root",
        type=Path,
        default=Path("external_repos/WiLoR"),
    )
    parser.add_argument(
        "--load-model",
        action="store_true",
        help="Load the 2.4 GB WiLoR checkpoint onto the GPU.",
    )
    args = parser.parse_args()
    root = args.wilor_root.resolve()

    required = {
        "WiLoR root": root,
        "model checkpoint": root / "pretrained_models/wilor_final.ckpt",
        "model config": root / "pretrained_models/model_config.yaml",
        "detector": root / "pretrained_models/detector.pt",
        "MANO right": root / "mano_data/MANO_RIGHT.pkl",
    }
    ready = True
    for name, path in required.items():
        exists = path.is_dir() if name == "WiLoR root" else path.is_file()
        print(f"[{'OK' if exists else 'MISS'}] {name}: {path}")
        ready &= exists

    import cv2
    import smplx
    import torch
    import ultralytics

    print(f"[OK] torch: {torch.__version__}")
    print(f"[OK] CUDA runtime: {torch.version.cuda}")
    print(f"[OK] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[OK] OpenCV: {cv2.__version__}")
    print(f"[OK] Ultralytics: {ultralytics.__version__}")
    print(f"[OK] smplx: {version('smplx')}")

    if args.load_model and ready:
        import os
        import sys

        os.chdir(root)
        sys.path.insert(0, str(root))
        from wilor.models import load_wilor

        model, _ = load_wilor(
            checkpoint_path="./pretrained_models/wilor_final.ckpt",
            cfg_path="./pretrained_models/model_config.yaml",
        )
        device = torch.device("cuda")
        model = model.to(device).eval()
        allocated = torch.cuda.memory_allocated() / (1024**3)
        print(f"[OK] WiLoR model loaded on GPU: {allocated:.2f} GiB allocated")

    print(f"WiLoR full 3D ready: {ready and torch.cuda.is_available()}")
    return 0 if ready and torch.cuda.is_available() else 1


if __name__ == "__main__":
    raise SystemExit(main())
