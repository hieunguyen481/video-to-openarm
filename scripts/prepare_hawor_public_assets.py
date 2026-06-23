"""Download public HaWoR assets that do not require a license-gated account.

This prepares the external HaWoR checkout as far as possible automatically:

- HaWoR checkpoint, infiller, and model config from Hugging Face.
- DROID-SLAM and Metric3D weights from Google Drive.
- WiLoR detector copied from the existing WiLoR checkout when available.
- ffmpeg.exe copied from imageio-ffmpeg when available.

MANO files are not downloaded here because they require accepting the MANO
license on the official website.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


HF_FILES = (
    ("hawor/checkpoints/hawor.ckpt", "weights/hawor/checkpoints/hawor.ckpt"),
    ("hawor/checkpoints/infiller.pt", "weights/hawor/checkpoints/infiller.pt"),
    ("hawor/model_config.yaml", "weights/hawor/model_config.yaml"),
)

GDRIVE_FILES = (
    (
        "1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh",
        "weights/external/droid.pth",
    ),
    (
        "1eT2gG-kwsVzNy5nJrbm4KC-9DbNKyLnr",
        "thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth",
    ),
)


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def download_huggingface(root: Path, force: bool) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "Install huggingface_hub before downloading HaWoR model files."
        ) from exc

    for filename, relative_output in HF_FILES:
        output = root / relative_output
        if output.is_file() and not force:
            print(f"exists {output}")
            continue
        ensure_parent(output)
        print(f"downloading {filename}")
        hf_hub_download(
            repo_id="ThunderVVV/HaWoR",
            filename=filename,
            local_dir=root / "weights",
        )


def download_gdrive(root: Path, force: bool) -> None:
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "Install gdown before downloading DROID-SLAM/Metric3D weights."
        ) from exc

    for file_id, relative_output in GDRIVE_FILES:
        output = root / relative_output
        if output.is_file() and not force:
            print(f"exists {output}")
            continue
        ensure_parent(output)
        print(f"downloading Google Drive file {file_id}")
        result = gdown.download(id=file_id, output=str(output), quiet=False)
        if result is None:
            raise RuntimeError(f"Failed to download Google Drive file {file_id}")


def copy_detector(root: Path, detector: Path, force: bool) -> None:
    output = root / "weights" / "external" / "detector.pt"
    if output.is_file() and not force:
        print(f"exists {output}")
        return
    if not detector.is_file():
        print(f"skip detector copy, missing {detector}")
        return
    ensure_parent(output)
    shutil.copy2(detector, output)
    print(f"copied {detector} -> {output}")


def copy_ffmpeg(root: Path, force: bool) -> None:
    output = root / "ffmpeg.exe"
    if output.is_file() and not force:
        print(f"exists {output}")
        return
    try:
        import imageio_ffmpeg
    except ImportError:
        print("skip ffmpeg copy, imageio-ffmpeg is not installed")
        return
    source = Path(imageio_ffmpeg.get_ffmpeg_exe())
    shutil.copy2(source, output)
    print(f"copied {source} -> {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare public HaWoR assets.")
    parser.add_argument(
        "--hawor-root",
        type=Path,
        default=Path("external_repos/HaWoR"),
        help="Path to the external HaWoR checkout.",
    )
    parser.add_argument(
        "--wilor-detector",
        type=Path,
        default=Path("external_repos/WiLoR/pretrained_models/detector.pt"),
        help="Existing WiLoR detector to copy into HaWoR.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-hf", action="store_true")
    parser.add_argument("--skip-gdrive", action="store_true")
    parser.add_argument("--skip-detector", action="store_true")
    parser.add_argument("--skip-ffmpeg", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.hawor_root.is_dir():
        raise FileNotFoundError(f"HaWoR root not found: {args.hawor_root}")

    if not args.skip_hf:
        download_huggingface(args.hawor_root, args.force)
    if not args.skip_gdrive:
        download_gdrive(args.hawor_root, args.force)
    if not args.skip_detector:
        copy_detector(args.hawor_root, args.wilor_detector, args.force)
    if not args.skip_ffmpeg:
        copy_ffmpeg(args.hawor_root, args.force)

    print("\nNext checks:")
    print(f"{sys.executable} scripts/check_hawor_setup.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
