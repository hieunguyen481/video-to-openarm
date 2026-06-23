"""Check whether the local HaWoR checkout is ready to run.

HaWoR is intentionally kept as an optional external workflow because it needs
large weights, MANO assets, CUDA-oriented dependencies, and third-party SLAM
components. This checker reports missing pieces without importing HaWoR itself.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True


REQUIRED_FILES = (
    (
        "HaWoR demo script",
        "demo.py",
        "Clone ThunderVVV/HaWoR under external_repos/HaWoR.",
    ),
    (
        "HaWoR checkpoint",
        "weights/hawor/checkpoints/hawor.ckpt",
        "Download from Hugging Face ThunderVVV/HaWoR.",
    ),
    (
        "HaWoR infiller",
        "weights/hawor/checkpoints/infiller.pt",
        "Download from Hugging Face ThunderVVV/HaWoR.",
    ),
    (
        "HaWoR model config",
        "weights/hawor/model_config.yaml",
        "Download from Hugging Face ThunderVVV/HaWoR.",
    ),
    (
        "WiLoR detector for HaWoR",
        "weights/external/detector.pt",
        "Download WiLoR detector.pt or copy the existing external WiLoR weight.",
    ),
    (
        "DROID-SLAM weights",
        "weights/external/droid.pth",
        "Download the official DROID-SLAM droid.pth.",
    ),
    (
        "Metric3D weights",
        "thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth",
        "Download Metric3D official metric_depth_vit_large_800k.pth.",
    ),
    (
        "MANO right model",
        "_DATA/data/mano/MANO_RIGHT.pkl",
        "Download MANO from mano.is.tue.mpg.de and place MANO_RIGHT.pkl here.",
    ),
    (
        "MANO left model",
        "_DATA/data_left/mano_left/MANO_LEFT.pkl",
        "Download MANO from mano.is.tue.mpg.de and place MANO_LEFT.pkl here.",
    ),
)

PYTHON_MODULES = (
    ("torch", True),
    ("cv2", True),
    ("ultralytics", True),
    ("smplx", True),
    ("yacs", True),
    ("joblib", True),
    ("natsort", True),
    ("pycocotools", True),
    ("skimage", True),
    ("trimesh", True),
    ("easydict", True),
    ("loguru", True),
    ("mmengine", True),
    ("pytorch3d", True),
    ("aitviewer", True),
)


def check_file(root: Path, name: str, relative: str, hint: str) -> Check:
    path = root / relative
    return Check(
        name=name,
        ok=path.is_file(),
        detail=str(path) if path.is_file() else f"Missing {path}. {hint}",
    )


def check_module(name: str, required: bool) -> Check:
    ok = importlib.util.find_spec(name) is not None
    return Check(
        name=f"Python module: {name}",
        ok=ok,
        detail="importable" if ok else f"Not importable: {name}",
        required=required,
    )


def check_ffmpeg(root: Path) -> Check:
    exe = shutil.which("ffmpeg")
    local_exe = root / "ffmpeg.exe"
    if exe is None and local_exe.is_file():
        exe = str(local_exe)
    return Check(
        name="ffmpeg",
        ok=exe is not None,
        detail=exe if exe is not None else (
            "ffmpeg is not on PATH. Install ffmpeg or place ffmpeg.exe in "
            "the HaWoR root."
        ),
    )


def check_tool(name: str, required: bool = False) -> Check:
    exe = shutil.which(name)
    return Check(
        name=f"Build tool: {name}",
        ok=exe is not None,
        detail=exe if exe is not None else f"{name} is not on PATH.",
        required=required,
    )


def check_cuda() -> Check:
    try:
        import torch
    except ImportError:
        return Check("CUDA availability", False, "torch is not importable.", False)
    available = bool(torch.cuda.is_available())
    detail = "torch.cuda.is_available() is True" if available else (
        "CUDA is not available. Some HaWoR stages may be very slow or fail."
    )
    return Check("CUDA availability", available, detail, False)


def build_demo_command(root: Path, video_arg: Path, vis_mode: str) -> str:
    return (
        f"cd {root} && python demo.py --video_path {video_arg} "
        f"--vis_mode {vis_mode}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check local HaWoR readiness.")
    parser.add_argument(
        "--hawor-root",
        type=Path,
        default=Path("external_repos/HaWoR"),
        help="Path to the external HaWoR checkout.",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=Path("example/video_0.mp4"),
        help="Video path relative to HaWoR root for the suggested demo command.",
    )
    parser.add_argument(
        "--vis-mode",
        choices=("world", "cam"),
        default="world",
        help="HaWoR visualization mode for the suggested command.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON only.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.hawor_root
    checks: list[Check] = []
    checks.append(
        Check(
            "HaWoR root",
            root.is_dir(),
            str(root) if root.is_dir() else f"Missing directory: {root}",
        )
    )
    for name, relative, hint in REQUIRED_FILES:
        checks.append(check_file(root, name, relative, hint))
    checks.extend(check_module(name, required) for name, required in PYTHON_MODULES)
    checks.append(check_ffmpeg(root))
    checks.append(check_tool("cl"))
    checks.append(check_tool("nvcc"))
    checks.append(check_cuda())

    required_ok = all(item.ok for item in checks if item.required)
    video = args.video if args.video.is_absolute() else root / args.video
    result = {
        "ready": required_ok,
        "hawor_root": str(root),
        "checks": [asdict(item) for item in checks],
        "suggested_command": build_demo_command(root, args.video, args.vis_mode),
        "expected_outputs": [
            str(video.with_suffix("") / "world_space_res.pth"),
            str(video.with_suffix("") / "SLAM"),
            str(video.with_suffix("") / "vis_0_*"),
        ],
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"HaWoR ready: {required_ok}")
        for item in checks:
            marker = "OK" if item.ok else "MISS"
            optional = " optional" if not item.required else ""
            print(f"[{marker}]{optional} {item.name}: {item.detail}")
        print("\nSuggested command when ready:")
        print(result["suggested_command"])
        print("\nExpected outputs:")
        for output in result["expected_outputs"]:
            print(f"- {output}")
    return 0 if required_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
