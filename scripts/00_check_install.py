from __future__ import annotations

import importlib
import platform
import sys


REQUIRED = {
    "numpy": "Core array operations",
    "yaml": "YAML configuration",
}

OPTIONAL = {
    "cv2": "Video input/output",
    "mediapipe": "Hand landmark detection",
    "matplotlib": "Diagnostic plots",
    "mujoco": "OpenArm simulation and IK",
    "openarm_mujoco": "OpenArm MuJoCo model",
    "sklearn": "Baseline policy training",
}


def inspect_module(name: str, purpose: str, required: bool) -> bool:
    try:
        module = importlib.import_module(name)
    except ImportError:
        level = "ERROR" if required else "WARN"
        print(f"[{level}] {name:<18} missing ({purpose})")
        return not required
    version = getattr(module, "__version__", "installed")
    print(f"[OK]    {name:<18} {version}")
    return True


def main() -> int:
    print(f"[OK]    Python             {platform.python_version()}")
    if sys.version_info < (3, 10):
        print("[ERROR] Python >= 3.10 is required")
        return 1

    success = all(
        inspect_module(name, purpose, required=True)
        for name, purpose in REQUIRED.items()
    )
    for name, purpose in OPTIONAL.items():
        inspect_module(name, purpose, required=False)

    print("\nCore environment is ready." if success else "\nCore environment is incomplete.")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())

