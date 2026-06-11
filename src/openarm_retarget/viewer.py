from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from .openarm_model import resolve_model_path


def build_viewer_command(
    config: Mapping[str, Any],
    *,
    keyframe: str = "home",
    static: bool = False,
    walls: bool = False,
    no_sheet: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "openarm_mujoco.v2.launch",
        str(resolve_model_path(config)),
        "--keyframe",
        keyframe,
    ]
    if static:
        command.append("--static")
    if walls:
        command.append("--walls")
    if no_sheet:
        command.append("--no-sheet")
    return command


def launch_viewer(
    config: Mapping[str, Any],
    *,
    keyframe: str = "home",
    static: bool = False,
    walls: bool = False,
    no_sheet: bool = False,
) -> int:
    command = build_viewer_command(
        config,
        keyframe=keyframe,
        static=static,
        walls=walls,
        no_sheet=no_sheet,
    )
    return subprocess.run(command, check=False).returncode

