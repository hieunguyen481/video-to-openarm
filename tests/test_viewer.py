from __future__ import annotations

import sys

import pytest

pytest.importorskip("openarm_mujoco")

from openarm_retarget.viewer import build_viewer_command


def test_viewer_command_uses_project_model_and_flags():
    command = build_viewer_command(
        {"model_asset": "cell.xml"},
        keyframe="home",
        static=True,
        walls=True,
        no_sheet=True,
    )

    assert command[:3] == [sys.executable, "-m", "openarm_mujoco.v2.launch"]
    assert command[3].endswith("cell.xml")
    assert command[-3:] == ["--static", "--walls", "--no-sheet"]

