from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("openarm_mujoco")
pytest.importorskip("matplotlib")

from openarm_retarget.pipeline import run_bimanual_synthetic


def test_bimanual_pipeline_creates_both_arm_trajectory(tmp_path):
    config_dir = Path(__file__).parents[1] / "configs"
    artifacts, quality = run_bimanual_synthetic(
        name="two_hands",
        frames=36,
        root=tmp_path,
        config_dir=config_dir,
        render=False,
    )

    for path in (
        artifacts.hand_pose,
        artifacts.pinch,
        artifacts.smooth,
        artifacts.target,
        artifacts.trajectory,
        artifacts.dataset,
        artifacts.left_pinch_plot,
        artifacts.right_pinch_plot,
        artifacts.left_ik_plot,
        artifacts.right_ik_plot,
        artifacts.quality_report,
    ):
        assert path.is_file()
    assert quality["mode"] == "bimanual"
    assert quality["left"]["max_ik_error_m"] < 0.05
    assert quality["right"]["max_ik_error_m"] < 0.05
    saved = json.loads(artifacts.quality_report.read_text(encoding="utf-8"))
    assert saved["left"]["ee_site"] == "left_ee_control_point"
    assert saved["right"]["ee_site"] == "right_ee_control_point"
