from __future__ import annotations

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("openarm_mujoco")

from openarm_retarget.openarm_model import load_openarm, model_report


def test_openarm_v2_model_is_discovered_and_inspected():
    model, info = load_openarm(
        {
            "model_asset": "cell.xml",
            "side": "left",
            "home_keyframe": "home",
            "ee_site": "auto",
            "arm_joint_names": "auto",
            "gripper_actuator": "auto",
        }
    )

    assert model.nq == 19
    assert len(info.arm_joint_names) == 7
    assert info.ee_site == "left_ee_control_point"
    assert info.gripper_actuator == "left_finger1_ctrl"
    report = model_report(model, info)
    assert "ee_home_position" in report
    assert "left_joint7_ctrl" in report

