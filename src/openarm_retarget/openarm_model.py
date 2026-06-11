from __future__ import annotations

import importlib.metadata
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _mujoco():
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError(
            'MuJoCo is required. Install with: python -m pip install -e ".[simulation]"'
        ) from exc
    return mujoco


@dataclass(frozen=True)
class OpenArmModelInfo:
    model_path: Path
    ee_site: str
    arm_joint_names: tuple[str, ...]
    arm_actuator_names: tuple[str, ...]
    gripper_actuator: str
    home_keyframe: str | None


def resolve_model_path(config: Mapping[str, Any]) -> Path:
    explicit = config.get("model_path") or os.environ.get("OPENARM_MJCF")
    if explicit:
        path = Path(str(explicit)).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"OpenArm MJCF not found: {path}")
        return path

    asset = str(config.get("model_asset", "cell.xml"))
    try:
        distribution = importlib.metadata.distribution("openarm-mujoco")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "openarm-mujoco is not installed and model_path was not provided"
        ) from exc

    matches = [
        Path(distribution.locate_file(item)).resolve()
        for item in distribution.files or []
        if str(item).replace("\\", "/").endswith(f"/{asset}")
        or str(item) == asset
    ]
    matches = [path for path in matches if path.is_file()]
    if not matches:
        raise FileNotFoundError(
            f"Could not find {asset!r} in openarm-mujoco distribution"
        )
    return matches[0]


def object_names(model: Any, object_type: Any, count: int) -> list[str]:
    mujoco = _mujoco()
    return [
        name
        for index in range(count)
        if (name := mujoco.mj_id2name(model, object_type, index)) is not None
    ]


def _name_id(model: Any, object_type: Any, name: str) -> int:
    mujoco = _mujoco()
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"Model does not contain {name!r}")
    return int(object_id)


def _auto_joint_names(model: Any, side: str) -> tuple[str, ...]:
    mujoco = _mujoco()
    pattern = re.compile(rf"openarm_{re.escape(side)}_joint(\d+)$")
    matches: list[tuple[int, str]] = []
    for name in object_names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt):
        match = pattern.fullmatch(name)
        if match:
            matches.append((int(match.group(1)), name))
    matches.sort()
    if not matches:
        raise ValueError(f"No arm joints found for side={side!r}")
    return tuple(name for _, name in matches)


def _actuators_for_joints(model: Any, joint_names: tuple[str, ...]) -> tuple[str, ...]:
    mujoco = _mujoco()
    joint_ids = {
        _name_id(model, mujoco.mjtObj.mjOBJ_JOINT, name): name
        for name in joint_names
    }
    actuator_by_joint: dict[int, str] = {}
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        if joint_id in joint_ids and name:
            actuator_by_joint[joint_id] = name
    missing = [
        name
        for joint_id, name in joint_ids.items()
        if joint_id not in actuator_by_joint
    ]
    if missing:
        raise ValueError(f"No actuator found for joints: {', '.join(missing)}")
    return tuple(
        actuator_by_joint[
            _name_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        ]
        for joint_name in joint_names
    )


def load_openarm(
    config: Mapping[str, Any],
) -> tuple[Any, OpenArmModelInfo]:
    mujoco = _mujoco()
    path = resolve_model_path(config)
    model = mujoco.MjModel.from_xml_path(str(path))
    side = str(config.get("side", "left")).lower()
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")

    configured_joints = config.get("arm_joint_names", "auto")
    joint_names = (
        _auto_joint_names(model, side)
        if configured_joints is None or configured_joints == "auto"
        else tuple(configured_joints)
    )
    for name in joint_names:
        _name_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

    configured_site = config.get("ee_site", "auto")
    ee_site = (
        f"{side}_ee_control_point"
        if configured_site is None or configured_site == "auto"
        else str(configured_site)
    )
    _name_id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site)

    configured_gripper = config.get("gripper_actuator", "auto")
    gripper = (
        f"{side}_finger1_ctrl"
        if configured_gripper is None or configured_gripper == "auto"
        else str(configured_gripper)
    )
    _name_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, gripper)

    keyframe = config.get("home_keyframe")
    if keyframe:
        _name_id(model, mujoco.mjtObj.mjOBJ_KEY, str(keyframe))

    info = OpenArmModelInfo(
        model_path=path,
        ee_site=ee_site,
        arm_joint_names=joint_names,
        arm_actuator_names=_actuators_for_joints(model, joint_names),
        gripper_actuator=gripper,
        home_keyframe=str(keyframe) if keyframe else None,
    )
    return model, info


def reset_home(model: Any, data: Any, keyframe: str | None) -> None:
    mujoco = _mujoco()
    if keyframe:
        key_id = _name_id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)


def model_report(model: Any, info: OpenArmModelInfo) -> str:
    mujoco = _mujoco()
    data = mujoco.MjData(model)
    reset_home(model, data, info.home_keyframe)
    site_id = _name_id(model, mujoco.mjtObj.mjOBJ_SITE, info.ee_site)

    lines = [
        "OpenArm MuJoCo Model Report",
        "=" * 28,
        f"model_path: {info.model_path}",
        f"nq: {model.nq}",
        f"nv: {model.nv}",
        f"nu: {model.nu}",
        f"njnt: {model.njnt}",
        f"nsite: {model.nsite}",
        f"ee_site: {info.ee_site}",
        f"ee_home_position: {np.array2string(data.site_xpos[site_id], precision=6)}",
        f"arm_joint_names: {list(info.arm_joint_names)}",
        f"arm_actuator_names: {list(info.arm_actuator_names)}",
        f"gripper_actuator: {info.gripper_actuator}",
        f"home_keyframe: {info.home_keyframe}",
        "",
        "Joints:",
    ]
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        limited = bool(model.jnt_limited[joint_id])
        joint_range = model.jnt_range[joint_id].tolist() if limited else None
        lines.append(
            f"  [{joint_id:02d}] {name}: qpos={model.jnt_qposadr[joint_id]}, "
            f"dof={model.jnt_dofadr[joint_id]}, range={joint_range}"
        )
    lines.extend(["", "Actuators:"])
    for actuator_id in range(model.nu):
        name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id
        )
        lines.append(
            f"  [{actuator_id:02d}] {name}: "
            f"ctrlrange={model.actuator_ctrlrange[actuator_id].tolist()}"
        )
    return "\n".join(lines) + "\n"
