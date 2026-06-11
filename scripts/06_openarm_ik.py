from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from openarm_retarget.config import load_config
from openarm_retarget.ik_solver import JacobianIKSolver
from openarm_retarget.io import load_npz, save_npz
from openarm_retarget.openarm_model import load_openarm
from openarm_retarget.plots import plot_ik_error


def main() -> int:
    parser = argparse.ArgumentParser(description="Solve OpenArm position IK")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--openarm-config", type=Path, default=Path("configs/openarm.yaml"))
    parser.add_argument("--ik-config", type=Path, default=Path("configs/ik.yaml"))
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()

    target_data = load_npz(args.input)
    openarm_config = load_config(args.openarm_config)
    ik_config = load_config(args.ik_config)
    if ik_config.get("solver", "jacobian_dls") != "jacobian_dls":
        raise ValueError("Only solver=jacobian_dls is currently supported")
    model, info = load_openarm(openarm_config)
    result = JacobianIKSolver(model, info, ik_config).solve(target_data["target_pos"])
    payload = {
        "timestamps": target_data["timestamps"],
        "qpos": result.qpos,
        "arm_qpos": result.arm_qpos,
        "ee_pos": result.ee_pos,
        "target_pos": result.target_pos,
        "ik_error": result.ik_error,
        "ik_converged": result.converged,
        "ik_iterations": result.iterations,
        "gripper_cmd": target_data["gripper_cmd"],
    }
    save_npz(
        args.output,
        payload,
        stage="openarm_ik",
        metadata={
            "source": str(args.input),
            "model_path": str(info.model_path),
            "ee_site": info.ee_site,
            "arm_joint_names": info.arm_joint_names,
            "ik_config": ik_config,
        },
    )
    if args.plot:
        plot_ik_error(
            target_data["timestamps"],
            result.ik_error,
            args.plot,
            tolerance=float(ik_config["tolerance"]),
        )
    print(f"Saved IK trajectory to {args.output}")
    print(
        f"Mean error={np.mean(result.ik_error) * 100:.2f} cm; "
        f"max={np.max(result.ik_error) * 100:.2f} cm; "
        f"converged={np.mean(result.converged):.1%}"
    )
    return 0 if np.max(result.ik_error) < 0.20 else 2


if __name__ == "__main__":
    raise SystemExit(main())

