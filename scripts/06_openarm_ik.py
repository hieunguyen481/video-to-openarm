from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from openarm_retarget.config import load_config
from openarm_retarget.ik_solver import (
    BimanualJacobianIKSolver,
    JacobianIKSolver,
)
from openarm_retarget.io import load_npz, save_npz
from openarm_retarget.openarm_model import load_bimanual_openarm, load_openarm
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
    if "left_target_pos" in target_data:
        model, info = load_bimanual_openarm(openarm_config)
        result = BimanualJacobianIKSolver(model, info, ik_config).solve(
            target_data["left_target_pos"],
            target_data["right_target_pos"],
        )
        payload = {
            "timestamps": target_data["timestamps"],
            "qpos": result.qpos,
            "left_arm_qpos": result.left_arm_qpos,
            "right_arm_qpos": result.right_arm_qpos,
            "left_ee_pos": result.left_ee_pos,
            "right_ee_pos": result.right_ee_pos,
            "left_target_pos": result.left_target_pos,
            "right_target_pos": result.right_target_pos,
            "left_ik_error": result.left_ik_error,
            "right_ik_error": result.right_ik_error,
            "ik_converged": result.converged,
            "ik_iterations": result.iterations,
            "left_gripper_cmd": target_data["left_gripper_cmd"],
            "right_gripper_cmd": target_data["right_gripper_cmd"],
        }
        metadata = {
            "source": str(args.input),
            "model_path": str(info.model_path),
            "left_ee_site": info.sides["left"].ee_site,
            "right_ee_site": info.sides["right"].ee_site,
            "ik_config": ik_config,
        }
        if args.plot:
            left_plot = args.plot.with_name(
                f"{args.plot.stem}_left{args.plot.suffix}"
            )
            right_plot = args.plot.with_name(
                f"{args.plot.stem}_right{args.plot.suffix}"
            )
            plot_ik_error(
                target_data["timestamps"],
                result.left_ik_error,
                left_plot,
                tolerance=float(ik_config["tolerance"]),
            )
            plot_ik_error(
                target_data["timestamps"],
                result.right_ik_error,
                right_plot,
                tolerance=float(ik_config["tolerance"]),
            )
        all_errors = np.concatenate(
            (result.left_ik_error, result.right_ik_error)
        )
    else:
        model, info = load_openarm(openarm_config)
        result = JacobianIKSolver(model, info, ik_config).solve(
            target_data["target_pos"]
        )
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
        metadata = {
            "source": str(args.input),
            "model_path": str(info.model_path),
            "ee_site": info.ee_site,
            "arm_joint_names": info.arm_joint_names,
            "ik_config": ik_config,
        }
        if args.plot:
            plot_ik_error(
                target_data["timestamps"],
                result.ik_error,
                args.plot,
                tolerance=float(ik_config["tolerance"]),
            )
        all_errors = result.ik_error
    save_npz(
        args.output,
        payload,
        stage="bimanual_openarm_ik" if "left_target_pos" in target_data else "openarm_ik",
        metadata=metadata,
    )
    print(f"Saved IK trajectory to {args.output}")
    print(
        f"Mean error={np.mean(all_errors) * 100:.2f} cm; "
        f"max={np.max(all_errors) * 100:.2f} cm; "
        f"converged={np.mean(result.converged):.1%}"
    )
    return 0 if np.max(all_errors) < 0.20 else 2


if __name__ == "__main__":
    raise SystemExit(main())
