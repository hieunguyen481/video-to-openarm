from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .openarm_model import OpenArmModelInfo, reset_home


def _mujoco():
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError("MuJoCo is required for IK") from exc
    return mujoco


@dataclass(frozen=True)
class IKResult:
    qpos: np.ndarray
    arm_qpos: np.ndarray
    ee_pos: np.ndarray
    target_pos: np.ndarray
    ik_error: np.ndarray
    converged: np.ndarray
    iterations: np.ndarray


class JacobianIKSolver:
    def __init__(
        self,
        model: Any,
        info: OpenArmModelInfo,
        config: Mapping[str, Any],
    ) -> None:
        mujoco = _mujoco()
        self.model = model
        self.info = info
        self.tolerance = float(config.get("tolerance", 0.02))
        self.max_iterations = int(config.get("max_iterations", 100))
        self.damping = float(config.get("damping", 0.01))
        self.step_size = float(config.get("step_size", 0.5))
        self.max_delta_q = float(config.get("max_delta_q", 0.05))
        self.max_frame_delta_q = float(config.get("max_frame_delta_q", 0.10))
        if min(
            self.tolerance,
            self.max_iterations,
            self.damping,
            self.step_size,
            self.max_delta_q,
            self.max_frame_delta_q,
        ) <= 0:
            raise ValueError("All IK tuning parameters must be positive")

        self.site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, info.ee_site
        )
        self.joint_ids = np.asarray(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in info.arm_joint_names
            ],
            dtype=int,
        )
        self.qpos_indices = model.jnt_qposadr[self.joint_ids].astype(int)
        self.dof_indices = model.jnt_dofadr[self.joint_ids].astype(int)

    def _clamp_joint_limits(self, qpos: np.ndarray) -> None:
        for joint_id, qpos_index in zip(
            self.joint_ids, self.qpos_indices, strict=True
        ):
            if self.model.jnt_limited[joint_id]:
                lower, upper = self.model.jnt_range[joint_id]
                qpos[qpos_index] = np.clip(qpos[qpos_index], lower, upper)

    def solve(self, target_pos: np.ndarray) -> IKResult:
        mujoco = _mujoco()
        targets = np.asarray(target_pos, dtype=float)
        if targets.ndim != 2 or targets.shape[1] != 3:
            raise ValueError(f"target_pos must have shape [T, 3], got {targets.shape}")
        if not np.all(np.isfinite(targets)):
            raise ValueError("target_pos contains NaN or infinite values")

        data = mujoco.MjData(self.model)
        reset_home(self.model, data, self.info.home_keyframe)
        frame_count = len(targets)
        qpos_output = np.empty((frame_count, self.model.nq), dtype=np.float64)
        ee_output = np.empty((frame_count, 3), dtype=np.float64)
        errors = np.empty(frame_count, dtype=np.float64)
        converged = np.zeros(frame_count, dtype=bool)
        iterations = np.zeros(frame_count, dtype=np.int32)
        jac_pos = np.zeros((3, self.model.nv), dtype=np.float64)
        previous_arm = data.qpos[self.qpos_indices].copy()

        for frame_index, target in enumerate(targets):
            best_qpos = data.qpos.copy()
            best_error = np.inf
            for iteration in range(1, self.max_iterations + 1):
                mujoco.mj_forward(self.model, data)
                delta = target - data.site_xpos[self.site_id]
                error = float(np.linalg.norm(delta))
                if error < best_error:
                    best_error = error
                    best_qpos = data.qpos.copy()
                if error <= self.tolerance:
                    converged[frame_index] = True
                    iterations[frame_index] = iteration - 1
                    break

                jac_pos.fill(0)
                mujoco.mj_jacSite(
                    self.model, data, jac_pos, None, self.site_id
                )
                jacobian = jac_pos[:, self.dof_indices]
                regularized = (
                    jacobian @ jacobian.T
                    + (self.damping**2) * np.eye(3)
                )
                delta_q = jacobian.T @ np.linalg.solve(regularized, delta)
                delta_q = np.clip(
                    delta_q, -self.max_delta_q, self.max_delta_q
                )
                data.qpos[self.qpos_indices] += self.step_size * delta_q
                self._clamp_joint_limits(data.qpos)
            else:
                iterations[frame_index] = self.max_iterations

            data.qpos[:] = best_qpos
            arm = data.qpos[self.qpos_indices]
            frame_delta = np.clip(
                arm - previous_arm,
                -self.max_frame_delta_q,
                self.max_frame_delta_q,
            )
            data.qpos[self.qpos_indices] = previous_arm + frame_delta
            self._clamp_joint_limits(data.qpos)
            mujoco.mj_forward(self.model, data)
            previous_arm = data.qpos[self.qpos_indices].copy()
            actual = data.site_xpos[self.site_id].copy()
            qpos_output[frame_index] = data.qpos
            ee_output[frame_index] = actual
            errors[frame_index] = np.linalg.norm(target - actual)
            if errors[frame_index] <= self.tolerance:
                converged[frame_index] = True

        return IKResult(
            qpos=qpos_output,
            arm_qpos=qpos_output[:, self.qpos_indices],
            ee_pos=ee_output,
            target_pos=targets.astype(np.float32),
            ik_error=errors.astype(np.float32),
            converged=converged,
            iterations=iterations,
        )

