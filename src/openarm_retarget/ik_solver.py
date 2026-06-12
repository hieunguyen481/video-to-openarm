from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .openarm_model import BimanualOpenArmInfo, OpenArmModelInfo, reset_home


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


@dataclass(frozen=True)
class BimanualIKResult:
    qpos: np.ndarray
    left_arm_qpos: np.ndarray
    right_arm_qpos: np.ndarray
    left_ee_pos: np.ndarray
    right_ee_pos: np.ndarray
    left_target_pos: np.ndarray
    right_target_pos: np.ndarray
    left_ik_error: np.ndarray
    right_ik_error: np.ndarray
    converged: np.ndarray
    iterations: np.ndarray


@dataclass(frozen=True)
class BimanualIKFrame:
    qpos: np.ndarray
    left_ee_pos: np.ndarray
    right_ee_pos: np.ndarray
    left_ik_error: float
    right_ik_error: float
    converged: bool
    iterations: int


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


class BimanualJacobianIKSolver:
    def __init__(
        self,
        model: Any,
        info: BimanualOpenArmInfo,
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

        self.site_ids: dict[str, int] = {}
        self.joint_ids: dict[str, np.ndarray] = {}
        self.qpos_indices: dict[str, np.ndarray] = {}
        self.dof_indices: dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            arm = info.sides[side]
            self.site_ids[side] = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SITE, arm.ee_site
            )
            joint_ids = np.asarray(
                [
                    mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_JOINT, name
                    )
                    for name in arm.arm_joint_names
                ],
                dtype=int,
            )
            self.joint_ids[side] = joint_ids
            self.qpos_indices[side] = model.jnt_qposadr[joint_ids].astype(int)
            self.dof_indices[side] = model.jnt_dofadr[joint_ids].astype(int)
        self.all_joint_ids = np.concatenate(
            (self.joint_ids["left"], self.joint_ids["right"])
        )
        self.all_qpos_indices = np.concatenate(
            (self.qpos_indices["left"], self.qpos_indices["right"])
        )
        self.all_dof_indices = np.concatenate(
            (self.dof_indices["left"], self.dof_indices["right"])
        )

    def _clamp_joint_limits(self, qpos: np.ndarray) -> None:
        for joint_id, qpos_index in zip(
            self.all_joint_ids, self.all_qpos_indices, strict=True
        ):
            if self.model.jnt_limited[joint_id]:
                lower, upper = self.model.jnt_range[joint_id]
                qpos[qpos_index] = np.clip(qpos[qpos_index], lower, upper)

    def solve(
        self,
        left_target_pos: np.ndarray,
        right_target_pos: np.ndarray,
    ) -> BimanualIKResult:
        mujoco = _mujoco()
        targets = {
            "left": np.asarray(left_target_pos, dtype=float),
            "right": np.asarray(right_target_pos, dtype=float),
        }
        for side, target in targets.items():
            if target.ndim != 2 or target.shape[1] != 3:
                raise ValueError(
                    f"{side}_target_pos must have shape [T, 3], got {target.shape}"
                )
            if not np.all(np.isfinite(target)):
                raise ValueError(f"{side}_target_pos contains NaN or infinite values")
        if len(targets["left"]) != len(targets["right"]):
            raise ValueError("Left and right targets must have the same length")

        data = mujoco.MjData(self.model)
        reset_home(self.model, data, self.info.home_keyframe)
        frame_count = len(targets["left"])
        qpos_output = np.empty((frame_count, self.model.nq), dtype=np.float64)
        ee_output = {
            side: np.empty((frame_count, 3), dtype=np.float64)
            for side in ("left", "right")
        }
        errors = {
            side: np.empty(frame_count, dtype=np.float64)
            for side in ("left", "right")
        }
        converged = np.zeros(frame_count, dtype=bool)
        iterations = np.zeros(frame_count, dtype=np.int32)
        jac_pos = {
            side: np.zeros((3, self.model.nv), dtype=np.float64)
            for side in ("left", "right")
        }
        previous_arm = data.qpos[self.all_qpos_indices].copy()

        for frame_index in range(frame_count):
            best_qpos = data.qpos.copy()
            best_error = np.inf
            for iteration in range(1, self.max_iterations + 1):
                mujoco.mj_forward(self.model, data)
                deltas = {
                    side: targets[side][frame_index]
                    - data.site_xpos[self.site_ids[side]]
                    for side in ("left", "right")
                }
                side_errors = {
                    side: float(np.linalg.norm(delta))
                    for side, delta in deltas.items()
                }
                combined_error = float(
                    np.linalg.norm(
                        np.concatenate((deltas["left"], deltas["right"]))
                    )
                )
                if combined_error < best_error:
                    best_error = combined_error
                    best_qpos = data.qpos.copy()
                if max(side_errors.values()) <= self.tolerance:
                    converged[frame_index] = True
                    iterations[frame_index] = iteration - 1
                    break

                jacobian_blocks = []
                for side in ("left", "right"):
                    jac_pos[side].fill(0)
                    mujoco.mj_jacSite(
                        self.model,
                        data,
                        jac_pos[side],
                        None,
                        self.site_ids[side],
                    )
                    jacobian_blocks.append(
                        jac_pos[side][:, self.all_dof_indices]
                    )
                jacobian = np.vstack(jacobian_blocks)
                delta = np.concatenate((deltas["left"], deltas["right"]))
                regularized = (
                    jacobian @ jacobian.T
                    + (self.damping**2) * np.eye(6)
                )
                delta_q = jacobian.T @ np.linalg.solve(regularized, delta)
                delta_q = np.clip(
                    delta_q, -self.max_delta_q, self.max_delta_q
                )
                data.qpos[self.all_qpos_indices] += self.step_size * delta_q
                self._clamp_joint_limits(data.qpos)
            else:
                iterations[frame_index] = self.max_iterations

            data.qpos[:] = best_qpos
            arm = data.qpos[self.all_qpos_indices]
            frame_delta = np.clip(
                arm - previous_arm,
                -self.max_frame_delta_q,
                self.max_frame_delta_q,
            )
            data.qpos[self.all_qpos_indices] = previous_arm + frame_delta
            self._clamp_joint_limits(data.qpos)
            mujoco.mj_forward(self.model, data)
            previous_arm = data.qpos[self.all_qpos_indices].copy()
            qpos_output[frame_index] = data.qpos
            for side in ("left", "right"):
                actual = data.site_xpos[self.site_ids[side]].copy()
                ee_output[side][frame_index] = actual
                errors[side][frame_index] = np.linalg.norm(
                    targets[side][frame_index] - actual
                )
            if max(errors["left"][frame_index], errors["right"][frame_index]) <= self.tolerance:
                converged[frame_index] = True

        return BimanualIKResult(
            qpos=qpos_output,
            left_arm_qpos=qpos_output[:, self.qpos_indices["left"]],
            right_arm_qpos=qpos_output[:, self.qpos_indices["right"]],
            left_ee_pos=ee_output["left"],
            right_ee_pos=ee_output["right"],
            left_target_pos=targets["left"].astype(np.float32),
            right_target_pos=targets["right"].astype(np.float32),
            left_ik_error=errors["left"].astype(np.float32),
            right_ik_error=errors["right"].astype(np.float32),
            converged=converged,
            iterations=iterations,
        )


class StatefulBimanualJacobianIKSolver(BimanualJacobianIKSolver):
    """Warm-started bimanual IK for one target pair at a time."""

    def __init__(
        self,
        model: Any,
        info: BimanualOpenArmInfo,
        config: Mapping[str, Any],
    ) -> None:
        super().__init__(model, info, config)
        mujoco = _mujoco()
        self.data = mujoco.MjData(model)
        self._jac_pos = {
            side: np.zeros((3, model.nv), dtype=np.float64)
            for side in ("left", "right")
        }
        self.reset()

    def reset(self, qpos: np.ndarray | None = None) -> None:
        mujoco = _mujoco()
        reset_home(self.model, self.data, self.info.home_keyframe)
        if qpos is not None:
            values = np.asarray(qpos, dtype=float)
            if values.shape != (self.model.nq,):
                raise ValueError(f"qpos must have shape {(self.model.nq,)}")
            self.data.qpos[:] = values
            self._clamp_joint_limits(self.data.qpos)
            mujoco.mj_forward(self.model, self.data)
        self._previous_arm = self.data.qpos[self.all_qpos_indices].copy()

    def solve_frame(
        self,
        left_target_pos: np.ndarray,
        right_target_pos: np.ndarray,
    ) -> BimanualIKFrame:
        mujoco = _mujoco()
        targets = {
            "left": np.asarray(left_target_pos, dtype=float),
            "right": np.asarray(right_target_pos, dtype=float),
        }
        for side, target in targets.items():
            if target.shape != (3,) or not np.all(np.isfinite(target)):
                raise ValueError(
                    f"{side}_target_pos must contain three finite values"
                )

        best_qpos = self.data.qpos.copy()
        best_error = np.inf
        iterations = self.max_iterations
        for iteration in range(self.max_iterations + 1):
            mujoco.mj_forward(self.model, self.data)
            deltas = {
                side: targets[side] - self.data.site_xpos[self.site_ids[side]]
                for side in ("left", "right")
            }
            side_errors = {
                side: float(np.linalg.norm(delta))
                for side, delta in deltas.items()
            }
            combined_error = float(
                np.linalg.norm(
                    np.concatenate((deltas["left"], deltas["right"]))
                )
            )
            if combined_error < best_error:
                best_error = combined_error
                best_qpos = self.data.qpos.copy()
            if max(side_errors.values()) <= self.tolerance:
                iterations = iteration
                break
            if iteration == self.max_iterations:
                break

            jacobian_blocks = []
            for side in ("left", "right"):
                self._jac_pos[side].fill(0)
                mujoco.mj_jacSite(
                    self.model,
                    self.data,
                    self._jac_pos[side],
                    None,
                    self.site_ids[side],
                )
                jacobian_blocks.append(
                    self._jac_pos[side][:, self.all_dof_indices]
                )
            jacobian = np.vstack(jacobian_blocks)
            delta = np.concatenate((deltas["left"], deltas["right"]))
            regularized = (
                jacobian @ jacobian.T
                + (self.damping**2) * np.eye(6)
            )
            delta_q = jacobian.T @ np.linalg.solve(regularized, delta)
            delta_q = np.clip(
                delta_q, -self.max_delta_q, self.max_delta_q
            )
            self.data.qpos[self.all_qpos_indices] += (
                self.step_size * delta_q
            )
            self._clamp_joint_limits(self.data.qpos)

        self.data.qpos[:] = best_qpos
        arm = self.data.qpos[self.all_qpos_indices]
        frame_delta = np.clip(
            arm - self._previous_arm,
            -self.max_frame_delta_q,
            self.max_frame_delta_q,
        )
        self.data.qpos[self.all_qpos_indices] = (
            self._previous_arm + frame_delta
        )
        self._clamp_joint_limits(self.data.qpos)
        mujoco.mj_forward(self.model, self.data)
        self._previous_arm = self.data.qpos[self.all_qpos_indices].copy()

        ee_pos = {
            side: self.data.site_xpos[self.site_ids[side]].copy()
            for side in ("left", "right")
        }
        errors = {
            side: float(np.linalg.norm(targets[side] - ee_pos[side]))
            for side in ("left", "right")
        }
        return BimanualIKFrame(
            qpos=self.data.qpos.copy(),
            left_ee_pos=ee_pos["left"],
            right_ee_pos=ee_pos["right"],
            left_ik_error=errors["left"],
            right_ik_error=errors["right"],
            converged=max(errors.values()) <= self.tolerance,
            iterations=iterations,
        )
