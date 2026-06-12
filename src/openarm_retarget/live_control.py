from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .pinch import FINGER_KEYS
from .retargeting import retarget_wrist


@dataclass(frozen=True)
class LiveHandSample:
    wrist: np.ndarray
    palm_scale: float
    thumb_tip: np.ndarray
    finger_tips: np.ndarray


class LiveRetargeter:
    def __init__(
        self,
        configs: Mapping[str, Mapping[str, Any]],
        *,
        smoothing_tau_s: float = 0.08,
        max_target_speed_m_s: float = 0.8,
        lost_hand_timeout_s: float = 0.5,
    ) -> None:
        if smoothing_tau_s <= 0:
            raise ValueError("smoothing_tau_s must be positive")
        if max_target_speed_m_s <= 0:
            raise ValueError("max_target_speed_m_s must be positive")
        if lost_hand_timeout_s <= 0:
            raise ValueError("lost_hand_timeout_s must be positive")
        self.configs = {
            side: dict(configs[side]) for side in ("left", "right")
        }
        self.smoothing_tau_s = smoothing_tau_s
        self.max_target_speed_m_s = max_target_speed_m_s
        self.lost_hand_timeout_s = lost_hand_timeout_s
        self._targets = {
            side: np.asarray(
                self.configs[side]["openarm_origin"], dtype=float
            )
            for side in ("left", "right")
        }
        self._references: dict[str, np.ndarray | None] = {
            "left": None,
            "right": None,
        }
        self._anchors = {
            side: target.copy() for side, target in self._targets.items()
        }
        self._last_update: dict[str, float | None] = {
            "left": None,
            "right": None,
        }
        self._last_seen: dict[str, float | None] = {
            "left": None,
            "right": None,
        }

    def reset_reference(self) -> None:
        for side in ("left", "right"):
            self._references[side] = None
            self._anchors[side] = self._targets[side].copy()

    def target(self, side: str) -> np.ndarray:
        return self._targets[side].copy()

    def update(
        self,
        side: str,
        sample: LiveHandSample | None,
        timestamp: float,
    ) -> np.ndarray:
        if side not in self._targets:
            raise ValueError("side must be 'left' or 'right'")
        if sample is None:
            last_seen = self._last_seen[side]
            if (
                last_seen is not None
                and timestamp - last_seen >= self.lost_hand_timeout_s
            ):
                self._references[side] = None
                self._anchors[side] = self._targets[side].copy()
            return self.target(side)

        human = np.asarray(
            [sample.wrist[0], sample.wrist[1], sample.palm_scale],
            dtype=float,
        )
        if not np.all(np.isfinite(human)):
            return self.update(side, None, timestamp)
        self._last_seen[side] = timestamp

        reference = self._references[side]
        if reference is None:
            self._references[side] = human
            self._anchors[side] = self._targets[side].copy()
            self._last_update[side] = timestamp
            return self.target(side)

        config = dict(self.configs[side])
        config["openarm_origin"] = self._anchors[side].tolist()
        desired = retarget_wrist(
            np.stack((reference, human)),
            config,
        )[1].astype(float)
        previous_time = self._last_update[side]
        dt = (
            max(timestamp - previous_time, 1e-3)
            if previous_time is not None
            else 1.0 / 30.0
        )
        alpha = 1.0 - math.exp(-dt / self.smoothing_tau_s)
        filtered = self._targets[side] + alpha * (
            desired - self._targets[side]
        )
        delta = filtered - self._targets[side]
        distance = float(np.linalg.norm(delta))
        allowed = self.max_target_speed_m_s * dt
        if distance > allowed:
            filtered = self._targets[side] + delta * (allowed / distance)
        self._targets[side] = filtered
        self._last_update[side] = timestamp
        return self.target(side)


class LivePinchDetector:
    def __init__(
        self,
        *,
        close_threshold: float,
        open_threshold: float,
        initial_state: int = 0,
        close_confirm_frames: int = 1,
        open_confirm_frames: int = 1,
        lost_hand_timeout_s: float = 0.5,
        open_on_loss: bool = True,
    ) -> None:
        if close_threshold >= open_threshold:
            raise ValueError("close_threshold must be smaller than open_threshold")
        if close_confirm_frames < 1 or open_confirm_frames < 1:
            raise ValueError("confirmation frames must be positive")
        self.close_threshold = close_threshold
        self.open_threshold = open_threshold
        self.close_confirm_frames = close_confirm_frames
        self.open_confirm_frames = open_confirm_frames
        self.lost_hand_timeout_s = lost_hand_timeout_s
        self.open_on_loss = open_on_loss
        self.state = float(bool(initial_state))
        self.close_count = 0
        self.open_count = 0
        self.last_seen: float | None = None

    def update(
        self,
        sample: LiveHandSample | None,
        timestamp: float,
    ) -> float:
        if sample is None:
            self.close_count = 0
            self.open_count = 0
            if (
                self.open_on_loss
                and self.last_seen is not None
                and timestamp - self.last_seen >= self.lost_hand_timeout_s
            ):
                self.state = 0.0
            return self.state

        self.last_seen = timestamp
        distance = float(
            np.min(
                np.linalg.norm(
                    sample.finger_tips - sample.thumb_tip[None, :],
                    axis=1,
                )
            )
        )
        if distance < self.close_threshold:
            self.close_count += 1
            self.open_count = 0
            if self.close_count >= self.close_confirm_frames:
                self.state = 1.0
        elif distance > self.open_threshold:
            self.open_count += 1
            self.close_count = 0
            if self.open_count >= self.open_confirm_frames:
                self.state = 0.0
        else:
            self.close_count = 0
            self.open_count = 0
        return self.state


class LiveMetrics:
    def __init__(self, *, window: int = 300) -> None:
        self.started_at = time.monotonic()
        self.processed_frames = 0
        self.captured_sequence = 0
        self.latency_ms: deque[float] = deque(maxlen=window)
        self.inference_ms: deque[float] = deque(maxlen=window)
        self.ik_ms: deque[float] = deque(maxlen=window)
        self.render_ms: deque[float] = deque(maxlen=window)

    def record(
        self,
        *,
        sequence: int,
        latency_ms: float,
        inference_ms: float,
        ik_ms: float,
        render_ms: float = 0.0,
    ) -> None:
        self.processed_frames += 1
        self.captured_sequence = max(self.captured_sequence, sequence)
        self.latency_ms.append(latency_ms)
        self.inference_ms.append(inference_ms)
        self.ik_ms.append(ik_ms)
        self.render_ms.append(render_ms)

    @staticmethod
    def _stats(values: deque[float]) -> dict[str, float]:
        if not values:
            return {"mean": 0.0, "p95": 0.0, "max": 0.0}
        array = np.asarray(values, dtype=float)
        return {
            "mean": float(np.mean(array)),
            "p95": float(np.percentile(array, 95)),
            "max": float(np.max(array)),
        }

    def summary(self) -> dict[str, Any]:
        elapsed = max(time.monotonic() - self.started_at, 1e-6)
        return {
            "processed_frames": self.processed_frames,
            "dropped_or_skipped_frames": max(
                self.captured_sequence - self.processed_frames, 0
            ),
            "processing_fps": self.processed_frames / elapsed,
            "latency_ms": self._stats(self.latency_ms),
            "inference_ms": self._stats(self.inference_ms),
            "ik_ms": self._stats(self.ik_ms),
            "render_ms": self._stats(self.render_ms),
        }


def sample_from_landmarks(landmarks: Any) -> LiveHandSample:
    def point(index: int) -> np.ndarray:
        item = landmarks[index]
        return np.asarray([item.x, item.y, item.z], dtype=np.float32)

    wrist_2d = point(0)
    palm_distances = [
        np.linalg.norm(point(index)[:2] - wrist_2d[:2])
        for index in (5, 9, 13, 17)
    ]
    finger_indices = {
        "index_tip": 8,
        "middle_tip": 12,
        "ring_tip": 16,
        "pinky_tip": 20,
    }
    return LiveHandSample(
        wrist=wrist_2d,
        palm_scale=float(np.mean(palm_distances)),
        thumb_tip=point(4),
        finger_tips=np.stack(
            [point(finger_indices[key]) for key in FINGER_KEYS]
        ),
    )
