from __future__ import annotations

from pathlib import Path

import numpy as np


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            'Plotting requires matplotlib. Install with: python -m pip install -e ".[plot]"'
        ) from exc
    return plt


def plot_pinch(
    timestamps: np.ndarray,
    distance: np.ndarray,
    command: np.ndarray,
    path: str | Path,
    *,
    close_threshold: float,
    open_threshold: float,
) -> Path:
    plt = _pyplot()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(timestamps, distance, label="minimum fingertip distance", color="#2563eb")
    axis.axhline(close_threshold, color="#dc2626", linestyle="--", label="close")
    axis.axhline(open_threshold, color="#16a34a", linestyle="--", label="open")
    axis.fill_between(
        timestamps,
        0,
        np.nanmax(distance) if np.any(np.isfinite(distance)) else 1,
        where=command > 0.5,
        alpha=0.15,
        color="#f59e0b",
        label="gripper closed",
    )
    axis.set(xlabel="time (s)", ylabel="normalized distance", title="Pinch detection")
    axis.grid(alpha=0.25)
    axis.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination


def plot_wrist(
    timestamps: np.ndarray,
    raw: np.ndarray,
    smooth: np.ndarray,
    path: str | Path,
) -> Path:
    plt = _pyplot()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for index, axis_name in enumerate("xyz"):
        axes[index].plot(timestamps, raw[:, index], alpha=0.4, label=f"raw {axis_name}")
        axes[index].plot(timestamps, smooth[:, index], label=f"smooth {axis_name}")
        axes[index].grid(alpha=0.25)
        axes[index].legend(loc="upper right")
    axes[-1].set_xlabel("time (s)")
    figure.suptitle("Wrist trajectory smoothing")
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination


def plot_target(target: np.ndarray, path: str | Path) -> Path:
    plt = _pyplot()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(8, 7))
    axis = figure.add_subplot(111, projection="3d")
    axis.plot(target[:, 0], target[:, 1], target[:, 2], color="#7c3aed")
    axis.scatter(*target[0], color="#16a34a", label="start")
    axis.scatter(*target[-1], color="#dc2626", label="end")
    axis.set(xlabel="robot x (m)", ylabel="robot y (m)", zlabel="robot z (m)")
    axis.set_title("OpenArm target trajectory")
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination

