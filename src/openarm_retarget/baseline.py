from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np


def prepare_supervised_data(
    dataset: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qpos = np.asarray(dataset["obs_qpos"], dtype=np.float32)
    if "obs_left_target_pos" in dataset:
        target = np.concatenate(
            (
                np.asarray(dataset["obs_left_target_pos"], dtype=np.float32),
                np.asarray(dataset["obs_right_target_pos"], dtype=np.float32),
            ),
            axis=1,
        )
        gripper_state = np.concatenate(
            (
                np.asarray(dataset["obs_left_gripper_state"], dtype=np.float32),
                np.asarray(dataset["obs_right_gripper_state"], dtype=np.float32),
            ),
            axis=1,
        )
        arm_action = np.concatenate(
            (
                np.asarray(
                    dataset["action_left_arm_joint_target"], dtype=np.float32
                ),
                np.asarray(
                    dataset["action_right_arm_joint_target"], dtype=np.float32
                ),
            ),
            axis=1,
        )
        gripper_action = np.concatenate(
            (
                np.asarray(
                    dataset["action_left_gripper_cmd"], dtype=np.float32
                ),
                np.asarray(
                    dataset["action_right_gripper_cmd"], dtype=np.float32
                ),
            ),
            axis=1,
        ).astype(int)
    else:
        target = np.asarray(dataset["obs_target_pos"], dtype=np.float32)
        gripper_state = np.asarray(dataset["obs_gripper_state"], dtype=np.float32)
        arm_action = np.asarray(dataset["action_arm_joint_target"], dtype=np.float32)
        gripper_action = np.asarray(
            dataset["action_gripper_cmd"], dtype=np.float32
        )
    lengths = {len(item) for item in (qpos, target, gripper_state, arm_action, gripper_action)}
    if len(lengths) != 1 or len(qpos) < 3:
        raise ValueError("Dataset arrays must have the same length and at least 3 samples")
    features = np.concatenate((qpos[:-1], target[:-1], gripper_state[:-1]), axis=1)
    next_gripper = gripper_action[1:].astype(int)
    if next_gripper.shape[1] == 1:
        next_gripper = next_gripper.reshape(-1)
    return features, arm_action[1:], next_gripper


def train_baseline(
    dataset: Mapping[str, Any],
    *,
    output: str | Path,
    test_fraction: float = 0.2,
    random_state: int = 7,
) -> dict[str, float]:
    try:
        import joblib
        from sklearn.dummy import DummyClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, mean_squared_error
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError(
            'Baseline training requires scikit-learn and joblib. Install with: '
            'python -m pip install -e ".[learning]"'
        ) from exc

    features, arm_target, gripper_target = prepare_supervised_data(dataset)
    split = int(round(len(features) * (1.0 - test_fraction)))
    split = min(max(split, 1), len(features) - 1)
    train_x, test_x = features[:split], features[split:]
    train_arm, test_arm = arm_target[:split], arm_target[split:]
    train_gripper, test_gripper = gripper_target[:split], gripper_target[split:]

    arm_model = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            early_stopping=True,
            max_iter=500,
            random_state=random_state,
        ),
    )
    arm_model.fit(train_x, train_arm)
    arm_prediction = arm_model.predict(test_x)
    metrics = {
        "train_samples": float(len(train_x)),
        "test_samples": float(len(test_x)),
        "arm_rmse_rad": float(
            np.sqrt(mean_squared_error(test_arm, arm_prediction))
        ),
    }
    gripper_models = {}
    target_columns = (
        {"gripper": (train_gripper, test_gripper)}
        if train_gripper.ndim == 1
        else {
            "left_gripper": (train_gripper[:, 0], test_gripper[:, 0]),
            "right_gripper": (train_gripper[:, 1], test_gripper[:, 1]),
        }
    )
    for label, (train_column, test_column) in target_columns.items():
        if len(np.unique(train_column)) > 1:
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=500, random_state=random_state),
            )
        else:
            model = DummyClassifier(strategy="most_frequent")
        model.fit(train_x, train_column)
        prediction = model.predict(test_x)
        gripper_models[label] = model
        metrics[f"{label}_accuracy"] = float(
            accuracy_score(test_column, prediction)
        )
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "arm_model": arm_model,
            "gripper_models": gripper_models,
            "feature_order": ["qpos", "target_pos_both", "gripper_state_both"],
            "metrics": metrics,
        },
        destination,
    )
    return metrics
