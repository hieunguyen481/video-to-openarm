from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def require_keys(data: Mapping[str, object], keys: Sequence[str]) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")


def require_time_length(data: Mapping[str, object], keys: Sequence[str]) -> int:
    require_keys(data, keys)
    lengths = {key: len(np.asarray(data[key])) for key in keys}
    unique = set(lengths.values())
    if len(unique) != 1:
        details = ", ".join(f"{key}={value}" for key, value in lengths.items())
        raise ValueError(f"Inconsistent time dimensions: {details}")
    return next(iter(unique))


def require_shape(name: str, value: object, trailing_shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value)
    expected_ndim = 1 + len(trailing_shape)
    if array.ndim != expected_ndim or array.shape[1:] != trailing_shape:
        raise ValueError(
            f"{name} must have shape [T, {', '.join(map(str, trailing_shape))}], "
            f"got {array.shape}"
        )
    return array


def require_finite(name: str, value: object, *, valid: np.ndarray | None = None) -> None:
    array = np.asarray(value)
    checked = array if valid is None else array[np.asarray(valid, dtype=bool)]
    if not np.all(np.isfinite(checked)):
        raise ValueError(f"{name} contains NaN or infinite values")

