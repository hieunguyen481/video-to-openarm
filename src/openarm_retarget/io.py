from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

SCHEMA_VERSION = "1.0"


def _as_storable(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (str, bytes, bool, int, float, np.generic)):
        return np.asarray(value)
    if value is None:
        return np.asarray("null")
    if isinstance(value, (dict, list, tuple)):
        return np.asarray(json.dumps(value, ensure_ascii=True, sort_keys=True))
    raise TypeError(f"Unsupported NPZ value type: {type(value).__name__}")


def save_npz(
    path: str | Path,
    data: Mapping[str, Any],
    *,
    stage: str,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: _as_storable(value) for key, value in data.items()}
    payload["_schema_version"] = np.asarray(SCHEMA_VERSION)
    payload["_stage"] = np.asarray(stage)
    payload["_metadata_json"] = np.asarray(
        json.dumps(dict(metadata or {}), ensure_ascii=True, sort_keys=True)
    )

    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.stem}.",
            suffix=".npz",
            delete=False,
        ) as handle:
            temp_name = handle.name
        np.savez_compressed(temp_name, **payload)
        os.replace(temp_name, destination)
    finally:
        if temp_name and Path(temp_name).exists():
            Path(temp_name).unlink()
    return destination


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"NPZ file not found: {source}")
    with np.load(source, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def metadata_from(data: Mapping[str, np.ndarray]) -> dict[str, Any]:
    raw = data.get("_metadata_json")
    if raw is None:
        return {}
    return json.loads(str(np.asarray(raw).item()))

