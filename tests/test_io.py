from __future__ import annotations

import numpy as np

from openarm_retarget.io import load_npz, metadata_from, save_npz


def test_npz_round_trip(tmp_path):
    path = save_npz(
        tmp_path / "sample.npz",
        {"values": np.arange(6, dtype=np.float32).reshape(2, 3)},
        stage="test",
        metadata={"source": "synthetic"},
    )

    loaded = load_npz(path)

    np.testing.assert_array_equal(
        loaded["values"], np.arange(6, dtype=np.float32).reshape(2, 3)
    )
    assert loaded["_stage"].item() == "test"
    assert metadata_from(loaded) == {"source": "synthetic"}

