from __future__ import annotations

import numpy as np
import pytest

from flashs.core.bandwidth import compute_adaptive_scales, estimate_bandwidth


def test_estimate_bandwidth_rejects_unknown_method() -> None:
    coords = np.random.default_rng(0).random((20, 2))
    with pytest.raises(ValueError, match="Unknown method"):
        estimate_bandwidth(coords, method="bad")  # type: ignore[arg-type]


def test_compute_adaptive_scales_rejects_unknown_coverage() -> None:
    coords = np.random.default_rng(1).random((20, 2))
    with pytest.raises(ValueError, match="Unknown coverage"):
        compute_adaptive_scales(coords, coverage="bad")  # type: ignore[arg-type]


def test_small_sample_fallbacks_are_stable() -> None:
    coords = np.array([[0.0, 1.0]])

    bw = estimate_bandwidth(coords)
    assert bw.primary == 1.0
    assert bw.multiscale == [1.0]

    scales = compute_adaptive_scales(coords, n_scales=4)
    assert scales == [1.0, 1.0, 1.0, 1.0]
