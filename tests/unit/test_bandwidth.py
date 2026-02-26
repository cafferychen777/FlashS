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


def test_spectral_fallback_handles_expected_solver_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coords = np.random.default_rng(3).random((60, 2))

    def _raise_value_error(*args, **kwargs):  # noqa: ANN002, ANN003
        raise ValueError("solver failed")

    monkeypatch.setattr("flashs.core.bandwidth.lobpcg", _raise_value_error)
    bw = estimate_bandwidth(coords, method="spectral", random_state=0)
    assert np.isfinite(bw.primary)
    assert bw.primary > 0


def test_spectral_fallback_does_not_swallow_interrupts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coords = np.random.default_rng(4).random((60, 2))

    def _raise_interrupt(*args, **kwargs):  # noqa: ANN002, ANN003
        raise KeyboardInterrupt

    monkeypatch.setattr("flashs.core.bandwidth.lobpcg", _raise_interrupt)
    with pytest.raises(KeyboardInterrupt):
        estimate_bandwidth(coords, method="spectral", random_state=0)


def test_estimate_bandwidth_quantile_and_median_paths() -> None:
    coords = np.random.default_rng(5).random((80, 2))
    q = estimate_bandwidth(coords, method="quantile", quantiles=[0.2, 0.5, 0.8])
    assert len(q.multiscale) == 3
    assert q.primary == q.multiscale[1]
    assert q.median_distance > 0

    m = estimate_bandwidth(coords, method="median")
    assert m.primary == pytest.approx(m.median_distance)


def test_estimate_bandwidth_handles_collocated_points() -> None:
    coords = np.zeros((10, 2), dtype=np.float64)
    out = estimate_bandwidth(coords, method="auto")
    assert out.primary == 1.0
    assert out.multiscale == [1.0]
    assert out.median_distance == 0.0


def test_compute_adaptive_scales_linear_and_adaptive_lengths() -> None:
    coords = np.random.default_rng(6).random((120, 2))
    linear = compute_adaptive_scales(coords, n_scales=6, coverage="linear")
    adaptive = compute_adaptive_scales(coords, n_scales=6, coverage="adaptive")
    assert len(linear) == 6
    assert len(adaptive) == 6
    assert linear == sorted(linear)
    assert adaptive == sorted(adaptive)
