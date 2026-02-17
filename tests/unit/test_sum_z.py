from __future__ import annotations

import numpy as np

from flashs import FlashS
from flashs.core.sketch import compute_sum_z


def test_compute_sum_z_matches_explicit_feature_sum() -> None:
    rng = np.random.default_rng(0)
    coords = rng.normal(size=(80, 3))
    omega = rng.normal(size=(24, 3))
    bias = rng.uniform(0.0, 2.0 * np.pi, size=24)
    scale = np.sqrt(2.0 / omega.shape[0])

    got = compute_sum_z(coords, omega, bias, scale)
    expected = (scale * np.cos(coords @ omega.T + bias)).sum(axis=0)

    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_flashs_fit_uses_exact_sum_z_on_large_n() -> None:
    rng = np.random.default_rng(1)
    n_cells = 12000  # larger than common subsampling cutoffs
    coords = rng.normal(size=(n_cells, 2))

    model = FlashS(n_features=48, n_scales=4, random_state=3).fit(coords)
    expected = compute_sum_z(coords, model._rff.omega, model._rff.bias, model._rff.scale)

    np.testing.assert_allclose(model._sum_z, expected, rtol=1e-12, atol=1e-12)
