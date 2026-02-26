from __future__ import annotations

import numpy as np
import pytest

from flashs.core.rff import KernelType, RFFParams, sample_spectral_frequencies


@pytest.mark.parametrize(
    "kernel",
    [
        KernelType.GAUSSIAN,
        KernelType.LAPLACIAN,
        KernelType.CAUCHY,
        KernelType.MATERN_1_2,
        KernelType.MATERN_3_2,
        KernelType.MATERN_5_2,
    ],
)
def test_sample_spectral_frequencies_supported_kernels(kernel: KernelType) -> None:
    omega = sample_spectral_frequencies(
        n_features=16,
        n_dims=3,
        bandwidth=np.array([0.5, 1.0, 2.0]),
        kernel=kernel,
        rng=np.random.default_rng(0),
    )
    assert omega.shape == (16, 3)
    assert omega.dtype == np.float64
    assert np.isfinite(omega).all()


def test_sample_spectral_frequencies_validates_bandwidth_shape() -> None:
    with pytest.raises(ValueError, match="must be 1 or n_dims"):
        sample_spectral_frequencies(
            n_features=8,
            n_dims=3,
            bandwidth=np.array([1.0, 2.0]),
            kernel=KernelType.GAUSSIAN,
            rng=np.random.default_rng(1),
        )


def test_sample_spectral_frequencies_rejects_unknown_kernel() -> None:
    with pytest.raises(ValueError, match="Unknown kernel type"):
        sample_spectral_frequencies(
            n_features=8,
            n_dims=2,
            bandwidth=1.0,
            kernel="bad_kernel",  # type: ignore[arg-type]
            rng=np.random.default_rng(2),
        )


def test_rffparams_properties() -> None:
    omega = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float64)
    bias = np.array([0.0, np.pi / 2], dtype=np.float64)
    params = RFFParams(omega=omega, bias=bias, scale=1.0)

    assert params.n_features == 2
    assert params.n_dims == 2
