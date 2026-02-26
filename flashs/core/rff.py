"""
Random Fourier Features (RFF) for kernel approximation.

Based on Bochner's theorem: shift-invariant positive definite kernels
can be represented as Fourier transforms of probability measures.

Reference:
    Rahimi & Recht (2007). Random Features for Large-Scale Kernel Machines.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray


class KernelType(Enum):
    """Supported kernel types with their spectral distributions."""

    GAUSSIAN = "gaussian"      # p(ω) = N(0, 1/σ²)
    LAPLACIAN = "laplacian"    # p(ω) = Cauchy(0, 1/σ)
    CAUCHY = "cauchy"          # p(ω) = Laplace(0, 1/σ)
    MATERN_1_2 = "matern_1_2"  # Equivalent to Laplacian
    MATERN_3_2 = "matern_3_2"  # Student-t spectral density
    MATERN_5_2 = "matern_5_2"  # Student-t spectral density


def sample_spectral_frequencies(
    n_features: int,
    n_dims: int,
    bandwidth: float | NDArray[np.floating],
    kernel: KernelType = KernelType.GAUSSIAN,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """
    Sample frequency vectors from kernel spectral distribution.

    This is the core operation for Random Fourier Features: sample ω from p(ω)
    where p(ω) is the Fourier transform of the kernel k(r).

    Parameters
    ----------
    n_features : int
        Number of frequency vectors to sample (D).
    n_dims : int
        Spatial dimensionality (d).
    bandwidth : float or ndarray
        Kernel bandwidth σ. Scalar for isotropic, array of shape (d,) for anisotropic.
    kernel : KernelType, default=GAUSSIAN
        Kernel type determining spectral distribution.
    rng : np.random.Generator, optional
        Random number generator. If None, creates new one.

    Returns
    -------
    omega : ndarray of shape (n_features, n_dims)
        Sampled frequency vectors scaled by bandwidth.

    Notes
    -----
    Spectral distributions by kernel type:
    - Gaussian: p(ω) = N(0, 1/σ²)
    - Laplacian/Matérn-1/2: p(ω) = Cauchy(0, 1/σ)
    - Cauchy: p(ω) = Laplace(0, 1/σ)
    - Matérn-3/2: Student-t with ν=3
    - Matérn-5/2: Student-t with ν=5
    """
    if rng is None:
        rng = np.random.default_rng()

    sigma = np.atleast_1d(bandwidth).astype(np.float64)
    if sigma.shape[0] == 1:
        sigma = np.repeat(sigma, n_dims)
    elif sigma.shape[0] != n_dims:
        raise ValueError(
            f"bandwidth length ({sigma.shape[0]}) must be 1 or n_dims ({n_dims})"
        )

    if kernel == KernelType.GAUSSIAN:
        omega = rng.standard_normal((n_features, n_dims))
        omega /= sigma

    elif kernel in (KernelType.LAPLACIAN, KernelType.MATERN_1_2):
        omega = rng.standard_cauchy((n_features, n_dims))
        omega /= sigma

    elif kernel == KernelType.CAUCHY:
        omega = rng.laplace(0, 1, (n_features, n_dims))
        omega /= sigma

    elif kernel == KernelType.MATERN_3_2:
        omega = rng.standard_t(df=3, size=(n_features, n_dims))
        omega *= np.sqrt(3) / sigma

    elif kernel == KernelType.MATERN_5_2:
        omega = rng.standard_t(df=5, size=(n_features, n_dims))
        omega *= np.sqrt(5) / sigma

    else:
        raise ValueError(f"Unknown kernel type: {kernel}")

    return omega.astype(np.float64)


@dataclass
class RFFParams:
    """
    Sampled RFF parameters for spatial projection.

    Encapsulates all parameters needed for RFF projection,
    reducing parameter threading in function calls.
    """

    omega: NDArray[np.float64]
    """Frequency vectors (D, d)."""

    bias: NDArray[np.float64]
    """Phase offsets (D,)."""

    scale: float
    """Normalization factor sqrt(2/D)."""

    @property
    def n_features(self) -> int:
        """Number of RFF features D."""
        return self.omega.shape[0]

    @property
    def n_dims(self) -> int:
        """Spatial dimensionality d."""
        return self.omega.shape[1]


