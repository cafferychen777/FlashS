"""
Base result classes for spatial transcriptomics tests.

Provides a unified interface for test results across different
spatial analysis methods (SVG, etc.).
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class SpatialTestResult(ABC):
    """
    Base class for spatial test results.

    Provides common functionality for all spatial testing methods:
    - P-value storage and multiple testing correction
    - Effect size computation
    - Significant gene filtering
    - DataFrame conversion

    Subclasses should add method-specific fields (e.g., binary/rank p-values
    for SVG).
    """

    gene_names: list[str]
    """Gene names, aligned with input order."""

    pvalues: NDArray[np.floating]
    """Raw p-values."""

    qvalues: NDArray[np.floating]
    """Multiple testing corrected q-values (FDR)."""

    statistics: NDArray[np.floating]
    """Test statistics."""

    effect_size: NDArray[np.floating]
    """Effect size measure (interpretation varies by method)."""

    n_tested: int
    """Number of genes tested."""

    n_significant: int = field(init=False)
    """Number of significant genes (q < 0.05)."""

    def __post_init__(self):
        self.n_significant = int(np.sum(self.qvalues < 0.05))

    def significant_genes(
        self,
        q_threshold: float = 0.05,
        effect_threshold: float | None = None,
    ) -> list[str]:
        """
        Get list of significant genes.

        Parameters
        ----------
        q_threshold : float, default=0.05
            Q-value (FDR) threshold.
        effect_threshold : float, optional
            Minimum effect size threshold.

        Returns
        -------
        genes : list[str]
            Names of significant genes.
        """
        mask = self.qvalues < q_threshold
        if effect_threshold is not None:
            mask = mask & (self.effect_size > effect_threshold)
        return [g for g, m in zip(self.gene_names, mask) if m]

    def _build_dataframe_dict(self) -> dict:
        """
        Build base dictionary for DataFrame conversion.

        Subclasses should call this and extend with additional columns.
        """
        return {
            "gene": self.gene_names,
            "pvalue": self.pvalues,
            "qvalue": self.qvalues,
            "statistic": self.statistics,
            "effect_size": self.effect_size,
        }

    def to_dataframe(self):
        """
        Convert results to pandas DataFrame.

        Returns
        -------
        df : pd.DataFrame
            Results sorted by p-value.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for to_dataframe()")

        df_dict = self._build_dataframe_dict()
        return pd.DataFrame(df_dict).sort_values("pvalue")
