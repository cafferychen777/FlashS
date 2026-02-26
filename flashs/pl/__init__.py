"""Plotting module (scanpy-style API)."""

from ._svg import spatial_variable_genes, volcano

svg = spatial_variable_genes

__all__ = ["spatial_variable_genes", "svg", "volcano"]
