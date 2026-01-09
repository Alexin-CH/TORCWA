"""
TORCWA - Torch Rigorous Coupled Wave Analysis

A PyTorch-based implementation of RCWA for photonic simulations.
"""

from .main import rcwa as rcwa
from .core.torch_eig import Eig as Eig
from .core.geometry import geometry as geometry, rcwa_geo as rcwa_geo

__all__ = ['rcwa']
