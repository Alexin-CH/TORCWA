"""
Utility functions mixin - delegates to implementation functions
"""

from ..utils.angle import _set_incident_angle, _matching_indices, _diffraction_angle
from ..utils.eigen_decomposition import _eigen_decomposition, _eigen_decomposition_homogenous
from ..utils.kvectors import _kvectors


class UtilsMixin:
    """
    Mixin providing utility functionality.

    Public methods:
    - set_incident_angle: Set the incident angle for simulation
    - diffraction_angle:  Calculate diffraction angles for specified orders

    Private methods:
    - _matching_indices: Get matching indices for Fourier orders
    - _kvectors: Calculate k-vector components
    - _eigen_decomposition: Eigenmode decomposition for inhomogeneous layers
    - _eigen_decomposition_homogenous: Eigenmode decomposition for homogeneous layers
    """

    # Public API
    set_incident_angle = _set_incident_angle
    diffraction_angle = _diffraction_angle

    # Internal methods
    _matching_indices = _matching_indices
    _kvectors = _kvectors
    _eigen_decomposition = _eigen_decomposition
    _eigen_decomposition_homogenous = _eigen_decomposition_homogenous
    