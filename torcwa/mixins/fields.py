"""
Field calculation mixin - delegates to implementation functions
"""
from ..fields.source import _source_planewave, _source_fourier
from ..fields.xy import _field_xy
from ..fields.xz import _field_xz
from ..fields.yz import _field_yz


class FieldMixin:
    """
    Mixin for electromagnetic field calculations.

    Public methods:
    - source_planewave: Set the incident field from a plane wave
    - source_fourier: Set the incident field from Fourier amplitudes
    - field_xy: XY-plane field distribution at a given z position
    - field_xz: XZ-plane field distribution at a given y position
    - field_yz: YZ-plane field distribution at a given x position
    """

    # Direct assignment - no code duplication
    source_planewave = _source_planewave
    source_fourier = _source_fourier
    field_xy = _field_xy
    field_xz = _field_xz
    field_yz = _field_yz
    