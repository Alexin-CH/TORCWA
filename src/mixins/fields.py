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
    Implementation is in src/fields/ modules.
    """
    
    # Direct assignment - no code duplication
    source_planewave = _source_planewave
    source_fourier = _source_fourier
    field_xy = _field_xy
    field_xz = _field_xz
    field_yz = _field_yz
    