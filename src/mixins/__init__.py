"""
Mixins for RCWA class organization. 

Each mixin groups related functionality: 
- LayerMixin: Layer management operations
- FieldMixin:  Electromagnetic field calculations
- SMatrixMixin: S-matrix solving operations
- UtilsMixin:  Utility functions (angles, k-vectors, eigendecomposition)
"""

from .layers import LayerMixin
from .fields import FieldMixin
from .smatrix import SMatrixMixin
from .utils import UtilsMixin

__all__ = ['LayerMixin', 'FieldMixin', 'SMatrixMixin', 'UtilsMixin']
