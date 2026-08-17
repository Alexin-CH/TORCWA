"""
S-matrix solving mixin - delegates to implementation functions
"""

from ..smatrix.solve import _solve_global_smatrix, _solve_layer_smatrix
from ..smatrix.rsprod import _RS_prod
from ..smatrix.sparameters import _S_parameters


class SMatrixMixin:
    """
    Mixin providing S-matrix solving functionality.
    
    Public methods:
    - solve_global_smatrix:  Solve the global scattering matrix
    - S_parameters: Calculate S-parameters for specified diffraction orders
    
    Private methods: 
    - _solve_layer_smatrix: Calculate S-matrix for a single layer
    - _RS_prod:  Redheffer star product for connecting S-matrices
    """
    
    # Public API
    solve_global_smatrix = _solve_global_smatrix
    s_parameters = _S_parameters
    
    # Internal methods
    _solve_layer_smatrix = _solve_layer_smatrix
    _rs_prod = _RS_prod
    