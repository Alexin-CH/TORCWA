"""
Layer management mixin - delegates to implementation functions
"""
from ..utils.layers import (
    _add_layer,
    _add_input_layer, 
    _add_output_layer,
    _return_layer,
    _material_conv
)


class LayerMixin: 
    """
    Mixin providing layer management functionality.
    
    Public methods:
    - add_layer:  Add an internal layer to the structure
    - add_input_layer: Define the input layer properties
    - add_output_layer: Define the output layer properties
    - return_layer:  Retrieve spatial distribution of a layer
    
    Private methods:
    - _material_conv: Convert material distribution to convolution matrix
    """
    
    # Direct assignment
    add_layer = _add_layer
    add_input_layer = _add_input_layer
    add_output_layer = _add_output_layer
    return_layer = _return_layer
    _material_conv = _material_conv
    