import torch
from math import pi

from .torch_eig import Eig
from .fields.source import _source_planewave, _source_fourier
from .fields.xy import _field_xy
from .fields.xz import _field_xz
from .fields.yz import _field_yz

from .smatrix.solve import _solve_global_smatrix, _solve_layer_smatrix
from .smatrix.rsprod import _RS_prod
from .smatrix.sparameters import _S_parameters

from .utils.angle import _set_incident_angle, _matching_indices, _diffraction_angle
from .utils.eigen_decomposition import _eigen_decomposition, _eigen_decomposition_homogenous
from .utils.kvectors import _kvectors
from .utils.layers import _return_layer, _add_input_layer, _add_output_layer, _add_layer, _material_conv

class rcwa:
    # Simulation setting
    def __init__(
        self,
        freq,
        order,
        L,
        *,
        dtype=torch.complex64,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        stable_eig_grad=True,
        avoid_Pinv_instability=False,
        max_Pinv_instability=0.005
    ):
        """
        Rigorous Coupled Wave Analysis
        - Lorentz-Heaviside units
        - Speed of light: 1
        - Time harmonics notation: exp(-jωt)

        Parameters
        - freq: simulation frequency (unit: length^-1)
        - order: Fourier order [x_order (int), y_order (int)]
        - L: Lattice constant [Lx, Ly] (unit: length)

        Keyword Parameters
        - dtype: simulation data type (only torch.complex64 and torch.complex128 are allowed.)
        - device: simulation device (only torch.device('cpu') and torch.device('cuda') are allowed.)
        - stable_eig_grad: stabilize gradient calculation of eigendecompsition (default as True)
        - avoid_Pinv_instability: avoid instability of P inverse (P: H to E) (default as False)
        - max_Pinv_instability: allowed maximum instability value for P inverse (default as 0.005 if avoid_Pinv_instability is True)
        """

        # Hardware
        if dtype != torch.complex64 and dtype != torch.complex128:
            raise ValueError("Invalid simulation data type")
        else:
            self._dtype = dtype
        self._device = device

        # Stabilize the gradient of eigendecomposition
        self.stable_eig_grad = True if stable_eig_grad else False

        # Stability setting for inverse matrix of P and Q
        if avoid_Pinv_instability is True:
            self.avoid_Pinv_instability = True
            self.max_Pinv_instability = max_Pinv_instability
            self.Pinv_instability = []
            self.Qinv_instability = []
        else:
            self.avoid_Pinv_instability = False
            self.max_Pinv_instability = None
            self.Pinv_instability = None
            self.Qinv_instability = None

        # Simulation parameters
        self.freq = torch.as_tensor(
            freq, dtype=self._dtype, device=self._device
        )  # unit^-1
        self.omega = 2 * pi * freq  # same as k0a
        self.L = torch.as_tensor(L, dtype=self._dtype, device=self._device)

        # Fourier order
        self.order = order
        self.order_x = torch.linspace(
            -self.order[0],
            self.order[0],
            2 * self.order[0] + 1,
            dtype=torch.int64,
            device=self._device,
        )
        self.order_y = torch.linspace(
            -self.order[1],
            self.order[1],
            2 * self.order[1] + 1,
            dtype=torch.int64,
            device=self._device,
        )
        self.order_N = len(self.order_x) * len(self.order_y)

        # Lattice vector
        self.L = L  # unit
        self.Gx_norm, self.Gy_norm = 1 / (L[0] * self.freq), 1 / (L[1] * self.freq)

        # Input and output layer (Default: free space)
        self.eps_in = torch.tensor(1.0, dtype=self._dtype, device=self._device)
        self.mu_in = torch.tensor(1.0, dtype=self._dtype, device=self._device)
        self.eps_out = torch.tensor(1.0, dtype=self._dtype, device=self._device)
        self.mu_out = torch.tensor(1.0, dtype=self._dtype, device=self._device)

        # Internal layers
        self.layer_N = 0  # total number of layers
        self.thickness = []
        self.eps_conv, self.mu_conv = [], []

        # Internal layer eigenmodes
        self.P, self.Q = [], []
        self.kz_norm, self.E_eigvec, self.H_eigvec = [], [], []

        # Internal layer mode coupling coefficiencts
        self.Cf, self.Cb = [], []

        # Single layer scattering matrices
        self.layer_S11, self.layer_S21, self.layer_S12, self.layer_S22 = [], [], [], []

    def add_input_layer(self, eps=1.0, mu=1.0):
        return _add_input_layer(
            self=self,
            eps=eps,
            mu=mu
        )

    def add_output_layer(self, eps=1.0, mu=1.0):
        return _add_output_layer(
            self=self,
            eps=eps,
            mu=mu
        )

    def set_incident_angle(self, inc_ang, azi_ang, angle_layer="input"):
        return _set_incident_angle(
            self=self,
            inc_ang=inc_ang,
            azi_ang=azi_ang,
            angle_layer=angle_layer
        )

    def add_layer(self, thickness, eps=1.0, mu=1.0):
        return _add_layer(
            self=self,
            thickness=thickness,
            eps=eps,
            mu=mu
        )
            
    def solve_global_smatrix(self):
        return _solve_global_smatrix(self)

    def diffraction_angle(self, orders, *, layer="output", unit="radian"):
        return _diffraction_angle(
            self=self,
            orders=orders,
            layer=layer,
            unit=unit
        )

    def return_layer(self, layer_num, nx=100, ny=100):
        return _return_layer(
            self=self,
            layer_num=layer_num,
            nx=nx,
            ny=ny
        )

    def S_parameters(
        self,
        orders,
        direction="forward",
        port="transmission",
        polarization="xx",
        ref_order=[0, 0],
        power_norm=True,
        evanscent=1e-3
    ):
        return _S_parameters(
                self=self,
                orders=orders,
                direction=direction,
                port=port,
                polarization=polarization,
                ref_order=ref_order,
                power_norm=power_norm,
                evanscent=evanscent,
            )

    def source_planewave(
        self, *, amplitude=[1.0, 0.0], direction="forward", notation="xy"
    ):
        return _source_planewave(
            self,
            amplitude=amplitude,
            direction=direction,
            notation=notation,
        )

    def source_fourier(self, *, amplitude, orders, direction="forward", notation="xy"):
        return _source_fourier(
            self,
            amplitude=amplitude,
            orders=orders,
            direction=direction,
            notation=notation,
        )

    def field_xz(self, x_axis, z_axis, y):
        return _field_xz(
            self,
            x_axis=x_axis,
            z_axis=z_axis,
            y=y,
        )

    def field_yz(self, y_axis, z_axis, x):
        return _field_yz(
            self,
            y_axis=y_axis,
            z_axis=z_axis,
            x=x,
        )

    def field_xy(self, layer_num, x_axis, y_axis, z_prop=0.0):
        return _field_xy(
            self,
            layer_num=layer_num,
            x_axis=x_axis,
            y_axis=y_axis,
            z_prop=z_prop,
        )

    # Internal functions
    def _matching_indices(self, orders):
        return _matching_indices(
            self=self,
            orders=orders
        )

    def _kvectors(self):
        return _kvectors(
            self=self
        )

    def _material_conv(self, material):
        return _material_conv(
            self=self,
            material=material
        )

    def _eigen_decomposition_homogenous(self, eps, mu):
        return _eigen_decomposition_homogenous(
            self=self,
            eps=eps,
            mu=mu
        )

    def _eigen_decomposition(self):
        return _eigen_decomposition(
            self=self
        )

    def _solve_layer_smatrix(self):
        return _solve_layer_smatrix(
            self=self
        )

    def _RS_prod(self, Sm, Sn, Cm, Cn):
        return _RS_prod(
            self=self,
            Sm=Sm,
            Sn=Sn,
            Cm=Cm,
            Cn=Cn
        )
    