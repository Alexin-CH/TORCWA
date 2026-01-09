import torch
from math import pi

from .mixins import LayerMixin, FieldMixin, SMatrixMixin, UtilsMixin

class rcwa(LayerMixin, FieldMixin, SMatrixMixin, UtilsMixin):
    # Simulation setting
    def __init__(
        self,
        freq,
        order,
        lattice,
        *,
        dtype=torch.complex64,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        stable_eig_grad=True,
        avoid_pinv_instability=False,
        max_pinv_instability=0.005
    ):
        """
        Rigorous Coupled Wave Analysis
        - Lorentz-Heaviside units
        - Speed of light: 1
        - Time harmonics notation: exp(-jωt)

        Parameters
        - freq: simulation frequency (unit: length^-1)
        - order: Fourier order [x_order (int), y_order (int)]
        - lattice: Lattice constant [Lx, Ly] (unit: length)

        Keyword Parameters
        - dtype: simulation data type (only torch.complex64 and torch.complex128 are allowed.)
        - device: simulation device (only torch.device('cpu') and torch.device('cuda') are allowed.)
        - stable_eig_grad: stabilize gradient calculation of eigendecompsition (default as True)
        - avoid_pinv_instability: avoid instability of P inverse (P: H to E) (default as False)
        - max_pinv_instability: allowed maximum instability value for P inverse (default as 0.005 if avoid_pinv_instability is True)
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
        if avoid_pinv_instability is True:
            self.avoid_Pinv_instability = True
            self.max_Pinv_instability = max_pinv_instability
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
        self.L = torch.as_tensor(lattice, dtype=self._dtype, device=self._device)

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
        self.L = lattice  # unit
        self.Gx_norm, self.Gy_norm = 1 / (self.L[0] * self.freq), 1 / (self.L[1] * self.freq)

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
        