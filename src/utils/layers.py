import torch

def _return_layer(self, layer_num, nx=100, ny=100):
    """
    Return spatial distributions of eps and mu for the selected layer.
    The eps and mu are recovered from the trucated Fourier orders.

    Parameters
    - layer_num: selected layer (int)
    - nx: x-direction grid number (int)
    - ny: y-direction grid number (int)

    Return
    - eps_recover (torch.Tensor), mu_recover (torch.Tensor)
    """

    eps_fft = torch.zeros([nx, ny], dtype=self._dtype, device=self._device)
    mu_fft = torch.zeros([nx, ny], dtype=self._dtype, device=self._device)
    for i in range(-2 * self.order[0], 2 * self.order[0] + 1):
        for j in range(-2 * self.order[1], 2 * self.order[1] + 1):
            if i >= 0 and j >= 0:
                eps_fft[i, j] = self.eps_conv[layer_num][
                    i * (2 * self.order[1] + 1) + j, 0
                ]
                mu_fft[i, j] = self.mu_conv[layer_num][
                    i * (2 * self.order[1] + 1) + j, 0
                ]
            elif i >= 0 and j < 0:
                eps_fft[i, j] = self.eps_conv[layer_num][
                    i * (2 * self.order[1] + 1), -j
                ]
                mu_fft[i, j] = self.mu_conv[layer_num][
                    i * (2 * self.order[1] + 1), -j
                ]
            elif i < 0 and j >= 0:
                eps_fft[i, j] = self.eps_conv[layer_num][
                    j, -i * (2 * self.order[1] + 1)
                ]
                mu_fft[i, j] = self.mu_conv[layer_num][
                    j, -i * (2 * self.order[1] + 1)
                ]
            else:
                eps_fft[i, j] = self.eps_conv[layer_num][
                    0, -i * (2 * self.order[1] + 1) - j
                ]
                mu_fft[i, j] = self.mu_conv[layer_num][
                    0, -i * (2 * self.order[1] + 1) - j
                ]

    eps_recover = torch.fft.ifftn(eps_fft) * nx * ny
    mu_recover = torch.fft.ifftn(mu_fft) * nx * ny

    return eps_recover, mu_recover

def _add_layer(self, thickness, eps=1.0, mu=1.0):
    """
    Add internal layer

    Parameters
    - thickness: layer thickness (unit: length)
    - eps: relative permittivity
    - mu: relative permeability
    """

    is_eps_homogenous = (
        isinstance(eps, float)
        or isinstance(eps, complex)
        or (eps.dim() == 0)
        or ((eps.dim() == 1) and eps.shape[0] == 1)
    )
    is_mu_homogenous = (
        isinstance(mu, float)
        or isinstance(mu, float)
        or (mu.dim() == 0)
        or ((mu.dim() == 1) and mu.shape[0] == 1)
    )

    self.eps_conv.append(
        eps * torch.eye(self.order_N, dtype=self._dtype, device=self._device)
        if is_eps_homogenous
        else self._material_conv(eps)
    )
    self.mu_conv.append(
        mu * torch.eye(self.order_N, dtype=self._dtype, device=self._device)
        if is_mu_homogenous
        else self._material_conv(mu)
    )

    self.layer_N += 1
    self.thickness.append(thickness)

    if is_eps_homogenous and is_mu_homogenous:
        self._eigen_decomposition_homogenous(eps, mu)
    else:
        self._eigen_decomposition()

    self._solve_layer_smatrix()

def _add_input_layer(self, eps=1.0, mu=1.0):
    """
    Add input layer
    - If this function is not used, simulation will be performed under free space input layer.

    Parameters
    - eps: relative permittivity
    - mu: relative permeability
    """

    self.eps_in = torch.as_tensor(eps, dtype=self._dtype, device=self._device)
    self.mu_in = torch.as_tensor(mu, dtype=self._dtype, device=self._device)
    self.Sin = []

def _add_output_layer(self, eps=1.0, mu=1.0):
    """
    Add output layer
    - If this function is not used, simulation will be performed under free space output layer.

    Parameters
    - eps: relative permittivity
    - mu: relative permeability
    """

    self.eps_out = torch.as_tensor(eps, dtype=self._dtype, device=self._device)
    self.mu_out = torch.as_tensor(mu, dtype=self._dtype, device=self._device)
    self.Sout = []

def _material_conv(self, material):
    material_N = material.shape[0] * material.shape[1]

    # Matching indices
    order_x_grid, order_y_grid = torch.meshgrid(
        self.order_x, self.order_y, indexing="ij"
    )
    ox = order_x_grid.to(torch.int64).reshape([-1])
    oy = order_y_grid.to(torch.int64).reshape([-1])

    ind = torch.arange(len(self.order_x) * len(self.order_y), device=self._device)
    indx, indy = torch.meshgrid(
        ind.to(torch.int64), ind.to(torch.int64), indexing="ij"
    )

    material_fft = torch.fft.fft2(material) / material_N

    material_fft_real = torch.real(material_fft)
    material_fft_imag = torch.imag(material_fft)

    material_convmat_real = material_fft_real[
        ox[indx] - ox[indy], oy[indx] - oy[indy]
    ]
    material_convmat_imag = material_fft_imag[
        ox[indx] - ox[indy], oy[indx] - oy[indy]
    ]

    material_convmat = torch.complex(material_convmat_real, material_convmat_imag)

    return material_convmat