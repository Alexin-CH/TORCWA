import torch

def _source_planewave(
    self, *, amplitude=[1.0, 0.0], direction="forward", notation="xy"
):
    """
    Generate planewave

    Paramters
    - amplitude: amplitudes at the matched diffraction orders ([Ex_amp, Ey_amp] for 'xy' notation, [Ep_amp, Es_amp] for 'ps' notation)
        (list / np.ndarray / torch.Tensor) (Recommended shape: 1x2)
    - direction: incident direction ('f', 'forward' / 'b', 'backward')
    - notation: amplitude notation (xy-pol: 'xy' / ps-pol: 'ps')
    """

    self.source_fourier(
        amplitude=amplitude, orders=[0, 0], direction=direction, notation=notation
    )

def _source_fourier(self, *, amplitude, orders, direction="forward", notation="xy"):
    """
    Generate Fourier source

    Paramters
    - amplitude: amplitudes at the matched diffraction orders [([Ex_amp, Ey_amp] at orders[0]), ..., ...]
        (list / np.ndarray / torch.Tensor) (Recommended shape: Nx2)
    - orders: diffraction orders (list / np.ndarray / torch.Tensor) (Recommended shape: Nx2)
    - direction: incident direction ('f', 'forward' / 'b', 'backward')
    - notation: amplitude notation (xy-pol: 'xy' / ps-pol: 'ps')
    """
    amplitude = torch.as_tensor(
        amplitude, dtype=self._dtype, device=self._device
    ).reshape([-1, 2])
    orders = torch.as_tensor(
        orders, dtype=torch.int64, device=self._device
    ).reshape([-1, 2])

    if direction in ["f", "forward"]:
        direction = "forward"
    elif direction in ["b", "backward"]:
        direction = "backward"
    else:
        raise ValueError(
            "Invalid source direction. Set as 'forward' or 'backward'."
        )

    if notation not in ["xy", "ps"]:
        raise ValueError(
            "Invalid amplitude notation. Set as 'xy' or 'ps' notation."
        )

    # Matching indices
    order_indices = self._matching_indices(orders)

    self.source_direction = direction

    E_i = torch.zeros([2 * self.order_N, 1], dtype=self._dtype, device=self._device)
    E_i[order_indices, 0] = amplitude[:, 0]
    E_i[order_indices + self.order_N, 0] = amplitude[:, 1]

    # Convert ps-pol to xy-pol
    if notation == "ps":
        if direction == "forward":
            eps, mu = self.eps_in, self.mu_in
            sign = 1
        else:
            eps, mu = self.eps_out, self.mu_out
            sign = -1

        Kt_norm_dn = torch.sqrt(self.Kx_norm_dn**2 + self.Ky_norm_dn**2)
        Kz_norm_dn = sign * torch.abs(
            torch.real(
                torch.sqrt(eps * mu - self.Kx_norm_dn**2 - self.Ky_norm_dn**2)
            )
        )

        inc_angle = torch.atan2(torch.real(Kt_norm_dn), Kz_norm_dn)
        azi_angle = torch.atan2(
            torch.real(self.Ky_norm_dn), torch.real(self.Kx_norm_dn)
        )

        tmp1 = torch.vstack(
            (
                torch.diag(torch.cos(inc_angle) * torch.cos(azi_angle)),
                torch.diag(torch.cos(inc_angle) * torch.sin(azi_angle)),
            )
        )
        tmp2 = torch.vstack(
            (torch.diag(-torch.sin(azi_angle)), torch.diag(torch.cos(azi_angle)))
        )
        ps2xy = torch.hstack((tmp1, tmp2))

        E_i = torch.matmul(ps2xy.to(self._dtype), E_i)

    self.E_i = E_i