import torch

def _field_xy(self, layer_num, x_axis, y_axis, z_prop=0.0):
    """
    XY-plane field distribution at the selected layer.
    Returns the field at z_prop away from the lower boundary of the layer.
    For the input layer, z_prop is the distance from the upper boundary and should be negative (calculate z_prop=0 if positive value is entered).

    Parameters
    - layer_num: selected layer (int)
    - x_axis: x-direction sampling coordinates (torch.Tensor)
    - y_axis: y-direction sampling coordinates (torch.Tensor)
    - z_prop: z-direction distance from the lower boundary of the layer (layer_num>-1),
        or the distance from the upper boundary of the layer and should be negative (layer_num=-1).

    Return
    - [Ex, Ey, Ez] (list[torch.Tensor]), [Hx, Hy, Hz] (list[torch.Tensor])
    """

    if not isinstance(layer_num, int):
        raise TypeError('Parameter "layer_num" must be int type.')

    if layer_num < -1 or layer_num > self.layer_N:
        raise IndexError("Layer number is out of range.")

    if not isinstance(x_axis, torch.Tensor) or not isinstance(y_axis, torch.Tensor):
        raise TypeError("x and y axis must be torch.Tensor type.")

    # [x, y, diffraction order]
    x_axis = x_axis.reshape([-1, 1, 1])
    y_axis = y_axis.reshape([1, -1, 1])

    Kx_norm, Ky_norm = self.Kx_norm, self.Ky_norm

    # Input and output layers
    if layer_num == -1 or layer_num == self.layer_N:
        Kx_norm_dn, Ky_norm_dn = self.Kx_norm_dn, self.Ky_norm_dn

        if layer_num == -1:
            z_prop = z_prop if z_prop <= 0.0 else 0.0
            eps = self.eps_in if hasattr(self, "eps_in") else 1.0
            mu = self.mu_in if hasattr(self, "mu_in") else 1.0
            Vi = self.Vi if hasattr(self, "Vi") else self.Vf
            Kz_norm_dn = torch.sqrt(eps * mu - Kx_norm_dn**2 - Ky_norm_dn**2)
            Kz_norm_dn = torch.where(
                torch.imag(Kz_norm_dn) > 0, torch.conj(Kz_norm_dn), Kz_norm_dn
            ).reshape([-1, 1])
        elif layer_num == self.layer_N:
            z_prop = z_prop if z_prop >= 0.0 else 0.0
            eps = self.eps_out if hasattr(self, "eps_in") else 1.0
            mu = self.mu_out if hasattr(self, "mu_in") else 1.0
            Vo = self.Vo if hasattr(self, "Vo") else self.Vf
            Kz_norm_dn = torch.sqrt(eps * mu - Kx_norm_dn**2 - Ky_norm_dn**2)
            Kz_norm_dn = torch.where(
                torch.imag(Kz_norm_dn) < 0, torch.conj(Kz_norm_dn), Kz_norm_dn
            ).reshape([-1, 1])

        # Phase
        Kz_norm_dn = torch.vstack((Kz_norm_dn, Kz_norm_dn))
        z_phase = torch.exp(1.0j * self.omega * Kz_norm_dn * z_prop)

        # Fourier domain fields
        # [diffraction order, diffraction order]
        if layer_num == -1 and self.source_direction == "forward":
            Exy_p = self.E_i * z_phase
            Hxy_p = torch.matmul(Vi, Exy_p)
            Exy_m = torch.matmul(self.S[1], self.E_i) * torch.conj(z_phase)
            Hxy_m = torch.matmul(-Vi, Exy_m)
        elif layer_num == -1 and self.source_direction == "backward":
            Exy_p = torch.zeros_like(self.E_i)
            Hxy_p = torch.zeros_like(self.E_i)
            Exy_m = torch.matmul(self.S[3], self.E_i) * torch.conj(z_phase)
            Hxy_m = torch.matmul(-Vi, Exy_m)
        elif layer_num == self.layer_N and self.source_direction == "forward":
            Exy_p = torch.matmul(self.S[0], self.E_i) * z_phase
            Hxy_p = torch.matmul(Vo, Exy_p)
            Exy_m = torch.zeros_like(self.E_i)
            Hxy_m = torch.zeros_like(self.E_i)
        elif layer_num == self.layer_N and self.source_direction == "backward":
            Exy_p = torch.matmul(self.S[2], self.E_i) * z_phase
            Hxy_p = torch.matmul(Vo, Exy_p)
            Exy_m = self.E_i * torch.conj(z_phase)
            Hxy_m = torch.matmul(-Vo, Exy_m)

        Ex_mn = Exy_p[: self.order_N] + Exy_m[: self.order_N]
        Ey_mn = Exy_p[self.order_N :] + Exy_m[self.order_N :]
        Hz_mn = (
            torch.matmul(Kx_norm, Ey_mn) / mu - torch.matmul(Ky_norm, Ex_mn) / mu
        )
        Hx_mn = Hxy_p[: self.order_N] + Hxy_m[: self.order_N]
        Hy_mn = Hxy_p[self.order_N :] + Hxy_m[self.order_N :]
        Ez_mn = (
            torch.matmul(Ky_norm, Hx_mn) / eps - torch.matmul(Kx_norm, Hy_mn) / eps
        )

        # Spatial domain fields
        xy_phase = torch.exp(
            1.0j
            * self.omega
            * (self.Kx_norm_dn * x_axis + self.Ky_norm_dn * y_axis)
        )
        Ex = torch.sum(Ex_mn.reshape(1, 1, -1) * xy_phase, dim=2)
        Ey = torch.sum(Ey_mn.reshape(1, 1, -1) * xy_phase, dim=2)
        Ez = torch.sum(Ez_mn.reshape(1, 1, -1) * xy_phase, dim=2)
        Hx = torch.sum(Hx_mn.reshape(1, 1, -1) * xy_phase, dim=2)
        Hy = torch.sum(Hy_mn.reshape(1, 1, -1) * xy_phase, dim=2)
        Hz = torch.sum(Hz_mn.reshape(1, 1, -1) * xy_phase, dim=2)

    # Internal layers
    else:
        if self.source_direction == "forward":
            C = torch.matmul(self.C[0][layer_num], self.E_i)
        elif self.source_direction == "backward":
            C = torch.matmul(self.C[1][layer_num], self.E_i)

        kz_norm = self.kz_norm[layer_num]
        E_eigvec = self.E_eigvec[layer_num]
        H_eigvec = self.H_eigvec[layer_num]

        Cp = torch.diag(C[: 2 * self.order_N, 0])
        Cm = torch.diag(C[2 * self.order_N :, 0])

        eps_conv_inv = torch.linalg.inv(self.eps_conv[layer_num])
        mu_conv_inv = torch.linalg.inv(self.mu_conv[layer_num])

        # Phase
        z_phase_p = torch.diag(torch.exp(1.0j * self.omega * kz_norm * z_prop))
        z_phase_m = torch.diag(
            torch.exp(
                1.0j * self.omega * kz_norm * (self.thickness[layer_num] - z_prop)
            )
        )

        # Fourier domain fields
        # [diffraction order, eigenmode number]
        Exy_p = torch.matmul(E_eigvec, z_phase_p)
        Ex_p = Exy_p[: self.order_N, :]
        Ey_p = Exy_p[self.order_N :, :]
        Hz_p = torch.matmul(
            mu_conv_inv, torch.matmul(Kx_norm, Ey_p)
        ) - torch.matmul(mu_conv_inv, torch.matmul(Ky_norm, Ex_p))
        Exy_m = torch.matmul(E_eigvec, z_phase_m)
        Ex_m = Exy_m[: self.order_N, :]
        Ey_m = Exy_m[self.order_N :, :]
        Hz_m = torch.matmul(
            mu_conv_inv, torch.matmul(Kx_norm, Ey_m)
        ) - torch.matmul(mu_conv_inv, torch.matmul(Ky_norm, Ex_m))
        Hxy_p = torch.matmul(H_eigvec, z_phase_p)
        Hx_p = Hxy_p[: self.order_N, :]
        Hy_p = Hxy_p[self.order_N :, :]
        Ez_p = torch.matmul(
            eps_conv_inv, torch.matmul(Ky_norm, Hx_p)
        ) - torch.matmul(eps_conv_inv, torch.matmul(Kx_norm, Hy_p))
        Hxy_m = torch.matmul(-H_eigvec, z_phase_m)
        Hx_m = Hxy_m[: self.order_N, :]
        Hy_m = Hxy_m[self.order_N :, :]
        Ez_m = torch.matmul(
            eps_conv_inv, torch.matmul(Ky_norm, Hx_m)
        ) - torch.matmul(eps_conv_inv, torch.matmul(Kx_norm, Hy_m))

        Ex_mn = torch.sum(torch.matmul(Ex_p, Cp) + torch.matmul(Ex_m, Cm), dim=1)
        Ey_mn = torch.sum(torch.matmul(Ey_p, Cp) + torch.matmul(Ey_m, Cm), dim=1)
        Ez_mn = torch.sum(torch.matmul(Ez_p, Cp) + torch.matmul(Ez_m, Cm), dim=1)
        Hx_mn = torch.sum(torch.matmul(Hx_p, Cp) + torch.matmul(Hx_m, Cm), dim=1)
        Hy_mn = torch.sum(torch.matmul(Hy_p, Cp) + torch.matmul(Hy_m, Cm), dim=1)
        Hz_mn = torch.sum(torch.matmul(Hz_p, Cp) + torch.matmul(Hz_m, Cm), dim=1)

        # Spatial domain fields
        xy_phase = torch.exp(
            1.0j
            * self.omega
            * (self.Kx_norm_dn * x_axis + self.Ky_norm_dn * y_axis)
        )
        Ex = torch.sum(Ex_mn.reshape(1, 1, -1) * xy_phase, dim=2)
        Ey = torch.sum(Ey_mn.reshape(1, 1, -1) * xy_phase, dim=2)
        Ez = torch.sum(Ez_mn.reshape(1, 1, -1) * xy_phase, dim=2)
        Hx = torch.sum(Hx_mn.reshape(1, 1, -1) * xy_phase, dim=2)
        Hy = torch.sum(Hy_mn.reshape(1, 1, -1) * xy_phase, dim=2)
        Hz = torch.sum(Hz_mn.reshape(1, 1, -1) * xy_phase, dim=2)

    return [Ex, Ey, Ez], [Hx, Hy, Hz]