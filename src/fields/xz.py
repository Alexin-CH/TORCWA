import torch

def _field_xz(self, x_axis, z_axis, y):
    """
    XZ-plane field distribution.
    Returns the field at the specific y point.

    Paramters
    - x_axis: x-direction sampling coordinates (torch.Tensor)
    - z_axis: z-direction sampling coordinates (torch.Tensor)
    - y: selected y point

    Return
    - [Ex, Ey, Ez] (list[torch.Tensor]), [Hx, Hy, Hz] (list[torch.Tensor])
    """

    if not isinstance(x_axis, torch.Tensor) or not isinstance(z_axis, torch.Tensor):
        raise TypeError("x and z axis must be torch.Tensor type.")

    x_axis = x_axis.reshape([-1, 1, 1])

    Kx_norm, Ky_norm = self.Kx_norm, self.Ky_norm

    Ex_split, Ey_split, Ez_split = [], [], []
    Hx_split, Hy_split, Hz_split = [], [], []

    # layer number
    zp = torch.zeros(len(self.thickness), device=self._device)
    zm = torch.zeros(len(self.thickness), device=self._device)
    layer_num = torch.zeros([len(z_axis)], dtype=torch.int64, device=self._device)
    layer_num[z_axis < 0.0] = -1

    for ti in range(len(self.thickness)):
        zp[ti:] += self.thickness[ti]
    zm[1:] = zp[0:-1]

    for bi in range(len(zp)):
        layer_num[z_axis > zp[bi]] += 1

    prev_layer_num = -2
    for zi in range(len(z_axis)):
        # Input and output layers
        if layer_num[zi] == -1 or layer_num[zi] == self.layer_N:
            Kx_norm_dn = self.Kx_norm_dn
            Ky_norm_dn = self.Ky_norm_dn

            if layer_num[zi] == -1:
                z_prop = z_axis[zi] if z_axis[zi] <= 0.0 else 0.0
                if layer_num[zi] != prev_layer_num:
                    eps = self.eps_in if hasattr(self, "eps_in") else 1.0
                    mu = self.mu_in if hasattr(self, "mu_in") else 1.0
                    Vi = self.Vi if hasattr(self, "Vi") else self.Vf
                    Kz_norm_dn = torch.sqrt(
                        eps * mu - Kx_norm_dn**2 - Ky_norm_dn**2
                    )
                    Kz_norm_dn = torch.where(
                        torch.imag(Kz_norm_dn) > 0,
                        torch.conj(Kz_norm_dn),
                        Kz_norm_dn,
                    ).reshape([-1, 1])
                    Kz_norm_dn = torch.vstack((Kz_norm_dn, Kz_norm_dn))
            elif layer_num[zi] == self.layer_N:
                if len(zp) == 0:
                    z_prop = z_axis[zi]
                else:
                    z_prop = (
                        z_axis[zi] - zp[-1] if z_axis[zi] - zp[-1] >= 0.0 else 0.0
                    )
                if layer_num[zi] != prev_layer_num:
                    eps = self.eps_out if hasattr(self, "eps_in") else 1.0
                    mu = self.mu_out if hasattr(self, "mu_in") else 1.0
                    Vo = self.Vo if hasattr(self, "Vo") else self.Vf
                    Kz_norm_dn = torch.sqrt(
                        eps * mu - Kx_norm_dn**2 - Ky_norm_dn**2
                    )
                    Kz_norm_dn = torch.where(
                        torch.imag(Kz_norm_dn) < 0,
                        torch.conj(Kz_norm_dn),
                        Kz_norm_dn,
                    ).reshape([-1, 1])
                    Kz_norm_dn = torch.vstack((Kz_norm_dn, Kz_norm_dn))

            # Phase
            z_phase = torch.exp(1.0j * self.omega * Kz_norm_dn * z_prop)

            # Fourier domain fields
            # [diffraction order]
            if layer_num[zi] == -1 and self.source_direction == "forward":
                Exy_p = self.E_i * z_phase
                Hxy_p = torch.matmul(Vi, Exy_p)
                Exy_m = torch.matmul(self.S[1], self.E_i) * torch.conj(z_phase)
                Hxy_m = torch.matmul(-Vi, Exy_m)
            elif layer_num[zi] == -1 and self.source_direction == "backward":
                Exy_p = torch.zeros_like(self.E_i)
                Hxy_p = torch.zeros_like(self.E_i)
                Exy_m = torch.matmul(self.S[3], self.E_i) * torch.conj(z_phase)
                Hxy_m = torch.matmul(-Vi, Exy_m)
            elif (
                layer_num[zi] == self.layer_N and self.source_direction == "forward"
            ):
                Exy_p = torch.matmul(self.S[0], self.E_i) * z_phase
                Hxy_p = torch.matmul(Vo, Exy_p)
                Exy_m = torch.zeros_like(self.E_i)
                Hxy_m = torch.zeros_like(self.E_i)
            elif (
                layer_num[zi] == self.layer_N
                and self.source_direction == "backward"
            ):
                Exy_p = torch.matmul(self.S[2], self.E_i) * z_phase
                Hxy_p = torch.matmul(Vo, Exy_p)
                Exy_m = self.E_i * torch.conj(z_phase)
                Hxy_m = torch.matmul(-Vo, Exy_m)

            Ex_mn = Exy_p[: self.order_N] + Exy_m[: self.order_N]
            Ey_mn = Exy_p[self.order_N :] + Exy_m[self.order_N :]
            Hz_mn = (
                torch.matmul(Kx_norm, Ey_mn) / mu
                - torch.matmul(Ky_norm, Ex_mn) / mu
            )
            Hx_mn = Hxy_p[: self.order_N] + Hxy_m[: self.order_N]
            Hy_mn = Hxy_p[self.order_N :] + Hxy_m[self.order_N :]
            Ez_mn = (
                torch.matmul(Ky_norm, Hx_mn) / eps
                - torch.matmul(Kx_norm, Hy_mn) / eps
            )

            # Spatial domain fields
            xy_phase = torch.exp(
                1.0j * self.omega * (self.Kx_norm_dn * x_axis + self.Ky_norm_dn * y)
            )
            Ex_split.append(torch.sum(Ex_mn.reshape(1, 1, -1) * xy_phase, dim=2))
            Ey_split.append(torch.sum(Ey_mn.reshape(1, 1, -1) * xy_phase, dim=2))
            Ez_split.append(torch.sum(Ez_mn.reshape(1, 1, -1) * xy_phase, dim=2))
            Hx_split.append(torch.sum(Hx_mn.reshape(1, 1, -1) * xy_phase, dim=2))
            Hy_split.append(torch.sum(Hy_mn.reshape(1, 1, -1) * xy_phase, dim=2))
            Hz_split.append(torch.sum(Hz_mn.reshape(1, 1, -1) * xy_phase, dim=2))

        # Internal layers
        else:
            z_prop = z_axis[zi] - zm[layer_num[zi]]

            if layer_num[zi] != prev_layer_num:
                if self.source_direction == "forward":
                    C = torch.matmul(self.C[0][layer_num[zi]], self.E_i)
                elif self.source_direction == "backward":
                    C = torch.matmul(self.C[1][layer_num[zi]], self.E_i)

                kz_norm = self.kz_norm[layer_num[zi]]
                E_eigvec = self.E_eigvec[layer_num[zi]]
                H_eigvec = self.H_eigvec[layer_num[zi]]

                Cp = torch.diag(C[: 2 * self.order_N, 0])
                Cm = torch.diag(C[2 * self.order_N :, 0])

                eps_conv_inv = torch.linalg.inv(self.eps_conv[layer_num[zi]])
                mu_conv_inv = torch.linalg.inv(self.mu_conv[layer_num[zi]])

            # Phase
            z_phase_p = torch.diag(torch.exp(1.0j * self.omega * kz_norm * z_prop))
            z_phase_m = torch.diag(
                torch.exp(
                    1.0j
                    * self.omega
                    * kz_norm
                    * (self.thickness[layer_num[zi]] - z_prop)
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

            Ex_mn = torch.sum(
                torch.matmul(Ex_p, Cp) + torch.matmul(Ex_m, Cm), dim=1
            )
            Ey_mn = torch.sum(
                torch.matmul(Ey_p, Cp) + torch.matmul(Ey_m, Cm), dim=1
            )
            Ez_mn = torch.sum(
                torch.matmul(Ez_p, Cp) + torch.matmul(Ez_m, Cm), dim=1
            )
            Hx_mn = torch.sum(
                torch.matmul(Hx_p, Cp) + torch.matmul(Hx_m, Cm), dim=1
            )
            Hy_mn = torch.sum(
                torch.matmul(Hy_p, Cp) + torch.matmul(Hy_m, Cm), dim=1
            )
            Hz_mn = torch.sum(
                torch.matmul(Hz_p, Cp) + torch.matmul(Hz_m, Cm), dim=1
            )

            # Spatial domain fields
            xy_phase = torch.exp(
                1.0j * self.omega * (self.Kx_norm_dn * x_axis + self.Ky_norm_dn * y)
            )
            Ex_split.append(torch.sum(Ex_mn.reshape(1, 1, -1) * xy_phase, dim=2))
            Ey_split.append(torch.sum(Ey_mn.reshape(1, 1, -1) * xy_phase, dim=2))
            Ez_split.append(torch.sum(Ez_mn.reshape(1, 1, -1) * xy_phase, dim=2))
            Hx_split.append(torch.sum(Hx_mn.reshape(1, 1, -1) * xy_phase, dim=2))
            Hy_split.append(torch.sum(Hy_mn.reshape(1, 1, -1) * xy_phase, dim=2))
            Hz_split.append(torch.sum(Hz_mn.reshape(1, 1, -1) * xy_phase, dim=2))

        prev_layer_num = layer_num[zi]

    Ex = torch.cat(Ex_split, dim=1)
    Ey = torch.cat(Ey_split, dim=1)
    Ez = torch.cat(Ez_split, dim=1)
    Hx = torch.cat(Hx_split, dim=1)
    Hy = torch.cat(Hy_split, dim=1)
    Hz = torch.cat(Hz_split, dim=1)

    return [Ex, Ey, Ez], [Hx, Hy, Hz]